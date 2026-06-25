// Fast n-gram contamination scanner (C++ / pybind11).
//
// The hot path of the ngram_overlap check: given a large reference corpus and
// many benchmark questions, find what fraction of each question's distinct
// n-grams also appear in the corpus. In pure Python this is dominated by
// per-token hashing and set operations over potentially tens of millions of
// n-grams. This module does the same work with:
//   - a single rolling hash over whitespace-tokenized text (FNV-1a based),
//   - n-grams stored as 64-bit hashes in one flat std::unordered_set,
//   - zero per-n-gram Python object allocation.
//
// It reports throughput (n-grams/second) so the speedup is measurable.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

// FNV-1a 64-bit hash of a byte range.
inline uint64_t fnv1a(const char* data, size_t len, uint64_t seed = 1469598103934665603ULL) {
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

// Lowercase + split on non-alphanumeric, returning per-token hashes. We hash
// tokens individually then combine into n-gram hashes, so we never materialize
// n-gram strings.
std::vector<uint64_t> token_hashes(const std::string& text) {
    std::vector<uint64_t> out;
    out.reserve(text.size() / 4 + 1);
    size_t i = 0, n = text.size();
    while (i < n) {
        // Skip non-word characters.
        while (i < n && !std::isalnum(static_cast<unsigned char>(text[i]))) ++i;
        size_t start = i;
        // Accumulate a word, lowercased into a small buffer.
        std::string word;
        while (i < n && std::isalnum(static_cast<unsigned char>(text[i]))) {
            word.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(text[i]))));
            ++i;
        }
        if (i > start) out.push_back(fnv1a(word.data(), word.size()));
    }
    return out;
}

// Combine consecutive token hashes into n-gram hashes (order-sensitive).
void ngram_hashes_into(const std::vector<uint64_t>& toks, int nval,
                       std::unordered_set<uint64_t>& dst) {
    if ((int)toks.size() < nval) {
        if (!toks.empty()) {
            uint64_t h = 1469598103934665603ULL;
            for (uint64_t t : toks) { h ^= t; h *= 1099511628211ULL; }
            dst.insert(h);
        }
        return;
    }
    for (size_t i = 0; i + nval <= toks.size(); ++i) {
        uint64_t h = 1469598103934665603ULL;
        for (int k = 0; k < nval; ++k) { h ^= toks[i + k]; h *= 1099511628211ULL; }
        dst.insert(h);
    }
}

}  // namespace

// Build the corpus n-gram hash set. Returns an opaque handle (the set wrapped
// in a capsule-like vector) plus the count, and records build throughput.
class NgramIndex {
public:
    NgramIndex(const std::vector<std::string>& corpus, int n) : n_(n) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t total = 0;
        for (const auto& doc : corpus) {
            auto toks = token_hashes(doc);
            size_t before = set_.size();
            ngram_hashes_into(toks, n_, set_);
            total += toks.size() >= (size_t)n_ ? toks.size() - n_ + 1 : (toks.empty() ? 0 : 1);
            (void)before;
        }
        auto t1 = std::chrono::steady_clock::now();
        build_seconds_ = std::chrono::duration<double>(t1 - t0).count();
        ngrams_processed_ = total;
    }

    // Fraction of the query's distinct n-grams found in the corpus.
    double overlap(const std::string& query) const {
        auto toks = token_hashes(query);
        std::unordered_set<uint64_t> q;
        ngram_hashes_into(toks, n_, q);
        if (q.empty()) return 0.0;
        size_t matched = 0;
        for (uint64_t h : q) if (set_.count(h)) ++matched;
        return static_cast<double>(matched) / static_cast<double>(q.size());
    }

    // Batch version: scan many queries, return overlaps + throughput stats.
    py::dict scan(const std::vector<std::string>& queries) const {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<double> scores;
        scores.reserve(queries.size());
        uint64_t total_ngrams = 0;
        for (const auto& qy : queries) {
            auto toks = token_hashes(qy);
            std::unordered_set<uint64_t> q;
            ngram_hashes_into(toks, n_, q);
            total_ngrams += q.size();
            if (q.empty()) { scores.push_back(0.0); continue; }
            size_t matched = 0;
            for (uint64_t h : q) if (set_.count(h)) ++matched;
            scores.push_back((double)matched / (double)q.size());
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        py::dict out;
        out["scores"] = scores;
        out["query_ngrams"] = total_ngrams;
        out["scan_seconds"] = secs;
        out["ngrams_per_second"] = secs > 0 ? total_ngrams / secs : 0.0;
        return out;
    }

    size_t size() const { return set_.size(); }
    double build_seconds() const { return build_seconds_; }
    uint64_t ngrams_processed() const { return ngrams_processed_; }

private:
    int n_;
    std::unordered_set<uint64_t> set_;
    double build_seconds_ = 0.0;
    uint64_t ngrams_processed_ = 0;
};

PYBIND11_MODULE(ngram_scan, m) {
    m.doc() = "Fast n-gram contamination scanner (C++).";
    py::class_<NgramIndex>(m, "NgramIndex")
        .def(py::init<const std::vector<std::string>&, int>(), py::arg("corpus"), py::arg("n"))
        .def("overlap", &NgramIndex::overlap, py::arg("query"))
        .def("scan", &NgramIndex::scan, py::arg("queries"))
        .def("size", &NgramIndex::size)
        .def("build_seconds", &NgramIndex::build_seconds)
        .def("ngrams_processed", &NgramIndex::ngrams_processed);
}
