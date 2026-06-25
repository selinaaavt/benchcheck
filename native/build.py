"""Compile the C++ ngram_scan extension in-place.

    python native/build.py

Produces ngram_scan*.so in the repo root so `import ngram_scan` works. The
Python ngram_overlap check uses it when present and falls back to pure Python
otherwise, so the build is optional.
"""
import subprocess
import sys
from pathlib import Path

import pybind11

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "native" / "ngram_scan.cpp"


def main() -> None:
    ext_suffix = subprocess.check_output(
        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"]
    ).decode().strip()
    out = ROOT / f"ngram_scan{ext_suffix}"
    cmd = [
        "g++", "-O3", "-Wall", "-shared", "-std=c++17", "-fPIC",
        f"-I{pybind11.get_include()}",
        *subprocess.check_output(
            [sys.executable, "-c",
             "import sysconfig;print(sysconfig.get_path('include'))"]
        ).decode().split(),
        str(SRC),
        "-o", str(out),
    ]
    # Add the python include path properly.
    py_inc = subprocess.check_output(
        [sys.executable, "-c", "import sysconfig;print(sysconfig.get_path('include'))"]
    ).decode().strip()
    cmd = [
        "g++", "-O3", "-Wall", "-shared", "-std=c++17", "-fPIC",
        f"-I{pybind11.get_include()}", f"-I{py_inc}",
        str(SRC), "-o", str(out),
    ]
    print("compiling:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("built", out.name)


if __name__ == "__main__":
    main()
