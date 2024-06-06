import sys

try:
    from skbuild import setup
    import nanobind
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

setup(
    name="PyRobustPCA",
    version="0.0.2",
    packages=["PyRobustPCA"],
    package_dir={"": "src"},
    cmake_args=["-DCMAKE_BUILD_TYPE=Release"],
    cmake_install_dir="src/PyRobustPCA",
    include_package_data=True,
    python_requires=">=3.8",
)
