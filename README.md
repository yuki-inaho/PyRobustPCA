# Dependencies

The following dependencies are required to build and run the project:

- CMake (version 3.18 - 3.22)
- Python (version >= 3.8)
- Eigen3
- Boost


# Installation

```
git clone https://github.com/yuki-inaho/PyRobustPCA
cd PyRobustPCA/PyRobustPCA
pip install .
```

or

```
pip install "git+https://github.com/yuki-inaho/PyRobustPCA.git#egg=PyRobustPCA&subdirectory=PyRobustPCA"
```

# Usage

[pca.ipynb](https://github.com/yuki-inaho/PyRobustPCA/blob/main/example/pca.ipynb)

# Running Tests

To run the tests, make sure you have `pytest` installed. You can install it using pip:

```bash
pip install pytest
```

After installing pytest, you can run the tests with the following command:

```bash
pytest test/test_pyrobustpca.py
```

# References

- Hubert, M., Rousseeuw, P. J., and Vanden Branden, K. (2005), “ROBPCA: A New Approach to
  Robust Principal Component Analysis,” Technometrics, 47, 64–79.
- Hubert, M., P. J. Rousseeuw, and T. Verdonck (2012), "A deterministic
  algorithm for robust location and scatter" Journal of Computational and
  Graphical Statistics 21, 618–637.
