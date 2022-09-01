from PyRobustPCA import VanillaPCA

pca = VanillaPCA()
import numpy as np

A = np.random.rand(90).reshape(-1, 3)
pca.fit(A)
