import numpy as np

class PCA:
    def __init__(self, n_components:int):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.principal_components = None

    def fit(self, X: np.ndarray):

        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        covariance_matrix = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:,idx]

        self.principal_components = self.eigenvectors[:,:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        return X_centered @ self.principal_components
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
