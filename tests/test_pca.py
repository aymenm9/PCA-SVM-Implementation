import numpy as np
from sklearn.decomposition import PCA as SKPCA
from src.pca import PCA


def test_pca_basic():
    np.random.seed(42)
    X = np.random.randn(100, 50)
    
    my_pca = PCA(n_components=10)
    my_result = my_pca.fit_transform(X)
    
    sk_pca = SKPCA(n_components=10)
    sk_result = sk_pca.fit_transform(X)
    
    assert my_result.shape == sk_result.shape
    
    for i in range(10):
        my_col = my_result[:, i]
        sk_col = sk_result[:, i]
        assert np.allclose(my_col, sk_col, rtol=1e-3) or np.allclose(my_col, -sk_col, rtol=1e-3)


def test_pca_fit_then_transform():
    np.random.seed(42)
    X_train = np.random.randn(80, 50)
    X_test = np.random.randn(20, 50)
    
    my_pca = PCA(n_components=10)
    my_pca.fit(X_train)
    my_result = my_pca.transform(X_test)
    
    sk_pca = SKPCA(n_components=10)
    sk_pca.fit(X_train)
    sk_result = sk_pca.transform(X_test)
    
    assert my_result.shape == sk_result.shape


def test_pca_variance():
    np.random.seed(42)
    X = np.random.randn(100, 50)
    
    my_pca = PCA(n_components=10)
    my_pca.fit(X)
    
    sk_pca = SKPCA(n_components=10)
    sk_pca.fit(X)
    
    my_var = my_pca.eigenvalues[:10]
    sk_var = sk_pca.explained_variance_
    
    assert np.allclose(my_var, sk_var, rtol=1e-3)