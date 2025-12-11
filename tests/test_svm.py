import numpy as np
from sklearn.svm import SVC
from src.svm import SVM


def test_svm_basic():
    np.random.seed(42)
    X = np.array([
        [1, 2], [2, 3], [2, 1],
        [6, 5], [7, 8], [8, 7]
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    
    my_svm = SVM(learning_rate=0.001, epochs=1000, C=1.0)
    my_svm.fit(X, y)
    my_pred = my_svm.predict(X)
    
    assert my_pred.shape == y.shape
    assert my_svm.score(X, y) >= 0.8


def test_svm_linearly_separable():
    np.random.seed(42)
    X_class1 = np.random.randn(50, 2) + np.array([2, 2])
    X_class2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack([X_class1, X_class2])
    y = np.array([1] * 50 + [-1] * 50)
    
    idx = np.random.permutation(100)
    X, y = X[idx], y[idx]
    
    my_svm = SVM(learning_rate=0.01, epochs=500, C=1.0)
    my_svm.fit(X, y)
    
    assert my_svm.score(X, y) >= 0.9


def test_svm_fit_then_predict():
    np.random.seed(42)
    X_train = np.random.randn(80, 10)
    y_train = np.where(X_train[:, 0] > 0, 1, -1)
    
    X_test = np.random.randn(20, 10)
    y_test = np.where(X_test[:, 0] > 0, 1, -1)
    
    my_svm = SVM(learning_rate=0.01, epochs=500, C=1.0)
    my_svm.fit(X_train, y_train)
    predictions = my_svm.predict(X_test)
    
    assert predictions.shape == y_test.shape
    assert my_svm.score(X_test, y_test) >= 0.7


def test_svm_vs_sklearn():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    
    my_svm = SVM(learning_rate=0.01, epochs=1000, C=1.0)
    my_svm.fit(X, y)
    my_accuracy = my_svm.score(X, y)
    
    sk_svm = SVC(kernel='linear', C=1.0)
    sk_svm.fit(X, y)
    sk_accuracy = sk_svm.score(X, y)
    
    assert abs(my_accuracy - sk_accuracy) < 0.15