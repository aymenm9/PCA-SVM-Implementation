import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from src.svm import SVM     # your implementation


def test_svm_vs_sklearn_real_split():
    np.random.seed(42)

    
    X = np.random.randn(300, 5)
    y = np.where(X[:, 0] + 0.3 * X[:, 1] > 0, 1, -1)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    my_svm = SVM(learning_rate=0.01, epochs=800, C=1.0)
    my_svm.fit(X_train, y_train)
    my_pred = my_svm.predict(X_test)
    my_acc = accuracy_score(y_test, my_pred)

   
    sk_svm = LinearSVC(C=1.0)
    sk_svm.fit(X_train, y_train)
    sk_pred = sk_svm.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)

    print("My SVM accuracy:      ", my_acc)
    print("Sklearn accuracy:     ", sk_acc)

    
    assert abs(my_acc - sk_acc) < 0.15
