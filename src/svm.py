import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, epochs=1000, C=1.0):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.w = None
        self.b = None
    
    

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        for epoch in range(self.epochs):
            for i in range(n_samples):
                prediction = np.dot(self.w,X[i]) + self.b
                condition = y[i] * prediction 
                if condition >= 1:
                    self.w = self.w - self.learning_rate * self.w
                else:
                    self.w = self.w - self.learning_rate * (self.w - self.C * y[i] * X[i])
                    self.b = self.b - self.learning_rate * (-self.C * y[i])
            if epoch % 100 == 0:
                predictions = np.sign(np.dot(X, self.w) + self.b)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}")
        return self.w, self.b
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)