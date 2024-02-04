import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

class SupportVectorMachine:

    def __init__(self, lr=0.001, lambdap=0.01, iterations=1000):
        self.lr = lr
        self.lambdap = lambdap
        self.iterations = iterations
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        print("Fitting model...")
        start_time = time.time()
        samples, features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(features)
        self.biases = 0

        for _ in range(self.iterations):
            for index, xi in enumerate(X):
                if y_[index] * (np.dot(xi, self.weights) - self.biases) >= 1:
                    self.weights -= self.lr * (2 * self.lambdap * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambdap * self.weights - np.dot(xi, y_[index]))
                    self.biases -= self.lr * y_[index]
        end_time = time.time()
        print(f"Fit complete! (Time taken: {np.round(end_time - start_time, 2)} seconds) ")
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.biases)

    def model_accuracy(self, y_values, y_predictions):
         accuracy = np.sum(y_values==y_predictions) / len(y_values)
         return print(f"Model Accuracy: {accuracy * 100}%")


'''

TESTING

Generate 10,000 samples of data, each with 10 features.
Test accuracy of our SVM model vs the SKLearn standard SVM model

'''

X, y = datasets.make_blobs(n_samples=10000, n_features=5, cluster_std=1.2, random_state=60)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# My model
mod = SupportVectorMachine()
mod.fit(X_train, y_train)
predictions = mod.predict(X_test)
mod.model_accuracy(y_test, predictions)

# SKLearn model
clf = SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(f"SKLearn model accuracy: {accuracy_score(y_test,y_pred)*100}")
