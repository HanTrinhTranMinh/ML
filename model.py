import os
import joblib
import numpy as np

import numpy as np


class MyOLSLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        if hasattr(X, "todense"):
            X = X.todense()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        n_samples = X.shape[0]
        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])

        # === Normal Equation: θ = (XᵀX)^(-1)Xᵀy ===
        theta = np.linalg.pinv(X.T @ X) @ X.T @ y

        # === Lưu lại hệ số ===
        if self.fit_intercept:
            self.intercept_ = theta[0, 0]
            self.coef_ = theta[1:, 0]
        else:
            self.coef_ = theta.flatten()
        return self

    def predict(self, X):
    # Nếu là sparse matrix, chuyển sang dense
        if hasattr(X, "todense"):
            X = X.todense()
        elif hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float64)

        y_pred = X @ self.coef_
        if self.fit_intercept:
            y_pred += self.intercept_
        return np.ravel(y_pred)


    def score(self, X, y_true):
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res / ss_tot

    def save(self, path="ols_model.pkl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "fit_intercept": self.fit_intercept
        }, path)
        print(f"OLS model saved to {os.path.abspath(path)}")


from sklearn.preprocessing import PolynomialFeatures

class MyPolynomialRegression:
    def __init__(self, degree=2, fit_intercept=True):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.linear = MyOLSLinearRegression(fit_intercept=fit_intercept)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.linear.fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.linear.predict(X_poly)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def save(self, path="poly_model.pkl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "degree": self.degree,
            "linear": {
                "coef_": self.linear.coef_,
                "intercept_": self.linear.intercept_
            }
        }, path)
        print(f"💾 Polynomial model saved to {os.path.abspath(path)}")
