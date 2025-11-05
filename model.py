import os
import joblib
import numpy as np

class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        if hasattr(X, "todense"):
            X = X.todense()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if self.fit_intercept:
            X_offset = np.mean(X, axis=0)
            y_offset = np.mean(y, axis=0)
            X_centered = X - X_offset
            y_centered = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0
            X_centered, y_centered = X, y

        coef, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        self.coef_ = coef.flatten()
        self.intercept_ = y_offset - X_offset @ self.coef_
        return self

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        y_pred = X @ self.coef_
        if self.fit_intercept:
            y_pred += self.intercept_
        return y_pred.flatten()

    def score(self, X, y_true):
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def mase(self, X, y_true):
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = self.predict(X).flatten()
        mae_model = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
        return mae_model / mae_naive if mae_naive != 0 else np.nan

    def save(self, path="linear_model.pkl"):
        """Lưu model vào file .pkl (đa nền tảng, tự tạo thư mục nếu cần)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "fit_intercept": self.fit_intercept
        }, path)
        print(f"💾 Model saved to {os.path.abspath(path)}")
