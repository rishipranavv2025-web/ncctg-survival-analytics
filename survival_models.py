import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


class ExponentialPHModel:
    """Exponential proportional-hazards model fitted by maximum likelihood."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
        self.feature_names_ = None
        self.log_likelihood_ = None
        self.scaler_ = StandardScaler()
        self.n_params_ = 0
        self.optimization_success_ = False
        self.optimization_message_ = None

    def fit(self, X, durations, events, feature_names=None):
        X = np.asarray(X, dtype=float)
        durations = np.asarray(durations, dtype=float).ravel()
        events = np.asarray(events, dtype=int).ravel()

        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if len(durations) != X.shape[0] or len(events) != X.shape[0]:
            raise ValueError("X, durations, and events must have matching rows.")

        if not np.isfinite(X).all():
            raise ValueError("X contains missing or non-finite values.")

        if not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("Durations must be finite and positive.")

        if not np.isin(events, [0, 1]).all():
            raise ValueError("Events must contain only 0 and 1.")

        if events.sum() == 0:
            raise ValueError("At least one observed event is required.")

        self.feature_names_ = (
            list(feature_names) if feature_names is not None else None
        )

        Z = self.scaler_.fit_transform(X)

        def neg_loglik(theta):
            eta = theta[0] + Z @ theta[1:]
            eta = np.clip(eta, -30, 30)
            hazard = np.exp(eta)
            value = -np.sum(events * eta - hazard * durations)
            return float(value) if np.isfinite(value) else np.inf

        event_rate = events.sum() / durations.sum()
        theta0 = np.zeros(Z.shape[1] + 1, dtype=float)
        theta0[0] = np.log(event_rate)

        bounds = [(-20, 5)] + [(-10, 10)] * Z.shape[1]

        result = minimize(
            neg_loglik,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-12}
        )

        self.optimization_success_ = bool(
            result.success and np.isfinite(result.fun)
        )
        self.optimization_message_ = str(result.message)

        if not self.optimization_success_:
            raise RuntimeError(
                "Exponential PH optimization failed: "
                + self.optimization_message_
            )

        self.intercept_ = float(result.x[0])
        self.coef_ = np.asarray(result.x[1:], dtype=float)
        self.log_likelihood_ = float(-result.fun)
        self.n_params_ = len(result.x)

        return self

    def _eta(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != len(self.coef_):
            raise ValueError("X has an incorrect number of features.")

        if not np.isfinite(X).all():
            raise ValueError("X contains missing or non-finite values.")

        Z = self.scaler_.transform(X)
        return np.clip(self.intercept_ + Z @ self.coef_, -30, 30)

    def predict_hazard(self, X):
        return np.exp(self._eta(X))

    def predict_survival(self, X, times):
        times = np.asarray(times, dtype=float).ravel()

        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("Prediction times must be finite and non-negative.")

        hazards = self.predict_hazard(X)
        return np.exp(-np.outer(times, hazards))

    @property
    def aic_(self):
        if self.log_likelihood_ is None:
            raise RuntimeError("Model has not been fitted.")

        return 2 * self.n_params_ - 2 * self.log_likelihood_