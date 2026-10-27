"""Mixed linear regression fitted with hard-assignment EM, plus a routing classifier."""
import numpy as np
from sklearn import linear_model, svm

ALPHA = 0.01  # ridge strength, as in the course lecture


def _ridge(alpha):
    return linear_model.Ridge(alpha=alpha)


class MixedRegression:
    """K linear (ridge) models; each training point belongs to the one that fits it best.

    fit():            start with one ridge per quantile bin of y, then alternate
                      E-step (assign each point to the model with the smallest
                      |residual|) and M-step (refit each model on its points).
    fit_classifier(): learn which model to use from the features alone
                      (LinearSVC), since y is unknown at test time.
    predict():        route each point to a model with the classifier.
    """

    def __init__(self, n_components=4, n_iter=20, alpha=ALPHA):
        self.n_components = n_components
        self.n_iter = n_iter
        self.alpha = alpha
        self.models = None
        self.clf = None

    def _init_models(self, X, y, y_orig):
        n = self.n_components
        if n == 1:
            self.models = [_ridge(self.alpha).fit(X, y)]
            return
        edges = np.unique(np.quantile(y_orig.values, np.linspace(0, 1, n + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        models, idx_all = [], np.arange(len(y_orig))
        for c in range(len(edges) - 1):
            mask = (y_orig.values >= edges[c]) & (y_orig.values < edges[c + 1])
            sel = idx_all[mask] if mask.sum() >= 5 else idx_all
            models.append(_ridge(self.alpha).fit(X.iloc[sel], y.iloc[sel]))
        while len(models) < n:  # pad if quantiles collapsed
            models.append(_ridge(self.alpha).fit(X, y))
        self.models = models[:n]

    def _predict_all(self, X):
        return np.column_stack([m.predict(X) for m in self.models])

    def assign(self, X, y):
        """E-step: index of the component with the smallest absolute residual."""
        residuals = self._predict_all(X) - y.values[:, None]
        return np.argmin(np.abs(residuals), axis=1)

    def _m_step(self, X, y, best):
        for c, model in enumerate(self.models):
            w = (best == c).astype(float)
            if w.sum() > 10:
                model.fit(X, y, sample_weight=w)

    def fit(self, X, y, y_orig):
        self._init_models(X, y, y_orig)
        if self.n_components > 1:
            for _ in range(self.n_iter):
                self._m_step(X, y, self.assign(X, y))
        return self

    def fit_classifier(self, X, y):
        self.clf = None
        if self.n_components == 1:
            return np.zeros(len(y), dtype=int)
        best = self.assign(X, y)
        if len(np.unique(best)) == 1:
            self._only_cluster = int(best[0])
            return best
        self.clf = svm.LinearSVC(max_iter=5000, dual=False, random_state=0)
        self.clf.fit(X, best)
        return best

    def predict(self, X):
        preds_all = self._predict_all(X)
        if self.n_components == 1:
            chosen = np.zeros(len(X), dtype=int)
        elif self.clf is None:
            chosen = np.full(len(X), self._only_cluster, dtype=int)
        else:
            chosen = self.clf.predict(X)
        return preds_all[np.arange(len(X)), chosen], chosen
