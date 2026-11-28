"""Itération V3, lot L7 : classifieur de convergence d'XFOIL, écrit pour entrer dans un QUBO.

L'échec d'XFOIL est une contrainte CACHÉE (taxonomie de Le Digabel et Wild, 2023) : on ne sait pas de
combien elle est violée, seulement si le calcul a abouti. La pratique de l'optimisation bayésienne
contrainte (Gramacy et Lee, 2010 ; Gelbart et al., 2014) est d'apprendre la PROBABILITÉ de succès.

Ici : régression logistique sur les monômes de degré <= 2 en z (les mêmes 66 que le modèle en z), avec
une pénalité ridge. Le log-odds (logit) est donc un polynôme quadratique en z, donc un QUBO après
substitution z = a0 + A b. La contrainte « P(succès) >= p_min » s'écrit logit(b) - logit(p_min) >= 0 :
une contrainte quadratique. Une pénalité au carré la porterait au degré 4 ; un terme lagrangien
-lambda * h(b) la garde au degré 2 (contraintes.formulations).
"""
import numpy as np

from optim.qubo_loop import z_of
from quantique.qubo import quadratic_features, surrogate_to_qubo


def logit(p):
    return float(np.log(p / (1 - p)))


class LogistiqueQuadZ:
    """P(succès | z) = sigmoïde(phi(z) . theta), phi = monômes de degré <= 2 en z.

    Ajustement par Newton (moindres carrés repondérés) sur la log-vraisemblance pénalisée
    -sum log p + (ridge / 2) |theta_sans_constante|^2. Le ridge est indispensable : avec 66 coefficients et
    quelques dizaines d'échecs, des données séparables feraient diverger les coefficients."""

    def __init__(self, levels, ridge=1.0, n_iter=30):
        self.levels, self.ridge, self.n_iter = levels, ridge, n_iter

    def _phi(self, I):
        return quadratic_features(z_of(np.atleast_2d(I), self.levels))

    def fit(self, I, y):
        X = self._phi(I)
        y = np.asarray(y, float)
        n, p = X.shape
        R = self.ridge * np.eye(p)
        R[0, 0] = 0.0
        th = np.zeros(p)
        m = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        th[0] = np.log(m / (1 - m))
        for _ in range(self.n_iter):
            eta = np.clip(X @ th, -30, 30)
            mu = 1 / (1 + np.exp(-eta))
            W = mu * (1 - mu)
            g = X.T @ (mu - y) + R @ th
            H = (X * W[:, None]).T @ X + R + 1e-9 * np.eye(p)
            step = np.linalg.solve(H, g)
            th = th - step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.coef = th
        return self

    def logit(self, I):
        return self._phi(I) @ self.coef

    def proba(self, I):
        return 1 / (1 + np.exp(-np.clip(self.logit(I), -30, 30)))

    def contrainte(self, p_min):
        """Coefficients en z de h(z) = logit(z) - logit(p_min) (contrainte h >= 0)."""
        h = self.coef.copy()
        h[0] -= logit(p_min)
        return h

    def to_qubo(self, bits, p_min):
        return surrogate_to_qubo(self.contrainte(p_min), bits, self.levels)


def auc(score, y):
    """Aire sous la courbe ROC (probabilité qu'un succès ait un score plus haut qu'un échec)."""
    y = np.asarray(y, bool)
    s = np.asarray(score, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    rangs = np.empty(len(order))
    rangs[order] = np.arange(1, len(order) + 1)
    return float((rangs[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
