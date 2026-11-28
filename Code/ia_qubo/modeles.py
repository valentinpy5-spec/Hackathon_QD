"""Itération V3, lot L6 : modèles de substitution appris, de l'approche actuelle aux méthodes d'IA.

Quatre familles, avec la même interface (fit, predict, et to_qubo quand le modèle est un QUBO) :

    QuadZ                 l'approche actuelle (V2) : polynôme quadratique en variables normalisées z,
                          converti en QUBO par substitution z = a0 + A b (optim/qubo_loop.py)
    FactorizationMachine  FMQA (Kitai et al., 2020) : modèle quadratique appris directement sur les
                          bits, couplages factorisés <v_i, v_j> de rang k ; c'est un QUBO par construction
    BayesQuadBits         BOCS (Baptista et Poloczek, 2018) : modèle quadratique complet sur les bits,
                          régression bayésienne ; on tire un modèle au hasard dans la loi a posteriori
                          (échantillonnage de Thompson), et ce tirage est un QUBO
    MLPEnsemble           réseau de neurones (ensemble de 5), en variables z ; PAS un QUBO : il faut une
                          recherche directe sur la grille (énumération ou recherche locale)

Tous les modèles normalisent la sortie (y centré réduit) avant l'ajustement.
"""
import numpy as np

from optim.qubo_loop import fit_quadratic, predict as predict_z, z_of
from quantique.qubo import surrogate_to_qubo


# ------------------------------------------------------------ indices <-> bits
def encode(I, bits):
    """Indices (N, d) -> bits (N, n) : bit j de la variable i = (x_i >> j) & 1, poids faible d'abord."""
    I = np.atleast_2d(np.asarray(I, dtype=int))
    cols = [(I[:, i:i + 1] >> np.arange(nb)) & 1 for i, nb in enumerate(bits)]
    return np.hstack(cols).astype(float)


def decode_bits(B, bits):
    """Bits (N, n) -> indices (N, d)."""
    B = np.atleast_2d(B)
    out, k = [], 0
    for nb in bits:
        out.append((B[:, k:k + nb] * (1 << np.arange(nb))).sum(axis=1))
        k += nb
    return np.column_stack(out).astype(int)


def bits_quad_features(B):
    """[1, b_i, b_i b_j (i<j)] : 1 + n + n(n-1)/2 colonnes."""
    B = np.atleast_2d(B)
    n = B.shape[1]
    iu, ju = np.triu_indices(n, 1)
    return np.hstack([np.ones((len(B), 1)), B, B[:, iu] * B[:, ju]])


def theta_to_qubo(theta, n):
    """Coefficients de bits_quad_features -> (Q triangulaire supérieure, c)."""
    Q = np.zeros((n, n))
    Q[np.diag_indices(n)] = theta[1:1 + n]
    iu, ju = np.triu_indices(n, 1)
    Q[iu, ju] = theta[1 + n:]
    return Q, float(theta[0])


class _Scaler:
    def __init__(self, y):
        self.mu = float(np.mean(y))
        self.sd = float(np.std(y)) if len(y) > 1 and np.std(y) > 1e-12 else 1.0

    def fwd(self, y):
        return (np.asarray(y) - self.mu) / self.sd

    def inv(self, y):
        return np.asarray(y) * self.sd + self.mu


# ------------------------------------------------------------ 1. approche actuelle : quadratique en z
class QuadZ:
    """Polynôme quadratique complet en z (15 coefficients pour 4 variables), moindres carrés ridge."""
    name = "quadratique en z"

    def __init__(self, bits, levels, ridge=1e-6):
        self.bits, self.levels, self.ridge = bits, levels, ridge

    def fit(self, B, y):
        I = decode_bits(B, self.bits)
        self.coef = fit_quadratic(I, np.asarray(y, float), ridge=self.ridge, levels=self.levels)
        return self

    def predict(self, B):
        return predict_z(self.coef, decode_bits(B, self.bits), levels=self.levels)

    def to_qubo(self):
        return surrogate_to_qubo(self.coef, self.bits, self.levels)

    @property
    def n_params(self):
        return len(self.coef)


# ------------------------------------------------------------ 2. machine de factorisation (FMQA)
class FactorizationMachine:
    """y(b) = w0 + sum_i w_i b_i + sum_{i<j} <v_i, v_j> b_i b_j, avec v_i dans R^k.

    Ajustée par descente de gradient (Adam) sur l'erreur quadratique + pénalité L2.
    n_models > 1 : moyenne de plusieurs machines (graines différentes) ; la moyenne de QUBO est un QUBO.
    """
    name = "machine de factorisation"

    def __init__(self, n, k=4, l2=1e-3, epochs=1500, lr=0.02, n_models=3, seed=0):
        self.n, self.k, self.l2, self.epochs, self.lr = n, k, l2, epochs, lr
        self.n_models, self.seed = n_models, seed

    def _fit_one(self, B, y, rng):
        n, k = self.n, self.k
        w0, w = 0.0, np.zeros(n)
        V = rng.normal(0, 0.1, (n, k))
        params = [np.zeros(1), w, V]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        N = len(y)
        for t in range(1, self.epochs + 1):
            BV = B @ V
            pred = params[0][0] + B @ w + 0.5 * ((BV ** 2).sum(1) - (B @ (V ** 2)).sum(1))
            r = pred - y
            g0 = np.array([2 * r.mean()])
            gw = 2 * B.T @ r / N + 2 * self.l2 * w
            gV = 2 * (B.T @ (r[:, None] * BV) - (B.T @ r)[:, None] * V) / N + 2 * self.l2 * V
            for p, g, mi, vi in zip(params, (g0, gw, gV), m, v):
                mi *= b1
                mi += (1 - b1) * g
                vi *= b2
                vi += (1 - b2) * g * g
                p -= self.lr * (mi / (1 - b1 ** t)) / (np.sqrt(vi / (1 - b2 ** t)) + eps)
        return params[0][0], w.copy(), V.copy()

    def fit(self, B, y):
        B = np.asarray(B, float)
        self.sc = _Scaler(y)
        ys = self.sc.fwd(y)
        rng = np.random.default_rng(self.seed)
        self.models = [self._fit_one(B, ys, rng) for _ in range(self.n_models)]
        return self

    def predict(self, B):
        B = np.atleast_2d(np.asarray(B, float))
        out = []
        for w0, w, V in self.models:
            BV = B @ V
            out.append(w0 + B @ w + 0.5 * ((BV ** 2).sum(1) - (B @ (V ** 2)).sum(1)))
        return self.sc.inv(np.mean(out, axis=0))

    def to_qubo(self):
        n = self.n
        Q, c = np.zeros((n, n)), 0.0
        iu, ju = np.triu_indices(n, 1)
        for w0, w, V in self.models:
            G = V @ V.T                                   # G_ij = <v_i, v_j>
            Qm = np.zeros((n, n))
            Qm[np.diag_indices(n)] = w
            Qm[iu, ju] = G[iu, ju]
            Q += Qm
            c += w0
        Q, c = Q / self.n_models, c / self.n_models
        return Q * self.sc.sd, c * self.sc.sd + self.sc.mu

    @property
    def n_params(self):
        return 1 + self.n + self.n * self.k


# ------------------------------------------------------------ 3. modèle quadratique bayésien sur les bits (BOCS)
class BayesQuadBits:
    """Régression bayésienne sur [1, b_i, b_i b_j] : a priori N(0, 1/alpha) sur chaque coefficient (sauf la
    constante), bruit de précision beta ; alpha et beta par maximum de l'évidence (MacKay).
    sample_qubo() tire un modèle dans la loi a posteriori (Thompson) et le rend sous forme de QUBO."""
    name = "quadratique bayésien sur les bits"

    def __init__(self, n, n_iter=30, seed=0):
        self.n, self.n_iter = n, n_iter
        self.rng = np.random.default_rng(seed)

    def fit(self, B, y):
        Phi = bits_quad_features(np.asarray(B, float))
        self.sc = _Scaler(y)
        t = self.sc.fwd(y)
        N, p = Phi.shape
        alpha, beta = 1.0, 10.0
        G = Phi.T @ Phi
        lam = np.linalg.eigvalsh(G)
        for _ in range(self.n_iter):
            A = alpha * np.eye(p)
            A[0, 0] = 1e-6
            S = np.linalg.inv(A + beta * G)
            mN = beta * S @ Phi.T @ t
            gamma = float(np.sum(beta * lam / (alpha + beta * lam)))
            alpha = max(gamma / max(mN[1:] @ mN[1:], 1e-12), 1e-6)
            beta = max((N - gamma) / max(np.sum((t - Phi @ mN) ** 2), 1e-12), 1e-6)
        A = alpha * np.eye(p)
        A[0, 0] = 1e-6
        self.S = np.linalg.inv(A + beta * G)
        self.S = 0.5 * (self.S + self.S.T)
        self.m = beta * self.S @ Phi.T @ t
        self.alpha, self.beta = alpha, beta
        return self

    def predict(self, B):
        return self.sc.inv(bits_quad_features(np.asarray(B, float)) @ self.m)

    def _to_qubo(self, theta):
        Q, c = theta_to_qubo(theta, self.n)
        return Q * self.sc.sd, c * self.sc.sd + self.sc.mu

    def to_qubo(self):
        return self._to_qubo(self.m)

    def sample_qubo(self):
        L = np.linalg.cholesky(self.S + 1e-10 * np.eye(len(self.m)))
        return self._to_qubo(self.m + L @ self.rng.standard_normal(len(self.m)))

    @property
    def n_params(self):
        return 1 + self.n + self.n * (self.n - 1) // 2


# ------------------------------------------------------------ 4. réseau de neurones (pas un QUBO)
class MLPEnsemble:
    """Ensemble de perceptrons à 2 couches cachées (tanh), entrée z, ajustés par Adam.
    La moyenne de l'ensemble sert de prédiction, l'écart entre membres d'estimation d'incertitude."""
    name = "réseau de neurones"

    def __init__(self, bits, levels, hidden=32, l2=1e-3, epochs=1500, lr=0.01, n_models=5, seed=0):
        self.bits, self.levels, self.h, self.l2 = bits, levels, hidden, l2
        self.epochs, self.lr, self.n_models, self.seed = epochs, lr, n_models, seed

    def _z(self, B):
        return z_of(decode_bits(B, self.bits), self.levels)

    def _fit_one(self, X, y, rng):
        d, h = X.shape[1], self.h
        P = [rng.normal(0, 1 / np.sqrt(d), (d, h)), np.zeros(h), rng.normal(0, 1 / np.sqrt(h), (h, h)),
             np.zeros(h), rng.normal(0, 1 / np.sqrt(h), (h, 1)), np.zeros(1)]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        N = len(y)
        for t in range(1, self.epochs + 1):
            W1, c1, W2, c2, W3, c3 = P
            a1 = np.tanh(X @ W1 + c1)
            a2 = np.tanh(a1 @ W2 + c2)
            out = (a2 @ W3 + c3)[:, 0]
            r = 2 * (out - y)[:, None] / N
            gW3, gc3 = a2.T @ r, r.sum(0)
            d2 = (r @ W3.T) * (1 - a2 ** 2)
            gW2, gc2 = a1.T @ d2, d2.sum(0)
            d1 = (d2 @ W2.T) * (1 - a1 ** 2)
            gW1, gc1 = X.T @ d1, d1.sum(0)
            grads = [gW1 + 2 * self.l2 * W1, gc1, gW2 + 2 * self.l2 * W2, gc2, gW3 + 2 * self.l2 * W3, gc3]
            for p, g, mi, vi in zip(P, grads, m, v):
                mi *= 0.9
                mi += 0.1 * g
                vi *= 0.999
                vi += 0.001 * g * g
                p -= self.lr * (mi / (1 - 0.9 ** t)) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
        return P

    def fit(self, B, y):
        X = self._z(B)
        self.sc = _Scaler(y)
        rng = np.random.default_rng(self.seed)
        self.models = [self._fit_one(X, self.sc.fwd(y), rng) for _ in range(self.n_models)]
        return self

    def _members(self, B):
        X = self._z(B)
        out = []
        for W1, c1, W2, c2, W3, c3 in self.models:
            out.append((np.tanh(np.tanh(X @ W1 + c1) @ W2 + c2) @ W3 + c3)[:, 0])
        return self.sc.inv(np.array(out))

    def predict(self, B):
        return self._members(B).mean(axis=0)

    def predict_std(self, B):
        return self._members(B).std(axis=0)

    @property
    def n_params(self):
        d = len(self.bits)
        return d * self.h + self.h + self.h * self.h + self.h + self.h + 1
