"""Itération V3, lot L1 : concurrent bayésien de la boucle QUBO.

Ce qui change par rapport à la V2 : la boucle QUBO (optim/qubo_loop.py) est une méthode à
modèle de substitution ; son vrai concurrent n'est pas NSGA-II mais l'optimisation
bayésienne. Ce module écrit une boucle de MÊME structure (mêmes points initiaux pour une
graine donnée, mêmes 5 poids, un profil par poids et par itération, même compteur), dans
laquelle on choisit séparément :

    model     "gp"   : krigeage (processus gaussien, bibliothèque SMT), un par grandeur
              "quad" : modèle quadratique bayésien (mêmes monômes que la boucle QUBO)
    selection "ei"   : amélioration espérée (acquisition bayésienne, utilise l'incertitude)
              "mean" : minimum de la prédiction (ce que fait un QUBO résolu exactement)
    scalar    "sum"  : somme pondérée des deux objectifs (comme la boucle QUBO)
              "tcheby" : Tchebychev augmenté (ParEGO, Knowles 2006), qui atteint aussi
                         les points non convexes du front

D'où l'ablation modèle x sélection demandée par la fiche (hypothèse H1) :
    quad + mean   = la boucle QUBO sans son encodage binaire (contrôle)
    gp   + mean   = effet du modèle seul
    quad + ei     = effet de la sélection seule
    gp   + ei     = optimisation bayésienne

Contraintes : t >= t_min est connue exactement (filtre sur l'indice) ; la manœuvre
C_L,max >= cl_min est modélisée (même famille que les objectifs) et prise en compte par la
probabilité de faisabilité ("pof", défaut : EI x PoF, ou prédiction + (1 - PoF)), ou imposée
en dur sur la prédiction ("mean", comme SEGO), ou pénalisée comme dans la boucle QUBO ("lin" :
modèle linéaire de C_L,max et pénalité quadratique de la violation, pour la sélection "mean").
Le front est au bord de cette contrainte : un traitement strict (en dur, ou "pof" avec la
sélection "mean") écarte ses profils ; voir la fiche projet, lot L1.

L'acquisition est évaluée sur TOUS les profils non simulés (la grille est énumérable).

    python -m optim.experiences_l1          # comparaison -> resultats/pipeline_L1_V3.json
"""
from itertools import product

import numpy as np
from scipy.stats import norm as _gauss

from quantique.qubo import quadratic_features
from .qubo_loop import z_of, fit_linear, predict

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ------------------------------------------------------------ modèles de substitution
class _Constant:
    """Repli quand il y a trop peu de points ou une sortie constante."""

    def __init__(self, y):
        self.m = float(np.mean(y)) if len(y) else 0.0
        self.s = float(np.std(y)) if len(y) > 1 else 1.0

    def predict(self, Z):
        return np.full(len(Z), self.m), np.full(len(Z), max(self.s, 1e-9))


class GPModel:
    """Krigeage SMT (noyau Matérn 5/2, bruit estimé : les sorties XFOIL ne sont pas lisses)."""

    def __init__(self, Z, y):
        from smt.surrogate_models import KRG
        self.mu, self.sd = float(np.mean(y)), float(np.std(y))
        self.fallback = None
        if len(y) < 4 or self.sd < 1e-12:
            self.fallback = _Constant(y)
            return
        d = Z.shape[1]
        self.gp = KRG(theta0=[0.5] * d, corr="matern52", poly="constant", eval_noise=True,
                      noise0=[1e-3], n_start=1, hyper_opt="Cobyla", print_global=False)
        self.gp.set_training_values(np.asarray(Z, float), ((y - self.mu) / self.sd).reshape(-1, 1))
        try:
            self.gp.train()
        except Exception:                      # matrice de corrélation mal conditionnée
            self.fallback = _Constant(y)

    def predict(self, Z):
        if self.fallback:
            return self.fallback.predict(Z)
        Z = np.asarray(Z, float)
        m = self.gp.predict_values(Z)[:, 0] * self.sd + self.mu
        v = np.maximum(self.gp.predict_variances(Z)[:, 0], 0.0)
        return m, np.sqrt(v) * self.sd


class QuadModel:
    """Régression quadratique bayésienne (sklearn BayesianRidge) sur les monômes de degré <= 2 :
    même famille que la boucle QUBO, mais avec une variance de prédiction."""

    def __init__(self, Z, y):
        from sklearn.linear_model import BayesianRidge
        self.fallback = None
        if len(y) < 4 or np.std(y) < 1e-12:
            self.fallback = _Constant(y)
            return
        self.br = BayesianRidge(fit_intercept=True)
        self.br.fit(quadratic_features(Z)[:, 1:], y)       # sans la colonne constante

    def predict(self, Z):
        if self.fallback:
            return self.fallback.predict(Z)
        m, s = self.br.predict(quadratic_features(Z)[:, 1:], return_std=True)
        return m, s


MODELS = {"gp": GPModel, "quad": QuadModel}


# ------------------------------------------------------------ acquisition
def expected_improvement(mean, std, best):
    """Amélioration espérée pour une minimisation : E[max(best - Y, 0)], Y ~ N(mean, std^2)."""
    std = np.maximum(std, 1e-12)
    u = (best - mean) / std
    return (best - mean) * _gauss.cdf(u) + std * _gauss.pdf(u)


def tchebychev(F, w, rho=0.05):
    """Tchebychev augmenté de ParEGO sur des objectifs normalisés dans [0, 1]."""
    W = np.array([w, 1.0 - w])
    return np.max(F * W, axis=1) + rho * (F @ W)


# ------------------------------------------------------------ boucle
def bayes_loop(problem, model="gp", selection="ei", scalar="sum", constraint="pof",
               n_init=10, weights=WEIGHTS, k_per_w=1, budget=120, seed=0, cons=None,
               stop_when=None, levels=(8, 4, 8, 4), keys=("im", "ip", "it", "iI"),
               con_col="CLmax", fit_on="feasible"):
    """Boucle à modèle de substitution, de même structure que qubo_loop.

    cons : {'t_min_index': True, 't_var': 2, 'cl_min': 1.2}, comme pour qubo_loop.
    fit_on : "feasible" (comme la boucle QUBO) ou "valid" (tous les profils convergés).
    Renvoie dict(history=[nombre de simulations après chaque itération], iterations)."""
    cons = cons or {}
    Model = MODELS[model]
    rng = np.random.default_rng(seed)
    all_idx = list(product(*[range(L) for L in levels]))
    Zall = z_of(np.array(all_idx), levels)
    # contrainte exacte sur l'épaisseur : l'indice 0 de la variable t_var est interdit
    allowed = np.ones(len(all_idx), bool)
    if cons.get("t_min_index"):
        allowed = np.array([i[cons.get("t_var", 2)] != 0 for i in all_idx])
    cl_min = cons.get("cl_min")
    pos = {idx: k for k, idx in enumerate(all_idx)}

    for j in rng.permutation(len(all_idx))[:n_init]:        # mêmes points initiaux que qubo_loop
        problem(all_idx[j])
    history = [problem.n_sims]
    it = 0
    while problem.n_sims < budget:
        it += 1
        rows = [problem.table[i] for i in problem.seen]
        Z = Zall[[pos[i] for i in problem.seen]]
        ok = np.array([r["feasible"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        free = np.array([i not in problem._set for i in all_idx]) & allowed
        if ok.sum() < 4:
            # pas assez de points faisables : exploration aléatoire (comme qubo_loop)
            new = [all_idx[j] for j in rng.permutation(len(all_idx)) if all_idx[j] not in problem._set][:2]
        else:
            cd = np.array([r["CD_counts"] for r in rows])
            ser = np.array([r["SER_db"] for r in rows])
            fit = ok if fit_on == "feasible" else (val & np.isfinite(cd) & np.isfinite(ser))
            lo = np.array([cd[ok].min(), ser[ok].min()])
            span = np.array([max(np.ptp(cd[ok]), 1e-6), max(np.ptp(ser[ok]), 1e-6)])
            Fn = (np.column_stack([cd, ser]) - lo) / span             # objectifs normalisés
            # contrainte de manœuvre
            feas_mask, pof = np.ones(len(all_idx), bool), np.ones(len(all_idx))
            lin_pen = np.zeros(len(all_idx))
            I_seen = np.array(problem.seen)
            if cl_min is not None:
                cl = np.array([r[con_col] for r in rows], dtype=float)
                vl = val & np.isfinite(cl)
                mc, sc = Model(Z[vl], cl[vl]).predict(Zall)
                pof = _gauss.cdf((mc - cl_min) / np.maximum(sc, 1e-12))
                if constraint == "mean":
                    feas_mask = mc >= cl_min
                elif constraint == "lin":
                    # pénalité de la boucle QUBO, sans l'encodage binaire : modèle LINÉAIRE de
                    # C_L,max et P * cl_weight * violation^2 (P = 3, cl_weight = 25)
                    L = predict(fit_linear(I_seen[vl], cl[vl], levels=levels), np.array(all_idx), levels)
                    lin_pen = 3.0 * cons.get("cl_weight", 25.0) * np.maximum(cl_min - L, 0.0) ** 2
            if scalar == "sum":
                m1, s1 = Model(Z[fit], Fn[fit, 0]).predict(Zall)
                m2, s2 = Model(Z[fit], Fn[fit, 1]).predict(Zall)
            new = []
            for w in weights:
                if scalar == "sum":
                    y_obs = w * Fn[:, 0] + (1 - w) * Fn[:, 1]
                    mean = w * m1 + (1 - w) * m2
                    std = np.sqrt((w * s1) ** 2 + ((1 - w) * s2) ** 2)
                else:                                              # ParEGO : un modèle par poids
                    y_obs = tchebychev(Fn, w)
                    mean, std = Model(Z[fit], y_obs[fit]).predict(Zall)
                if selection == "ei":
                    score = expected_improvement(mean, std, y_obs[ok].min())
                    if constraint == "pof":
                        score = score * pof
                else:
                    score = -mean
                    if constraint == "pof":            # pénalité : un profil infaisable coûte 1,
                        score = score - (1.0 - pof)    # soit toute l'étendue observée de l'objectif
                    score = score - lin_pen
                cand = free & feas_mask
                if not cand.any():                                  # aucun profil prédit faisable :
                    cand, score = free, pof                         # on prend le plus probablement faisable
                order = np.argsort(-np.where(cand, score, -np.inf), kind="stable")
                taken = 0
                for j in order:
                    if not cand[j]:
                        break
                    if all_idx[j] not in new:
                        new.append(all_idx[j])
                        taken += 1
                        if taken == k_per_w:
                            break
            if not new:
                new = [all_idx[j] for j in rng.permutation(len(all_idx)) if all_idx[j] not in problem._set][:1]
        for idx in new:
            if problem.n_sims >= budget:
                break
            problem(idx)
        history.append(problem.n_sims)
        if stop_when and stop_when(problem):
            break
    return dict(history=history, iterations=it)
