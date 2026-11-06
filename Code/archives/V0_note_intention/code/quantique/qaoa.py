"""QAOA simulé exactement (vecteur d'état NumPy), sans bibliothèque quantique.

Pour un QUBO, l'hamiltonien de coût est DIAGONAL dans la base de calcul :
H_C |x> = E(x) |x>. Appliquer exp(-i gamma H_C) revient donc à multiplier
chaque amplitude par exp(-i gamma E(x)). Le mélangeur H_M = sum_k X_k agit
qubit par qubit : exp(-i beta X) = cos(beta) I - i sin(beta) X.

Coût mémoire : 2^n amplitudes complexes (16 octets chacune). n = 9 -> 512
amplitudes : exactement la taille de notre grille de conception.
"""
import numpy as np
from scipy.optimize import minimize


def apply_mixer(psi, beta, n):
    """exp(-i beta sum_k X_k) |psi>."""
    psi = psi.reshape((2,) * n)
    c, s = np.cos(beta), -1j * np.sin(beta)
    for ax in range(n):
        psi = c * psi + s * np.flip(psi, axis=ax)
    return psi.reshape(-1)


def qaoa_state(gammas, betas, E, n):
    """|gamma, beta> = prod_k U_M(beta_k) U_C(gamma_k) |+>^n."""
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E)
        psi = apply_mixer(psi, b, n)
    return psi


def expectation(params, E, n):
    p = len(params) // 2
    psi = qaoa_state(params[:p], params[p:], E, n)
    return float(np.real(np.vdot(psi, E * psi)))


def normalize(E):
    """Centre-réduit l'énergie. Indispensable : gamma multiplie E dans une phase
    exp(-i gamma E). Si E vaut ~100 (CD en counts), un petit changement de
    gamma fait tourner les phases de plusieurs tours et le paysage devient
    illisible pour l'optimiseur classique."""
    return (E - E.mean()) / E.std()


def optimize_qaoa(E, n, p, n_restarts=10, seed=0, init=None):
    """Boucle variationnelle : l'ordinateur classique règle (gamma, beta).

    init : paramètres de départ (par exemple interpolés depuis p-1, stratégie
    INTERP de Zhou et al. 2020), sinon tirages aléatoires.
    """
    rng = np.random.default_rng(seed)
    best = None
    starts = [init] if init is not None else []
    starts += [np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
               for _ in range(n_restarts)]
    for x0 in starts:
        r = minimize(expectation, x0, args=(E, n), method="COBYLA",
                     options={"maxiter": 2000, "rhobeg": 0.3})
        if best is None or r.fun < best.fun:
            best = r
    return best


def interp_init(params, p_new):
    """Prolonge (gamma, beta) optimaux de p à p+1 par interpolation linéaire."""
    p = len(params) // 2
    out = []
    for v in (params[:p], params[p:]):
        x_old = np.linspace(0, 1, p)
        x_new = np.linspace(0, 1, p_new)
        out.append(np.interp(x_new, x_old, v) if p > 1 else np.full(p_new, v[0]))
    return np.concatenate(out)


def sample(psi, shots, seed=0):
    """Mesures dans la base de calcul : renvoie les indices x tirés (règle de Born)."""
    prob = np.abs(psi) ** 2
    rng = np.random.default_rng(seed)
    return rng.choice(len(prob), size=shots, p=prob / prob.sum())
