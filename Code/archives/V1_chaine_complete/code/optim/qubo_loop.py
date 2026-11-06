"""Boucle itérative « modèle quadratique -> QUBO -> solveur -> simulation » (phase 4), bi-objectif.

Une itération :
    1. ajuster, sur les profils déjà simulés, des modèles quadratiques
       (famille A : 10 coefficients en (m, p, t) normalisés) de C_D et de la SER,
       et un modèle LINÉAIRE de C_L (pour la contrainte de portance) ;
    2. pour chaque poids w d'une somme pondérée, construire un QUBO :
           w * C_D_hat + (1 - w) * SER_hat        (objectifs normalisés)
         + pénalité exacte « t >= t_min »          (1 qubit auxiliaire, réduction de Rosenberg)
         + pénalité « C_L_hat >= C_L,min »          (3 qubits d'écart, contrainte linéarisée)
       soit 9 + 1 + 3 = 13 qubits ;
    3. résoudre chaque QUBO (exact, recuit simulé ou QAOA) et garder les k
       meilleurs profils distincts pas encore simulés ;
    4. les simuler (compteur ++), mettre à jour l'archive.
Le front rendu n'est formé que de profils réellement simulés.
"""
import numpy as np

from quantique.qubo import quadratic_features, surrogate_to_qubo, all_bitstrings, qubo_energy
from quantique.recuit import simulated_annealing
from quantique import qaoa as qa
from .pareto import pareto_front

BITS = (3, 3, 3)
LEVELS = (8, 8, 8)


# ------------------------------------------------------------ modèles de substitution
def z_of(idx):
    return 2 * np.asarray(idx, dtype=float) / 7.0 - 1.0


def fit_quadratic(I, y, ridge=1e-6):
    """Moindres carrés (légèrement régularisés) sur les 10 monômes de degré <= 2."""
    Phi = quadratic_features(z_of(I))
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ y)


def fit_linear(I, y, ridge=1e-6):
    """Modèle affine en (m, p, t) normalisés, mis au format des 10 coefficients."""
    Z = z_of(I)
    Phi = np.column_stack([np.ones(len(Z)), Z])
    c = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(4), Phi.T @ y)
    return np.concatenate([c, np.zeros(6)])


def predict(coef, I):
    return quadratic_features(z_of(I)) @ coef


# ------------------------------------------------------------ pénalités en QUBO
def grow(Q, n_new):
    n = Q.shape[0]
    Q2 = np.zeros((n + n_new, n + n_new))
    Q2[:n, :n] = Q
    return Q2


def add_term(Q, i, j, v):
    """Ajoute v * b_i b_j (i = j : terme linéaire) dans un Q triangulaire supérieur."""
    i, j = min(i, j), max(i, j)
    Q[i, j] += v


def penalty_index_nonzero(Q, c, bits, y, P):
    """Pénalité P (1-b0)(1-b1)(1-b2) (interdit l'indice 0 d'une variable à 3 bits), rendue
    quadratique par un qubit auxiliaire y = (1-b1)(1-b2) (réduction de Rosenberg) :
        P (1-b0) y + 2P [x1 x2 - 2 x1 y - 2 x2 y + 3 y],  x1 = 1-b1, x2 = 1-b2.
    Le crochet vaut 0 si y = x1 x2 et au moins 1 sinon : au minimum sur y, on retrouve
    exactement le terme cubique."""
    b0, b1, b2 = bits
    PR = 2.0 * P
    c += PR                                   # PR * x1 x2 = PR (1 - b1 - b2 + b1 b2)
    add_term(Q, b1, b1, -PR)
    add_term(Q, b2, b2, -PR)
    add_term(Q, b1, b2, PR)
    add_term(Q, y, y, -PR + P)                # -2x1y -2x2y +3y -> -y + 2 b1 y + 2 b2 y ; + P y
    add_term(Q, b1, y, 2 * PR)
    add_term(Q, b2, y, 2 * PR)
    add_term(Q, b0, y, -P)                    # - P b0 y
    return Q, c


def penalty_linear_geq(Q, c, lin_coef, lin_const, bmin, slack_bits, slack_max, P):
    """Pénalité P (L(b) - bmin - s)^2 avec L(b) = lin_const + lin_coef . b (contrainte L >= bmin)
    et s = slack_max * sum_j 2^j s_j / (2^n - 1) >= 0 codé sur les qubits slack_bits."""
    n_s = len(slack_bits)
    step = slack_max / (2**n_s - 1)
    # vecteur affine complet : v . x + v0, sur les variables (bits de conception + écart)
    terms = [(i, a) for i, a in lin_coef] + [(slack_bits[j], -step * 2**j) for j in range(n_s)]
    v0 = lin_const - bmin
    # (v0 + sum a_i x_i)^2 = v0^2 + sum (a_i^2 + 2 v0 a_i) x_i + sum_{i<j} 2 a_i a_j x_i x_j
    c += P * v0**2
    for k, (i, a) in enumerate(terms):
        add_term(Q, i, i, P * (a * a + 2 * v0 * a))
        for (j, b) in terms[k + 1:]:
            add_term(Q, i, j, P * 2 * a * b)
    return Q, c


def lin_bits(coef_lin):
    """Modèle affine en z -> forme affine en bits : renvoie (const, [(bit, coef), ...])."""
    from quantique.qubo import affine_bit_maps
    a0, A = affine_bit_maps(BITS, LEVELS)
    g = coef_lin[1:4]
    const = coef_lin[0] + g @ a0
    w = g @ A
    return const, [(i, w[i]) for i in range(len(w))]


def build_qubo(coef_cd, coef_ser, coef_cl, w, scale_cd, scale_ser, cons, P=3.0, n_slack=3):
    """QUBO de la somme pondérée + contraintes. Renvoie (Q, c, n_qubits)."""
    coef = w * coef_cd / scale_cd + (1 - w) * coef_ser / scale_ser
    Q, c = surrogate_to_qubo(coef, BITS, LEVELS)
    n = Q.shape[0]
    extra = (1 if cons.get("t_min_index") else 0) + (n_slack if cons.get("cl_min") is not None else 0)
    Q = grow(Q, extra)
    k = n
    if cons.get("t_min_index"):
        Q, c = penalty_index_nonzero(Q, c, bits=(6, 7, 8), y=k, P=P)
        k += 1
    if cons.get("cl_min") is not None:
        const, lin = lin_bits(coef_cl)
        lmax = const + sum(max(a, 0) for _, a in lin)
        Q, c = penalty_linear_geq(Q, c, lin, const, cons["cl_min"], list(range(k, k + n_slack)),
                                  max(lmax - cons["cl_min"], 1e-3), P * cons.get("cl_weight", 25.0))
        k += n_slack
    return Q, c, k


# ------------------------------------------------------------ solveurs
def decode9(b):
    b = np.asarray(b[:9]).astype(int)
    return tuple(int(sum(b[3 * v + j] << j for j in range(3))) for v in range(3))


def ranked_candidates(Q, c, solver, rng, qaoa_p=2, shots=512):
    """Liste de profils (indices) classés du meilleur au moins bon selon le QUBO."""
    n = Q.shape[0]
    if solver == "exact":
        B = all_bitstrings(n)
        E = qubo_energy(Q, c, B)
        order = np.argsort(E)
        states = B[order[:400]]
    elif solver == "sa":
        _, _, fin, st = simulated_annealing(Q, c, n_sweeps=150, n_reads=40,
                                            seed=int(rng.integers(1 << 30)), return_states=True)
        states = st[np.argsort(fin)]
    elif solver == "qaoa":
        B = all_bitstrings(n)
        E = qubo_energy(Q, c, B)
        En = qa.normalize(E)
        r = qa.optimize_qaoa(En, n, qaoa_p, n_restarts=3, seed=int(rng.integers(1 << 30)))
        psi = qa.qaoa_state(r.x[:qaoa_p], r.x[qaoa_p:], En, n)
        xs = qa.sample(psi, shots, seed=int(rng.integers(1 << 30)))
        uniq = np.unique(xs)
        uniq = uniq[np.argsort(E[uniq])]       # on recalcule l'énergie classiquement et on trie
        states = B[uniq]
    else:
        raise ValueError(solver)
    out, seen = [], set()
    for b in states:
        idx = decode9(b)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


# ------------------------------------------------------------ boucle
def qubo_loop(problem, n_init=10, weights=(0.0, 0.25, 0.5, 0.75, 1.0), k_per_w=1,
              solver="exact", budget=120, seed=0, cons=None, stop_when=None):
    """Boucle bi-objectif. cons : {'t_min_index': True, 'cl_min': 0.3} par exemple.

    stop_when(problem) -> bool permet d'arrêter dès que le front exact est retrouvé.
    Renvoie l'historique (nombre de simulations après chaque itération)."""
    cons = cons or {}
    rng = np.random.default_rng(seed)
    all_idx = [(a, b, c) for a in range(8) for b in range(8) for c in range(8)]
    for j in rng.permutation(len(all_idx))[:n_init]:
        problem(all_idx[j])
    history = [problem.n_sims]
    it = 0
    while problem.n_sims < budget:
        it += 1
        rows = [problem.table[i] for i in problem.seen]
        I = np.array([[r["im"], r["ip"], r["it"]] for r in rows])
        ok = np.array([r["feasible"] for r in rows])
        val = np.array([r["valid"] for r in rows])
        if ok.sum() < 4:
            # pas assez de points faisables : exploration aléatoire
            new = [all_idx[j] for j in rng.permutation(len(all_idx)) if all_idx[j] not in problem._set][:2]
        else:
            cd = np.array([r["CD_counts"] for r in rows])
            ser = np.array([r["SER_db"] for r in rows])
            cl = np.array([r["CL"] for r in rows])
            c_cd = fit_quadratic(I[ok], cd[ok])
            c_ser = fit_quadratic(I[ok], ser[ok])
            c_cl = fit_linear(I[val], cl[val])
            s_cd = max(np.ptp(cd[ok]), 1e-6)
            s_ser = max(np.ptp(ser[ok]), 1e-6)
            new = []
            for w in weights:
                Q, c, _ = build_qubo(c_cd, c_ser, c_cl, w, s_cd, s_ser, cons)
                taken = 0
                for idx in ranked_candidates(Q, c, solver, rng):
                    if idx not in problem._set and idx not in new:
                        new.append(idx)
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
