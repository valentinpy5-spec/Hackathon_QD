"""NSGA-II écrit à la main, pour variables entières (niveaux de la grille).

Pourquoi le réécrire alors que pymoo existe : pour le comprendre (le cours le
détaille ligne à ligne) et pour contrôler exactement ce qui est compté. Le
script optim/check_pymoo.py vérifie qu'on obtient le même comportement que
pymoo sur le même problème.

Algorithme (Deb, Pratap, Agarwal, Meyarivan, 2002) :
    1. population initiale aléatoire de N individus ;
    2. à chaque génération : sélection par tournoi binaire (rang, puis encombrement),
       croisement SBX + mutation polynomiale sur les indices relâchés en réels,
       arrondi, élimination des doublons -> N enfants ;
    3. survie élitiste : parents + enfants (2N) triés par dominance contrainte,
       on remplit front par front, le dernier front est départagé par la
       distance d'encombrement.
Contraintes : règle de Deb (faisable > infaisable ; entre infaisables, la plus
petite violation gagne ; entre faisables, la dominance de Pareto).
"""
import numpy as np

from .pareto import non_dominated_sort, crowding_distance


def constrained_ranks(F, CV):
    """Rang et encombrement selon la dominance contrainte de Deb."""
    n = len(F)
    rank = np.full(n, -1)
    crowd = np.zeros(n)
    feas = np.where(CV <= 0)[0]
    r = 0
    if len(feas):
        for front in non_dominated_sort(F[feas]):
            ids = feas[front]
            rank[ids] = r
            crowd[ids] = crowding_distance(F[ids])
            r += 1
    infeas = np.where(CV > 0)[0]
    for i in infeas[np.argsort(CV[infeas])]:
        rank[i] = r
        crowd[i] = 0.0
        r += 1
    return rank, crowd


def sbx(p1, p2, lo, hi, eta, rng):
    """Croisement binaire simulé, variable par variable (probabilité 1/2 par variable)."""
    c1, c2 = p1.copy(), p2.copy()
    for k in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[k] - p2[k]) < 1e-12:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        m = 0.5 * (p1[k] + p2[k])
        d = 0.5 * abs(p1[k] - p2[k])
        c1[k], c2[k] = m - beta * d, m + beta * d
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def poly_mutation(x, lo, hi, eta, pm, rng):
    """Mutation polynomiale de Deb : petit saut autour de x, de loi en puissance."""
    y = x.copy()
    for k in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            dq = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[k] = x[k] + dq * (np.broadcast_to(hi, x.shape)[k] - np.broadcast_to(lo, x.shape)[k])
    return np.clip(y, lo, hi)


def nsga2(problem, pop_size=16, n_gen=40, seed=0, eta_c=15.0, eta_m=20.0, p_c=0.9,
          n_levels=(8, 8, 8), max_sims=None, callback=None):
    """Lance NSGA-II sur un CountedProblem. Renvoie l'historique par génération."""
    rng = np.random.default_rng(seed)
    nv = len(n_levels)
    lo, hi = -0.5, np.array(n_levels) - 0.5          # relaxation continue des indices
    pm = 1.0 / nv

    def to_idx(x):
        return tuple(int(np.clip(np.rint(v), 0, L - 1)) for v, L in zip(x, n_levels))

    def evaluate(X):
        F, CV = [], []
        for x in X:
            f, cv = problem.F(to_idx(x))
            F.append(f)
            CV.append(cv)
        return np.array(F), np.array(CV)

    # population initiale : indices distincts tirés uniformément
    seen = set()
    X = []
    while len(X) < pop_size:
        idx = tuple(int(rng.integers(L)) for L in n_levels)
        if idx not in seen:
            seen.add(idx)
            X.append(np.array(idx, dtype=float))
    X = np.array(X)
    F, CV = evaluate(X)
    rank, crowd = constrained_ranks(F, CV)
    history = [problem.n_sims]
    if callback:
        callback(0, problem)

    for gen in range(1, n_gen + 1):
        # ---- reproduction
        def tournament():
            i, j = rng.integers(len(X), size=2)
            if rank[i] != rank[j]:
                return i if rank[i] < rank[j] else j
            return i if crowd[i] > crowd[j] else j

        pop_keys = {to_idx(x) for x in X}
        children, keys = [], set()
        tries = 0
        while len(children) < pop_size and tries < 50 * pop_size:
            tries += 1
            a, b = X[tournament()], X[tournament()]
            if rng.random() < p_c:
                c1, c2 = sbx(a, b, lo, hi, eta_c, rng)
            else:
                c1, c2 = a.copy(), b.copy()
            for c in (c1, c2):
                c = poly_mutation(c, lo, hi, eta_m, pm, rng)
                k = to_idx(c)
                if k in pop_keys or k in keys:       # élimination des doublons
                    continue
                keys.add(k)
                children.append(np.array(k, dtype=float))
                if len(children) == pop_size:
                    break
        if not children:
            break
        Xc = np.array(children)
        Fc, CVc = evaluate(Xc)
        # ---- survie élitiste
        XA, FA, CVA = np.vstack([X, Xc]), np.vstack([F, Fc]), np.concatenate([CV, CVc])
        rA, cA = constrained_ranks(FA, CVA)
        order = np.lexsort((-cA, rA))                 # rang croissant, puis encombrement décroissant
        keep = order[:pop_size]
        X, F, CV = XA[keep], FA[keep], CVA[keep]
        rank, crowd = constrained_ranks(F, CV)
        history.append(problem.n_sims)
        if callback:
            callback(gen, problem)
        if max_sims and problem.n_sims >= max_sims:
            break
    return dict(X=X, F=F, CV=CV, history=history)
