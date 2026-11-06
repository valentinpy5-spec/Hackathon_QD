"""Recuit simulé sur un QUBO (Metropolis, retournement d'un bit à la fois).

C'est l'algorithme classique dont le recuit quantique est l'analogue : on
sort des minima locaux par agitation thermique (ici) ou par effet tunnel
(recuit quantique). C'est aussi ce que fait `neal.SimulatedAnnealingSampler`
du D-Wave Ocean SDK.
"""
import numpy as np


def simulated_annealing(Q, c, n_sweeps=200, T0=None, T1=None, n_reads=20, seed=0, return_states=False):
    """Renvoie (meilleur b, meilleure énergie, énergies finales de chaque lecture).

    return_states=True ajoute en 4e sortie les états finaux de chaque lecture,
    comme le « sampleset » d'Ocean : utile pour proposer plusieurs candidats."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    S = np.triu(Q, 1)
    S = S + S.T                                   # couplages symétriques
    d = np.diag(Q)
    scale = np.abs(Q).max()
    T0 = T0 if T0 is not None else 2.0 * scale
    T1 = T1 if T1 is not None else 0.01 * scale
    temps = np.geomspace(T0, T1, n_sweeps)
    best_b, best_e, finals, states = None, np.inf, [], []
    for _ in range(n_reads):
        b = rng.integers(0, 2, n).astype(float)
        e = b @ np.triu(Q) @ b + c
        for T in temps:
            for i in rng.permutation(n):
                # variation d'énergie si on retourne le bit i
                dE = (1 - 2 * b[i]) * (d[i] + S[i] @ b)
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    b[i] = 1 - b[i]
                    e += dE
        finals.append(e)
        states.append(b.copy())
        if e < best_e:
            best_b, best_e = b.copy(), e
    if return_states:
        return best_b, best_e, np.array(finals), np.array(states)
    return best_b, best_e, np.array(finals)
