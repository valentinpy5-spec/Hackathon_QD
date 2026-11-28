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


# ---------------------------------------------------------------------------
# Le même circuit écrit avec PennyLane (pour comprendre ; non utilisé)
# ---------------------------------------------------------------------------
# qaoa_state ci-dessus calcule exactement ce que ferait le circuit ci-dessous :
# états identiques à 1e-16 près (vérifié pour n = 3 à 14, p = 1 à 3).
#
# import pennylane as qml
#
# def ising_terms(E, n):
#     """Étape 0 : réécrire E(x) en hamiltonien d'Ising.
#     Un ordinateur quantique ne sait pas « multiplier par exp(-i gamma E(x)) »
#     d'un coup : il lui faut des portes. On décompose donc E sur les produits
#     de Pauli Z :  E = sum_S c_S prod_{k in S} Z_k,  avec
#     c_S = 2^-n sum_x E(x) (-1)^(nombre de bits de x dans S)
#     (transformée de Walsh-Hadamard). Pour un QUBO, seuls les termes à 1
#     qubit (champs h_k, portes RZ) et à 2 qubits (couplages J_kl, portes ZZ)
#     sont non nuls : 1 + n + n(n-1)/2 termes au plus (105 pour n = 14).
#     Le terme constant (S vide) n'est qu'une phase globale : on l'ignore.
#     Convention : le fil 0 est le bit de poids FORT de l'indice x de psi."""
#     c = np.asarray(E, dtype=float).copy()
#     h = 1
#     while h < len(c):                       # Walsh-Hadamard rapide, n * 2^n opérations
#         c = c.reshape(-1, 2, h)
#         c = np.stack([c[:, 0] + c[:, 1], c[:, 0] - c[:, 1]], axis=1).reshape(-1)
#         h *= 2
#     c /= len(c)
#     return [([w for w in range(n) if (S >> (n - 1 - w)) & 1], c[S])
#             for S in np.nonzero(np.abs(c) > 1e-12)[0] if S != 0]
#
# def qaoa_state_pennylane(gammas, betas, E, n):
#     terms = ising_terms(E, n)
#     dev = qml.device("lightning.qubit", wires=n)   # simulateur de vecteur d'état
#
#     @qml.qnode(dev)
#     def circuit():
#         # Étape 1 : état initial |+>^n = superposition uniforme des 2^n solutions.
#         # C'est l'état fondamental du mélangeur -sum X_k (équivalent du
#         # psi = 2^(-n/2) partout dans la version NumPy).
#         for w in range(n):
#             qml.Hadamard(wires=w)
#         # Étape 2 : p couches alternées coût / mélangeur.
#         for g, b in zip(gammas, betas):
#             # 2a. U_C(gamma) = exp(-i gamma H_C) = prod_S exp(-i gamma c_S Z_S).
#             #     Les Z commutent entre eux, donc on applique les termes un par un.
#             #     MultiRZ(theta) = exp(-i theta/2 Z...Z), d'où theta = 2 gamma c_S :
#             #     sur 1 fil c'est une porte RZ, sur 2 fils une porte ZZ
#             #     (CNOT - RZ - CNOT sur une vraie machine).
#             #     Effet : chaque |x> prend la phase exp(-i gamma E(x)),
#             #     comme « psi * exp(-1j * g * E) » en NumPy.
#             for wires, c in terms:
#                 qml.MultiRZ(2 * g * c, wires=wires)
#             # 2b. U_M(beta) = exp(-i beta sum_k X_k) = prod_k RX(2 beta) sur chaque qubit
#             #     (RX(theta) = exp(-i theta/2 X)). Fait « interférer » les solutions
#             #     voisines (un bit de différence) : c'est apply_mixer en NumPy.
#             for w in range(n):
#                 qml.RX(2 * b, wires=w)
#         # Étape 3 : lire l'état. En simulation on récupère les 2^n amplitudes ;
#         # sur une vraie machine on ferait qml.sample() (mesures, règle de Born).
#         return qml.state()
#
#     return np.asarray(circuit())
#
# Pourquoi on ne l'utilise pas : PennyLane applique les ~240 portes d'un
# circuit p = 2 à 14 qubits une par une, avec ~0,15 ms de surcoût Python par
# porte (création de l'objet, enregistrement, envoi au simulateur). Résultat :
# ~40 ms par circuit contre ~4 ms en NumPy, qui applique toute la couche de
# coût en UNE multiplication (le coût est diagonal). Or la boucle QUBO en mode
# "qaoa" (optim/qubo_loop.py) évalue des milliers de circuits par itération :
# un run passerait d'environ 12 min à environ 2 h.
# À éviter aussi : qml.DiagonalQubitUnitary(np.exp(-1j * g * E)) sur les n fils,
# qui paraît plus direct mais construit une matrice 2^n x 2^n (~5 s par circuit
# à 14 qubits).


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
