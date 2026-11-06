"""Du modèle quadratique de substitution au QUBO, puis au modèle d'Ising.

Conventions (identiques à aero.design_space.DesignSpace.encode) :
    - vecteur de bits b = (b_m0, b_m1, b_m2, b_p0, b_p1, b_p2, b_t0, b_t1, b_t2),
      poids faible en premier pour chaque variable ;
    - QUBO : E(b) = b^T Q b + c, avec Q triangulaire supérieure (diagonale =
      termes linéaires, car b_i^2 = b_i) ;
    - Ising : b_i = (1 - s_i)/2, s_i = +1 pour |0>, -1 pour |1> (valeur
      propre de Z), E = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + offset.
"""
import numpy as np


def affine_bit_maps(bits_per_var, n_levels):
    """Chaque variable normalisée z_i = 2 x_i/(L-1) - 1 écrite z_i = a0_i + A_i . b.

    Renvoie (a0 (d,), A (d, n_bits)).
    """
    n = sum(bits_per_var)
    d = len(bits_per_var)
    a0 = np.full(d, -1.0)
    A = np.zeros((d, n))
    k = 0
    for i, (nb, L) in enumerate(zip(bits_per_var, n_levels)):
        for j in range(nb):
            A[i, k + j] = 2.0 * 2**j / (L - 1)
        k += nb
    return a0, A


def quadratic_features(z):
    """[1, z_i, z_i z_j (i<=j)] : 10 colonnes pour d = 3."""
    z = np.atleast_2d(z)
    d = z.shape[1]
    cols = [np.ones(len(z))] + [z[:, i] for i in range(d)]
    cols += [z[:, i] * z[:, j] for i in range(d) for j in range(i, d)]
    return np.column_stack(cols)


def surrogate_to_qubo(coef, bits_per_var, n_levels):
    """Polynôme quadratique en z (coefficients de quadratic_features) -> (Q, c).

    On écrit f(z) = c0 + g.z + z^T M z, puis on substitue z = a0 + A b :
        f = c0 + g.a0 + a0^T M a0 + (g + 2 M a0)^T A b + b^T (A^T M A) b
    et on replie les termes linéaires sur la diagonale (b_i^2 = b_i).
    """
    a0, A = affine_bit_maps(bits_per_var, n_levels)
    d = len(a0)
    c0, g = coef[0], np.asarray(coef[1:1 + d])
    M = np.zeros((d, d))
    k = 1 + d
    for i in range(d):
        for j in range(i, d):
            if i == j:
                M[i, i] = coef[k]
            else:
                M[i, j] = M[j, i] = coef[k] / 2
            k += 1
    const = c0 + g @ a0 + a0 @ M @ a0
    lin = (g + 2 * M @ a0) @ A
    P = A.T @ M @ A                       # symétrique
    Q = np.triu(2 * P, 1) + np.diag(np.diag(P) + lin)
    return Q, const


def qubo_energy(Q, c, B):
    """Énergie pour un ou plusieurs vecteurs de bits (lignes de B)."""
    B = np.atleast_2d(B)
    return np.einsum("ki,ij,kj->k", B, Q, B) + c


def all_bitstrings(n):
    """Matrice (2^n, n) : ligne x = bits de l'entier x, bit k = (x >> k) & 1."""
    x = np.arange(2**n)
    return ((x[:, None] >> np.arange(n)) & 1).astype(float)


def brute_force(Q, c):
    B = all_bitstrings(Q.shape[0])
    E = qubo_energy(Q, c, B)
    return B, E


def qubo_to_ising(Q, c):
    """b = (1 - s)/2  ->  (h, J, offset), J triangulaire supérieure."""
    n = Q.shape[0]
    Qs = np.triu(Q, 1)
    h = -0.5 * np.diag(Q) - 0.25 * (Qs.sum(axis=1) + Qs.sum(axis=0))
    J = 0.25 * Qs
    offset = c + 0.5 * np.diag(Q).sum() + 0.25 * Qs.sum()
    return h, J, offset


def ising_energy(h, J, offset, S):
    S = np.atleast_2d(S)
    return S @ h + np.einsum("ki,ij,kj->k", S, J, S) + offset
