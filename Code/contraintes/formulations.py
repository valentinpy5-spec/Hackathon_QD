"""Itération V3, lot L7 : trois façons de mettre une contrainte h(b) >= 0 dans la boucle QUBO.

Toutes les contraintes sont écrites sous forme QUBO : h(b) = b^T H b + h0 (H triangulaire supérieure),
sur les seuls bits de conception. Une contrainte affine en z (le modèle de C_L,max) a un H diagonal ;
le classifieur de convergence a un H plein (contrainte quadratique).

    ecart          (V2, référence) : P (h(b) - s)^2 avec s >= 0 codé sur 3 bits d'écart. Exige h affine
                   (sinon degré 4), ajoute 3 qubits, et la pénalité domine le spectre.
    desequilibree  (Montañez-Barrera et al., 2024) : P (h^2 - 2 h* h), sans bit d'écart. Nulle en h = 0,
                   minimale en h = h* > 0, positive pour h < 0. Exige h affine. Défaut connu : pénalise
                   aussi les profils très faisables (h grand), donc déforme l'objectif.
    lagrange       objectif - sum_k lambda_k h_k(b) / s_k : un terme LINÉAIRE en h, donc de degré 2 même
                   pour une contrainte quadratique. Les multiplicateurs sont ajustés en quelques
                   résolutions (montée duale : partis de presque zéro, multipliés si la meilleure
                   lecture viole la contrainte, divisés sinon), puis chaque lecture est VÉRIFIÉE en classique : on garde la
                   meilleure, au sens de l'objectif seul, parmi celles qui respectent toutes les
                   contraintes prédites. Les multiplicateurs servent à orienter les lectures, la
                   vérification garantit la faisabilité prédite.

Le post-traitement « vérifier les lectures » est légitime pour un solveur quantique : un recuit
quantique ou un circuit QAOA rend lui aussi une liste d'échantillons.
"""
import numpy as np

from quantique.qubo import surrogate_to_qubo, qubo_energy


class Contrainte:
    """h(b) >= 0, sous forme QUBO sur les bits de conception. s : échelle typique de h (normalisation)."""

    def __init__(self, nom, H, h0, s=1.0, verifiee=True):
        """verifiee=False : la contrainte oriente le solveur (terme lagrangien) mais n'entre pas dans la
        vérification des lectures, confiée à un filtre exact (géométrie)."""
        self.nom, self.H, self.h0, self.s = nom, np.asarray(H, float), float(h0), max(float(s), 1e-9)
        self.verifiee = verifiee

    @classmethod
    def depuis_z(cls, nom, coef, bits, levels, seuil=0.0, s=1.0, verifiee=True):
        """h(z) = polynôme(coef) - seuil, coefficients au format de quadratic_features."""
        coef = np.array(coef, float)
        coef[0] -= seuil
        H, h0 = surrogate_to_qubo(coef, bits, levels)
        return cls(nom, H, h0, s, verifiee)

    @property
    def affine(self):
        return not np.any(np.triu(self.H, 1))

    def h(self, B):
        """Valeur de h sur des chaînes dont les premiers bits sont ceux de la conception."""
        B = np.atleast_2d(B)
        return qubo_energy(self.H, self.h0, B[:, :self.H.shape[0]])


def penalite_desequilibree(Q, c, ctr, P=75.0, h_etoile=0.05):
    """Ajoute P (h^2 - 2 h* h) : pas de qubit supplémentaire, mais h doit être affine."""
    assert ctr.affine, "pénalité déséquilibrée : contrainte affine requise (sinon degré 4)"
    n = Q.shape[0]
    w = np.diag(ctr.H).copy()                # h = h0 + w . b
    h0 = ctr.h0
    Q = Q.copy()
    n = w.size
    # h^2 = h0^2 + sum (w_i^2 + 2 h0 w_i) b_i + sum_{i<j} 2 w_i w_j b_i b_j
    iu, ju = np.triu_indices(n, 1)
    Q[iu, ju] += P * 2 * w[iu] * w[ju]
    Q[np.diag_indices(n)] += P * (w * w + 2 * h0 * w) - P * 2 * h_etoile * w
    c = c + P * (h0 * h0 - 2 * h_etoile * h0)
    return Q, c


def lagrangien(Q, c, ctrs, lam):
    """Objectif - sum lambda_k h_k / s_k."""
    Q = Q.copy()
    for k, ct in enumerate(ctrs):
        n = ct.H.shape[0]
        Q[:n, :n] -= lam[k] * ct.H / ct.s
        c = c - lam[k] * ct.h0 / ct.s
    return Q, c


def verifie(B, ctrs, tol=0.0):
    """Masque des lectures qui respectent toutes les contraintes prédites."""
    ok = np.ones(len(B), bool)
    for ct in ctrs:
        ok &= ct.h(B) >= -tol
    return ok


def resout_lagrange(Qo, co, ctrs, solve, lam0=None, n_tours=6, lam_min=0.05, filtre=None, lam_init=None,
                    hausse=2.0, baisse=2.0):
    """Montée duale sur les multiplicateurs + vérification des lectures.

    solve(Q, c) -> (états (n_reads, n), énergies). Renvoie (lectures classées, lambda final, info) :
    d'abord les lectures faisables (prédites) par objectif croissant, puis les autres par violation
    croissante. lam0 : multiplicateurs de départ (ceux de l'itération précédente, pour le même poids).
    filtre(S) -> masque : contrainte connue sans modèle (géométrie), vérifiée sur les lectures ; la montée
    duale regarde alors la meilleure lecture QUI PASSE le filtre."""
    K = len(ctrs)
    lam = np.full(K, lam_min if lam_init is None else lam_init)
    if lam0 is not None:                     # une contrainte peut apparaître en cours de boucle
        lam[:min(K, len(lam0))] = np.asarray(lam0)[:K]
    pool = []
    for _ in range(n_tours):
        Q, c = lagrangien(Qo, co, ctrs, lam)
        S, E = solve(Q, c)
        pool.append(S)
        m = filtre(S) if filtre is not None else np.ones(len(S), bool)
        if not m.any():
            continue
        best = S[m][np.argmin(E[m])][None]
        for k, ct in enumerate(ctrs):
            if ct.h(best)[0] < 0:
                lam[k] = max(hausse * lam[k], lam_min)
            else:
                lam[k] = lam[k] / baisse
    S = np.unique(np.vstack(pool), axis=0)
    obj = qubo_energy(Qo, co, S)
    viol = np.zeros(len(S))
    for ct in ctrs:
        if not ct.verifiee:
            continue
        viol += np.maximum(0.0, -ct.h(S)) / ct.s
    ok = viol <= 0
    if filtre is not None:
        adm = filtre(S)
        ok &= adm
        viol = viol + (~adm) * 1e6           # les lectures hors filtre en dernier
    ordre = np.concatenate([np.where(ok)[0][np.argsort(obj[ok])], np.where(~ok)[0][np.argsort(viol[~ok])]])
    return S[ordre], lam, dict(n_lectures=len(S), part_faisable=float(ok.mean()))


def part_objectif(Qo, co, Q, c, B):
    """Part de l'objectif dans l'étendue des énergies du QUBO complet, sur un ensemble de chaînes B
    (bits de conception d'abord ; les bits auxiliaires éventuels suivent)."""
    Eo = qubo_energy(Qo, co, B[:, :Qo.shape[0]])
    E = qubo_energy(Q, c, B)
    return float(np.ptp(Eo) / max(np.ptp(E), 1e-12))
