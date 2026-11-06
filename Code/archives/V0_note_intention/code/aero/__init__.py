"""Modèle aérodynamique de la phase 1 du hackathon (panneaux + couche limite intégrale)."""
from .naca4 import naca4, naca4_from_code, leading_edge_radius
from .panel import solve
from .model import evaluate, evaluate_at_cl, alpha_for_cl, AeroResult
