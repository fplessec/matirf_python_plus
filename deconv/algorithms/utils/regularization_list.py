"""
Liste des régularisations disponibles pour la déconvolution 2D.

Sous-ensemble simplifié de algorithms/utils/regularization_list.py de matirf :
on ne garde que les régularisations classiques d'image 2D (pas de Hessien, pas
de SHV qui sont plus utiles en 3D MA-TIRF).
"""


REGULARIZATION_LIST = [
    "no regularization",
    "L2 norm",
    "L1 norm",
    "L2 norm of the gradient",      # Tikhonov
    "L1 norm of the gradient",      # Total Variation
]
