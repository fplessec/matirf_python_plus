"""
Algorithmes de déconvolution 2D + dictionnaire ALGORITHMS pour la GUI.

Pour ajouter un nouvel algorithme :
    1. créer un sous-paquet (ex: optim_admm/) avec algo + ui_dictionnary
    2. l'importer ici
    3. l'ajouter à ALGORITHMS avec son nom d'affichage

Le dictionnaire ALGORITHMS suit la même convention que algorithms/__init__.py
de matirf :
    {
        ALGORITHM_NAME: {
            "object": ALGORITHM_OBJECT,
            "ui_params": ALGORITHM_UI_PARAMETERS_DICT
        },
        ...
    }
"""

from .optim_adam import AdamAlgo, ADAM_UI_PARAMETERS


ALGORITHMS = {
    "ADAM": {
        'object': AdamAlgo,
        'ui_params': ADAM_UI_PARAMETERS
    },
}
