"""
Sous-projet 'deconv' : déconvolution d'images 2D PNG en niveaux de gris.

Package autonome qui vit côte à côte de matirf. Réutilise quelques briques
génériques (gui.SimpleParameterWidget, gui.more_widgets, in_out racine, settings)
mais possède son propre core, ses propres algorithmes, sa propre GUI, et son
propre cache.

Lancement :
    python main.py deconv gui                       # mode interactif
    python main.py deconv cli -c <conf> -o <out>    # mode batch
"""
