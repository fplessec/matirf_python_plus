"""
Sous-projet 'deconv' : déconvolution d'images 2D PNG en niveaux de gris.

Package autonome qui vit côte à côte de matirf. Réutilise quelques briques
génériques (gui.SimpleParameterWidget, gui.more_widgets, in_out racine, settings)
mais possède son propre core, ses propres algorithmes, sa propre GUI, et son
propre cache.

Lancement :
    python deconv/main.py gui                       # mode interactif
    python deconv/main.py cli -c <conf> -o <out>    # mode batch
"""
