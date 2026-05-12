"""
Encapsule TOUT ce qui sort d'une reconstruction MA-TIRF dans un objet typé.

Avant ce refactor, MaTirfPipeline mémorisait pêle-mêle f, f_true, H, g, metrics,
delta, messages, etc. comme des attributs dispersés. Difficile à tracer, à
sérialiser, à passer en bloc à un consommateur.

Après : un seul objet `ReconstructionResult` contient tout ce que produit le
pipeline. Il peut être passé entier au DisplayWindow, sauvegardé/chargé via
result_io.py, etc.

Convention :
    - 'f'           : la reconstruction principale (toujours présente après succès)
    - 'H', 'g'      : opérateur et mesure préprocessée (toujours présents après setup)
    - 'f_true', 'metrics', 'delta', 'diff' : présents uniquement en mode synthétique
    - 'messages'    : journal accumulé pendant la run (utile pour la sauvegarde)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ReconstructionResult:
    """Résultat complet d'une reconstruction MA-TIRF."""

    # --- Toujours présents après une run réussie ---
    f: Optional[torch.Tensor] = None    # reconstruction principale
    H: Optional[torch.Tensor] = None    # opérateur MA-TIRF construit
    g: Optional[torch.Tensor] = None    # mesure préprocessée

    # --- Présents en mode SYNTHETIC uniquement ---
    f_true: Optional[torch.Tensor] = None    # vérité terrain (image de référence)
    metrics: Optional[dict] = None           # métriques de qualité (SSIM, etc.)
    delta: Optional[float] = None            # ratio anisotropie Δz/Δxy
    diff: Optional[torch.Tensor] = None      # f_true - alpha*f (pour visualisation)

    # --- Métadonnées d'exécution ---
    messages: str = ""    # log textuel cumulé pendant la run

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def is_complete(self) -> bool:
        """True si f est disponible (reconstruction réellement effectuée)."""
        return self.f is not None

    def has_synthetic_truth(self) -> bool:
        """True si on a la vérité terrain pour comparaison."""
        return self.f_true is not None
