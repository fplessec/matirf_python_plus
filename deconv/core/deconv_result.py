"""
Dataclass encapsulant TOUT ce qui sort d'une déconvolution 2D.

Même rôle que core/reconstruction_result.py de matirf, mais spécialisée pour
la deconv 2D : pas de 'delta' (anisotropie), images 2D au lieu de 3D, et la
PSF est appelée H (par cohérence avec la notation g = H @ f).

Convention :
    - 'f'       : la reconstruction (image déconvoluée)
    - 'H'       : la PSF utilisée comme opérateur
    - 'g'       : l'image d'entrée (mesure floutée + bruit éventuel)
    - 'f_true' / 'metrics' / 'diff' : présents uniquement en mode SYNTHETIC
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeconvResult:
    """Résultat complet d'une déconvolution 2D."""

    # --- Toujours présents après une run réussie ---
    f: Optional[torch.Tensor] = None    # image déconvoluée (2D)
    H: Optional[torch.Tensor] = None    # PSF (2D, petite : e.g. 21x21)
    g: Optional[torch.Tensor] = None    # image d'entrée (2D, taille image complète)

    # --- Présents en mode SYNTHETIC uniquement ---
    f_true: Optional[torch.Tensor] = None    # image nette de référence
    metrics: Optional[dict] = None           # SSIM, PSNR, etc.
    diff: Optional[torch.Tensor] = None      # f_true - alpha*f (pour visualisation)

    # --- Métadonnées d'exécution ---
    messages: str = ""

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def is_complete(self) -> bool:
        """True si f est disponible (déconvolution réellement effectuée)."""
        return self.f is not None

    def has_synthetic_truth(self) -> bool:
        """True si on a la vérité terrain pour comparaison."""
        return self.f_true is not None
