from .differential_operators import DifferentialOperators
from .loss_computer import LossComputer
from .regularization_list import REGULARIZATION_LIST

"""
δ = Δz / Δxy << 1
delta est le ratio de la taille des pixels sur la direction z par rapport à celle sur les directions xy.
Si on veut régulariser le gradient spacial de l'image 3D f super-résolue sur z, il faut prendre en compte l'anisotropie !
> || grad(f) ||² = dz² + dy² + dx²
  ici on considère une variation entre deux pixels sur z aussi importante qu'une variation entre deux pixels sur x/y
  or entre deux pixels sur x/y on peut mettre parfois 10 à 30 pixels sur z
  ainsi en regularisant on regularise de la meme manière sur xy que sur z donc on sur-régularise la direction z
  (sur-lissage etc)
  autrement dit, la direction z étant plus fine elle est beaucoup plus sensible à la régularisation
> || grad(f) ||² = Δz² * dz² +  Δxy² * dy² + Δxy² * dx²  ou bien:
  || grad(f) ||² = δ² * dz² +  dy² + dx²
  ici en prenant en compte l'anisotropie on corrige la sur-régularisation sur z qui peut  etre responsable de la
  destruction de la super-résolution

Comment calculer delta ?
Δz = (zN - z0) / nz  est choisi par l'utilisateur
Δxy = FOV_camera / Npixel_xy
Si FOV_camera est inconnu, on peut estimer Δxy à partir des paramètres de mesures sous l'hypothèse suivante:
la résolution est limitée par la diffraction et donc la taille du pixel est bien choisie pour echantillonnée:
Δxy ≈ Δxy_Rayleigh / 2  (Nyquist)
et :
Δxy_Rayleigh ≈ 0.61 λ / n_medium / NA_obj  qui sont dans les paramètres de mesure (indice optique, longueur d'onde,
ouverture numérique, ...)
Comme on veut juste un ordre de grandeur pour delta on va ignoré Nyquist et prendre Δxy ≈ Δxy_Rayleigh

On a donc:
δ ≈ (zN - z0) / nz * n_medium * NA_obj / 0.61 / λ
"""