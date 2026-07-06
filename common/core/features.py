"""
Central catalogue of features that characterize inverse problems.

Each inverse problem package (matirf/, deconv/) defines its own set
of features. Algorithms, metrics, regularizations, and UI parameters
use these constants to adapt their behavior.
"""

## dimensionality:
THREE_D = "3d"
TWO_D = "2d"

## problem properties:
SCALE_AMBIGUOUS = "scale_ambiguous"
ANISOTROPIC = "anisotropic"
