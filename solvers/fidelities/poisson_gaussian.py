import torch

from .base import DataFidelity, NOISE_FLOOR, floored


class PoissonGaussianFidelity(DataFidelity):
    """
    Photon noise plus read noise: Var(g) = a * Hf + b.

        D(Hf, g) = mean( (Hf - g)^2 / (2 v) + 1/2 log(v / v_g) ),
                   v = a Hf + b,  v_g = a g + b

    The Gaussian approximation of the Poisson-Gaussian likelihood (exact for many photons),
    with a variance that depends on the signal — hence the log term, which is what stops
    the fit from inflating Hf to explain the residual as noise. The constant log(v_g) only
    makes D(g, g) = 0. Covers the two pure cases at the limits: a -> 0 Gaussian, b -> 0
    Poisson.
    """

    name = "poisson-gaussian"
    display_name = "Poisson-Gaussian"
    noise_parameters = ("a", "b")
    formula = (r"D(Hf,g) = \overline{\frac{(Hf - g)^2}{2\,(a\,Hf + b)}"
               r" + \frac{1}{2}\log\frac{a\,Hf + b}{a\,g + b}}")

    def __init__(self, a: float = 1.0, b: float = 1.0):
        ## either part may be absent, but not both: the variance must stay positive
        self.a = max(float(a), 0.0)
        self.b = max(float(b), 0.0)
        if self.a + self.b < NOISE_FLOOR:
            self.b = floored(self.b)

    def loss(self, Hf, g):
        variance = (self.a * Hf + self.b).clamp(min=NOISE_FLOOR)
        variance_g = (self.a * g + self.b).clamp(min=NOISE_FLOOR)
        return ((Hf - g) ** 2 / (2 * variance) + 0.5 * torch.log(variance / variance_g)).mean()
