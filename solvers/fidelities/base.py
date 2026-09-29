"""
The DataFidelity contract: D is the negative log-likelihood of the noise, per pixel.

    D(Hf, g) = - (1 / N_g) log p(g | f)  +  a constant chosen so that D(g, g) = 0

Written this way, three properties hold for every noise model, and together they are what
lets lambda_reg mean "how much prior" and nothing else:

    > D carries the noise LEVEL. Its parameters are the (a, b) of core/noise.py —
      Var(g) = a * Hf + b — so a noisier measurement is trusted less by construction,
      instead of asking lambda_reg to compensate.
    > D is a MEAN over the measurement's pixels, so it does not grow with the image size.
    > At the truth, D is about 1/2 whatever the model (the chi-square property of a
      correctly scaled likelihood). Gaussian and Poisson fidelities are therefore on the
      same scale, and the same lambda_reg means the same thing with either.

The parameters are floored at 1e-6: about the quantization noise of an 8-bit image
((1/255)^2 / 12). A noiseless measurement would otherwise make D infinite.
"""

import torch

## the smallest a or b a fidelity accepts (see the module docstring)
NOISE_FLOOR = 1e-6


class DataFidelity:
    """
    Base class for a noise model's data fidelity D(Hf, g).

    Class attributes (set by subclasses):
        name              short key ('gaussian', 'poisson', 'poisson-gaussian'), the one
                          solvers list in `supported_noise_models`
        display_name      the label shown in the interface and stored in config.toml
        noise_parameters  which of (a, b) this model uses
        formula           latex of D, symbolic (shown in the interface); n_g is the
                          number of pixels of g — D is always a mean
    """

    name = ""
    display_name = ""
    noise_parameters = ()
    formula = ""

    @classmethod
    def from_noise(cls, a: float, b: float) -> "DataFidelity":
        """The fidelity for a noise of parameters (a, b); the unused one is ignored."""
        values = {"a": a, "b": b}
        return cls(**{name: values[name] for name in cls.noise_parameters})

    def loss(self, Hf: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """D(Hf, g) -> scalar."""
        raise NotImplementedError

    def latex(self) -> str:
        """The formula with this instance's parameter values, for the interface and the log."""
        values = ",\\ ".join(f"{p} = {getattr(self, p):.3g}" for p in self.noise_parameters)
        return f"{self.formula},\\quad {values}" if values else self.formula

    def describe(self) -> str:
        values = ", ".join(f"{p}={getattr(self, p):.3g}" for p in self.noise_parameters)
        return f"{self.display_name} ({values})" if values else self.display_name


def floored(value: float) -> float:
    return max(float(value), NOISE_FLOOR)
