class DataFidelity:
    """
    Base class for data fidelity terms D(Hf, g).

    Each subclass represents a noise model and provides:
        - loss(Hf, g)  : the data fidelity value (for gradient-based algorithms)
        - prox(...)     : the proximal operator (for proximal algorithms)
    """

    name = ""
    display_name = ""
    noise_model = ""

    def loss(self, Hf, g):
        """D(Hf, g) -> scalar."""
        raise NotImplementedError

    def prox(self, Hf, g, **kwargs):
        """Proximal operator of the data fidelity term."""
        raise NotImplementedError
