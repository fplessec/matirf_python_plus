"""
Two-level object model, mirroring the algorithm architecture.

    ContinuousObject      the density carrier — a non-negative field in nm space,
                          with a single method density(X, Y, Z). Grid-independent.

    SyntheticObjectType   the "type descriptor" — like an Algorithm class: it has a
                          display 'name', a TOML section key 'toml_key', a parameter
                          UI dictionary 'ui_params' (rendered in the GUI exactly like an
                          algorithm's ui_params), and a sampler that turns a params dict
                          into concrete ContinuousObject instances.

A registry OBJECT_TYPES (see objects/__init__.py) maps toml_key -> SyntheticObjectType,
just as ALGORITHMS maps name -> Algorithm. The ground-truth TOML has one '[toml_key]'
section per type, each carrying a 'count' and the type's characteristic parameters.
"""


class ContinuousObject:
    """
    Non-negative density field defined in physical nm space.

    No shared state: the contract (nm coordinates in, density >= 0 out) is what lets the
    generator sum/integrate any mixture of primitives on any grid without knowing their
    concrete type. Density must be non-negative (a physical fluorophore density; the
    reconstruction algorithms assume positivity).
    """

    def density(self, X, Y, Z):
        raise NotImplementedError


class SyntheticObjectType:
    """
    Base class for a synthetic object TYPE (ellipsoid, filament, ...).

    Class attributes (set by subclasses):
        name       : human-readable label (e.g. "Ellipsoid")
        toml_key   : TOML section key (e.g. "ellipsoid")
        ui_params  : parameter UI dictionary — same format as an algorithm's ui_params.
                     Always contains a "count" parameter (how many instances to draw).
    """

    name = ""
    toml_key = ""
    ui_params = {}

    @classmethod
    def get_ui_params(cls) -> dict:
        """Returns the UI dictionary for this type (kept for symmetry with algorithms)."""
        return cls.ui_params

    @classmethod
    def default_params(cls) -> dict:
        """Default value of every parameter, read from the UI dictionary."""
        return {name: _default_of(cfg) for name, cfg in cls.ui_params.items()}

    @classmethod
    def sample(cls, params: dict, gen, grid) -> ContinuousObject:
        """Draws ONE ContinuousObject instance from the params dict. Override this."""
        raise NotImplementedError

    @classmethod
    def build(cls, params: dict, gen, grid) -> list:
        """
        Draws 'count' instances (count read from params; 0 or missing -> none).

        Missing characteristic parameters fall back to the type's defaults, so a pruned
        TOML section that only carries 'count' still generates correctly.
        """
        count = int(params.get("count", 0) or 0)
        if count <= 0:
            return []
        full = {**cls.default_params(), **params}
        return [cls.sample(full, gen, grid) for _ in range(count)]


def _default_of(param_config: dict):
    """Default value of a single UI parameter entry (value / bool / option)."""
    if param_config["type"] == "option":
        return param_config["param_info"]["options_list"][0]
    return param_config["param_info"]["default"]


# ── small helpers to declare UI parameters concisely ─────────────────────────

def value_param(title, latex_name, default, unit="", dtype=float):
    """A one-line 'value' UI parameter (number input + LaTeX display)."""
    return {"title": title, "type": "value",
            "param_info": {"dtype": dtype, "unit": unit,
                           "latex_name": latex_name, "default": default}}


def count_param(noun_plural, default=5):
    """The mandatory 'count' UI parameter (how many instances of this type)."""
    return value_param(f"Number of {noun_plural}", "N", default, unit="", dtype=int)


# ── count-gating: hide (and drop from the TOML) every parameter when count == 0 ──

def _count_positive(value) -> bool:
    """depends_on predicate: a parameter is shown only when its section's count > 0."""
    try:
        return (value or 0) > 0
    except TypeError:
        return False


def gate_by_count(ui_params: dict) -> dict:
    """
    Returns a copy of 'ui_params' where every parameter EXCEPT 'count' is made to depend
    on 'count > 0' (via depends_on). In the GUI the whole parameter set then appears only
    when the user sets count to a strictly positive value, and collapses when count is 0.
    """
    gated = {}
    for name, config in ui_params.items():
        if name == "count":
            gated[name] = config
            continue
        new = dict(config)
        deps = dict(new.get("depends_on", {}))
        deps.setdefault("count", _count_positive)
        new["depends_on"] = deps
        gated[name] = new
    return gated
