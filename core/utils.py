import inspect


def get_variables_from_dict(dico: dict, variable_name_list: list) -> tuple:
    """Unpacks values from a dictionary in the order given by variable_name_list."""
    return tuple(dico[key] for key in variable_name_list)


def extract_init_kwargs(cls, params):
    """Extract from params the keyword arguments accepted by cls.__init__."""
    sig = inspect.signature(cls.__init__)
    return {k: params[k] for k in sig.parameters if k != 'self' and k in params}
