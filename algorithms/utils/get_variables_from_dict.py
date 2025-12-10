def get_variables_from_dict(dico, variable_name_list):
    return tuple(dico[key] for key in variable_name_list)