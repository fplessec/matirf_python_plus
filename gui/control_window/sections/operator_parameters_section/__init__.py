"""
This package contains two files:
    > ‘qgroup_operator_parameters.py’ with the OperatorParametersSection class, a qgroup for the 'oper-params'
      parameter set
    > 'operator_ui_dictionary.py' with OPERATOR_PARAMETERS_UI, a specific dictionary for the uses of
      SimpleParameterWidget objects (see single_parameter_widget.py from the control_window package)

The 'oper-params' parameter set contains the following parameters:
    > nz: the number of cuts of the reconstructed image on the z axis
    > z0: the smallest depth on z of the reconstructed image, in nanometers
    > zN: the largest depth on z of the reconstructed image, in nanometers
    > normalize: a boolean to choose whether to normalize the operator
"""


from .qgroup_operator_parameters import OperatorParametersSection