"""
This package contains two files:
    > ‘qgroup_algorithm_and_algo_params.py’ with the AlgorithmAndAlgoParamsSection class, a qgroup for the:
        - the parameter 'algorithm': the name of the algorithm used for the reconstruction
        - the parameter set 'algo-params': the associated parameters of the precedent chosen algorithm, the list of
          parameter from 'algo-params' depends on the 'algorithm' parameter
    > 'qwidget_algorithm_parameters.py' with OPERATOR_PARAMETERS_UI, a specific dictionary for the uses of
      SimpleParameterWidget objects (see single_parameter_widget.py from the control_window package)

The 'algo-params' parameter set contains are different depending on the chosen algorithm, and the detail of each
possible set is written inside the algorithms package, on each different algorithm there is a dictionary_ui that
defines the set of parameter of each particular algorithms.
"""


from .qgroup_algorithm_and_algo_params import AlgorithmAndAlgoParamsSection