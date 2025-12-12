"""
This package contains two files:
    > ‘qgroup_operator_parameters.py’ with the OperatorParametersSection class, a qgroup for the 'oper-params'
      parameter set
    > 'operator_ui_dictionary.py' with OPERATOR_PARAMETERS_UI, a specific dictionary for the uses of
      SimpleParameterWidget objects (see

    SimpleParameterWidget


The specificity of the ‘input-path’ parameter set is the possibility of using two modes, with its ‘mode’ parameter:
    > mode = “real-data”: the tif file is a real MA-TIRF measurement stack, which I denote by the letter g:
      g is a  3D tensor from the mathematical set of the MA-TIRF measurement stacks.
      With this mode, the json file contains the measurement parameters used for this measurement g.
      After computing the MA-TIRF operator denoted by H, the reconstruction problem can be presented as:
      which is the 'best' possible reconstructed image f that would satisfy g = H*f
    > mode = "synthetic-data": the tif file is a made up synthetic 3D object, which I denote by the letter f:
      f is a 3D tensor from the mathematical set of the desired reconstructed image.
      More precisely, I will note it f_true in that case.
      With this mode, the json file contains made up measurement parameters to simulate a MA-TIRF measurement stack g
      by computing g = H*f_true.
      Then after solving the reconstruction problem, the user can compare the solution f with the ground truth f_true.
"""


from .qgroup_operator_parameters import OperatorParametersSection