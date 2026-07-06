"""
This package contains four files and one subpackage :
    > ‘qgroup_input_file.py’ with the InputFilesSection class, a qgroup for the 'input-paths' parameter set
    > 'qwidget_tif_fil_selector.py' with the TifFileSelector, a qwidget to manage the selection of the tif file, which can
      be either a MA-TIRF measurement stack or a 3D image corresponding to a ground truth, depending on the ‘mode’
      parameter of the ‘input-paths’ parameter set
    > 'qwidget_json_fil_selector.py' with the JsonFileSelector, a qwidget to manage the selection of the json file,
      which is the measurement parameters file in order to compute all the physical inside the MA-TIRF operator
    > 'tif_file_preprocess_viewer.py': with the TifFilePreprocessViewer class, a qwidget allowing the user to see the
      preprocessing of its data (the tif file data, but the preprocessing also uses the parameters of the measurement
      from the json file)
    > the subpackage 'measurement_parameters_editor, which contains:
        > 'qwidget_measurement_parameters_editor.py' with the MeasurementParametersEditor, a qwidget allowing the user
        to write or edit the measurement parameters file
        > 'measurement_parameters_ui_dictionary.py' with MEASUREMENT_PARAMETERS_UI that is written to build
        SimpleParameterWidget qwidgets (single_parameter_widget.py)

The 'input-paths' parameter set contains the following parameters:
    > mode: as explained bellow
    > tif: the path of the tif file
    > json: the path of the json file

The specificity of the ‘input-paths’ parameter set is the possibility of using two modes, with its ‘mode’ parameter:
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


from .qgroup_input_files import InputFilesSection