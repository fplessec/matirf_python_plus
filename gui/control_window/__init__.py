"""
This package contains three files and one subpackage:
    > ‘control_window.py’ with the ControlWindow class, which is the main window of this project's U.I.
    > ‘display_window_manager.py’ with the DisplayWindowManager class to manage instances of the DisplayWindow class (see
      the display_window package)
    > 'single_parameter_widget.py' with the SimpleParameterWidget class, which is a complex object inheriting from QWidget
    > the 'sections' subpackage which combines the subsubpackages corresponding to each subsection that  makes up
      ControlWindow; organizing all the necessary parameters for a MA-TIRF reconstruction into different sets of
      parameters

The control window allows the user to choose all the elements needed to perform a MA-TIRF reconstruction: measurement
files and measurement parameters, reconstruction and algorithm parameters, and other more specific options.

More precisely, a MA-TIRF reconstruction is dictated by a configuration file called config.toml. The control window
allows the user to continuously edit this toml file contained in the ‘cache’ folder, which we will call the cached
configuration.
This window allows the user to write to the ‘cache’ folder and also to read any other config.toml file by copying it to
the cache, and thus be able to restart a MA-TIRF reconstruction that has already been performed to verify or retest
with different parameters here and there, for example.

When the user has finished entering all the parameters, they simply press the run button to start the reconstruction,
which is managed by another window called the display window. They can also open a reconstruction that has already been
done using another dedicated button.
There can be as many different display windows open at the same time, and several reconstructions can be performed
simultaneously (but if everything is done on the same CPU/GPU, this will obviously slow down the calculation time; I
use the Python threading library to manage the algorithms independently).
The class DisplayWindowManager exists to organize the instantiation and the closures of display windows.

This window is composed of subsections using the QGroupBox object. Each qgroup/section is used to manage a specific set
of parameters. These sets of parameters are stored in the config.toml file under keys that I use to name each set of
parameters. The U.I. looks like this:
    > a qgroup for the 'input-paths' parameters
    > a qgroup for the 'oper-params' parameters
    > a qgroup for the 'add-noise' parameters
    > a qgroup for the 'algo-params' parameters and also the main parameter 'algorithm'
"""

from .control_window import ControlWindow, DisplayWindowManager