"""
problems — one package per inverse problem.

Each declares its physics and nothing else. There is no algorithm code here and no GUI
code: solvers come from `solvers/`, the interface is generated from the problem's own
declaration.

A problem package contains:

    operator.py   a ForwardOperator: apply / adjoint, plus any closed form it can offer
    problem.py    the InverseProblem declaration — features, loaders, validation
    physics.py    optional, when the physics is more than a few lines and deserves to be
                  readable and testable on its own (MA-TIRF's optics do; a PSF does not)
"""
