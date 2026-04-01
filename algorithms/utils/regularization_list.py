"""
To be able to compare the different optimization algorithms, we used the same regularizations in each algorithm.
Those different regularizations are variational based regularizations. They are used in the ADAM and PPXA algos.

If you want to implement another optimization algo, that used variational based regularizations, you should re-use
those so to be able to compare with the existing algos.

If you add a new regularization, you should make sure it is completely implemented for pre-existing algos, for example
you should add the proximal operator of your regularization in order to make PPXA be able to use it too.
Otherwise, you can just to different list of regularization, for each algos (REGULARIZATION_LIST_ADAM,
REGULARIZATION_LIST_PPXA, ...), and modify the LossComputer object to be able to swap between those different lists.


fiudhsjofsjdfoisdhfoidshfoidsjfodsiofjsodijfsldjflsdkjflsd BLABLA A COMPLETER
        - none          : R(f) = 0
        - l2            : R(f) = ||f||² = sum(f_ijk²)
        - l1            : R(f) = ||f||₁ = sum(|f_ijk|)
        - grad_l2       : R(f) = ||∇f||² = sum((∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)²)
        - grad_l1       : R(f) = ||∇f||₁ = sum(|∂f/∂x| + |∂f/∂y| + |∂f/∂z|)
        - hessian_frob  : R(f) = ||H(f)||_F² = sum(H_ij²) for Hessian matrix
        - shv           : R(f) = ρ * ||H(f)||_F² + (1-ρ) * ||f||₁
"""

REGULARIZATION_LIST = ["no regularization",
                       "L2 norm",
                       "L1 norm",
                       "L2 norm of the gradient",
                       "L1 norm of the gradient",
                       "frobenius norm of the hessian",
                       "sparse hessian variation"]

PPXA_REGULARIZATION_LIST = ["no regularization",
                            "L1",
                            "Tikhonov",
                            "Tikhonov Boulanger",
                            "TV",
                            "TV normalized",
                            "Hessian Frobenius",
                            "SHV"]