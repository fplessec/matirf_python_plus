from typing import Dict, Any
import time
import torch

from algorithms.abstract_algo import Algorithm
from core.operations import apply_matirf_operator, get_variables_from_dict, estimate_delta_anisotropy_from_params
from settings import device, dtype
from algorithms.utils.loss_computer import LossComputer
from algorithms.utils.proximal_operators import ProximalOperators


class PpxaAlgo(Algorithm):

    def run(self, g, H, params: Dict[str, Any]):
        self.fixe_randomness()

        (max_iter, lambda_relax, K, EPS, gamma, reg, lambda_reg, delta, rho) = get_variables_from_dict(
            params,['max_iter', 'lambda_relax', 'K', 'EPS', 'gamma', 'reg', 'lambda_reg', 'delta', 'rho']
        )

        delta_estimated = estimate_delta_anisotropy_from_params(self.measurement_params, self.oper_params)
        self._print(f"delta anisotropy estimation = {delta_estimated}\n")

        gamma_prox_estimated = 1 / torch.linalg.norm(H)**2 / (1 - lambda_reg)
        self._print(f"gamma prox estimation = {gamma_prox_estimated}\n")

        Ht = H.transpose(0, 1)
        HtH = Ht @ H
        Htg = apply_matirf_operator(Ht, g)
        identity = torch.eye(H.shape[1], dtype=dtype, device=device)
        P = torch.inverse(identity + gamma * (1 - lambda_reg) * HtH)
        #P = torch.inverse(identity + gamma * HtH)  # NEW

        # f initial: f0 = ( HtH + λ_rr Id )^(-1) Htg   (ridge regression)
        lambda_rr = 10000.  #  lambda_rr >> 1 to stabilize inversion
        f = apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Htg)

        # PPXA inital terms
        p = [torch.zeros_like(f) for _ in range(3)]
        u = [f.clone() for _ in range(3)]

        # weights for each 3 ppxa terms (uniform here??)
        weights = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        # weights = torch.tensor([
        #     1 - lambda_reg,
        #     lambda_reg,
        #     1.0
        # ]) #NEW   # vraie question: LES POIDS OUI OU NON ??
        weights /= weights.sum()

        loss_computer = LossComputer(f, g, H, reg, lambda_reg, rho, delta=delta)
        prox_ops = ProximalOperators(delta=delta)

        t0 = time.time()
        loss = loss_computer(f)
        self._print(f"iter 0 \nloss={loss:.3e} | lambda_relax={lambda_relax:.2e}")
        prev_loss = loss

        for iter in range(max_iter):
            if self.is_stop_requested():
                print("self.is_stop_requested() =", self.is_stop_requested())
                return None

            p[0] = apply_matirf_operator(P, u[0] + gamma * (1 - lambda_reg) * Htg)  # data fidelity prox
            #p[0] = apply_matirf_operator(P, u[0] + gamma * Htg)  # NEW
            #p[1] = self.apply_prox_regularization(u[1], prox_ops, reg, lambda_reg, rho=rho)  # regularization prox
            p[1] = self.apply_prox_regularization(u[1], prox_ops, reg, gamma*lambda_reg, rho=rho)
            p[2] = prox_ops.prox_positivity(u[2])  # positivity constraint prox


            # updates the ppxa terms and f
            f, u = self.ppxa_inner_step(p, u, f, weights, lambda_relax)

            if iter % K == K - 1:
                loss = loss_computer(f)
                self._print(
                    f"iter {iter + 1:4d} \nloss={loss.item():.3e} | "
                    f"lambda_relax={lambda_relax:.2e} | "
                    f"dloss={loss - prev_loss:+.3e}"
                )
                # prev_loss is the loss K iterations before
                if loss - prev_loss > 0:
                    lambda_relax = lambda_relax / 2  # divides lambda_relax by 2
                elif loss - prev_loss > - EPS:  # the stopping criterion is met
                    self._print("\nThe stopping criterion EPS has been met.")
                    self._print(f"Execution en {time.time() - t0} s.")
                    return f.clamp(min=0.)
                prev_loss = loss

        self._print("\nThe maximum iterations number has been reached.")
        self._print(f"Execution en {time.time() - t0} s.")
        return f.clamp(min=0.)

    @staticmethod
    def ppxa_inner_step(p, u, f, weights, lambda_relax=1.9):
        """
        Inner step of the ppxa algorithm using different weights
        p_{k} = \sum w_i p_{i,k}
        u_{i,k+1} = u_{i,k} + \lambda ( 2 p_{k} - x_k - p_{i,k} )
        x_{k+1} = x_k + \lambda (p_k - x_k)
        """
        # compute the weighted sum of the proximal operators:
        pl = torch.zeros_like(f)
        for j in range(len(p)):
            pl += weights[j] * p[j]
        # updates the ppxa terms
        for j in range(len(u)):
            u[j] = u[j] + lambda_relax * (2.0 * pl - f - p[j])
        # update the solution:
        f += lambda_relax * (pl - f)
        return f, u

    @staticmethod
    def apply_prox_regularization(f, prox_ops, reg, lambda_reg, rho=0.6):
        if reg == "no regularization":
            return f
        elif reg == "L1":
            return prox_ops.prox_l1(f, lambda_reg)
        elif reg == "Tikhonov":
            return prox_ops.prox_tikhonov(f, lambda_reg)
        elif reg == "Tikhonov Boulanger":
            return prox_ops.prox_tikhonov_boulanger(f, lambda_reg)
        elif reg == "TV":
            return prox_ops.prox_tv(f, lambda_reg)
        elif reg == "TV normalized":
            return prox_ops.prox_tv_normalized(f, lambda_reg)
        elif reg == "Hessian Frobenius":
            return prox_ops.prox_hessian_frobenius(f, lambda_reg)
        elif reg == "SHV":
            return prox_ops.prox_shv(f, lambda_reg, rho=rho)
        else:
            raise ValueError(f"Unknown regularization method '{reg}'.")









#
#
# from typing import Dict, Any
# import time
# import torch
#
# from algorithms.abstract_algo import Algorithm
# from operations import apply_matirf_operator, get_variables_from_dict, estimate_delta_anisotropy_from_params
# from settings import device, dtype
# from algorithms.utils.loss_computer import LossComputer
# from algorithms.utils.proximal_operators import ProximalOperators
#
# # VERSION DEUX, ON A LE PPXA CLASSIQUE MAIS 1-lbd_reg et lbd_reg SONT DANS LES WEIGHTS
# class PpxaAlgo(Algorithm):
#
#     def run(self, g, H, params: Dict[str, Any]):
#         (max_iter, lambda_relax, K, EPS, gamma, reg, lambda_reg, delta, rho) = get_variables_from_dict(
#             params,['max_iter', 'lambda_relax', 'K', 'EPS', 'gamma', 'reg', 'lambda_reg', 'delta', 'rho']
#         )
#
#         delta_estimated = estimate_delta_anisotropy_from_params(self.measurement_params, self.oper_params)
#         self._print(f"delta anisotropy estimation = {delta_estimated}\n")
#
#         gamma_prox_estimated = 1 / torch.linalg.norm(H)**2 / (1 - lambda_reg)
#         self._print(f"gamma prox estimation = {gamma_prox_estimated}\n")
#
#         Ht = H.transpose(0, 1)
#         HtH = Ht @ H
#         Htg = apply_matirf_operator(Ht, g)
#         identity = torch.eye(H.shape[1], dtype=dtype, device=device)
#         #P = torch.inverse(identity + gamma * (1 - lambda_reg) * HtH)
#         P = torch.inverse(identity + gamma * HtH)  # NEW
#
#         # f initial: f0 = ( HtH + λ_rr Id )^(-1) Htg   (ridge regression)
#         lambda_rr = 10000.  #  lambda_rr >> 1 to stabilize inversion
#         f = apply_matirf_operator(torch.inverse(HtH + lambda_rr * identity), Htg)
#
#         # PPXA inital terms
#         p = [torch.zeros_like(f) for _ in range(3)]
#         u = [f.clone() for _ in range(3)]
#
#         # weights for each 3 ppxa terms (uniform here??)
#         #weights = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
#         weights = torch.tensor([
#             1 - lambda_reg,
#             lambda_reg,
#             1.0
#         ]) #NEW   # vraie question: LES POIDS OUI OU NON ??
#         weights /= weights.sum()
#
#         loss_computer = LossComputer(f, g, H, reg, lambda_reg, rho, delta=delta)
#         prox_ops = ProximalOperators(delta=delta)
#
#         t0 = time.time()
#         loss = loss_computer(f)
#         self._print(f"iter 0 \nloss={loss:.3e} | lambda_relax={lambda_relax:.2e}")
#         prev_loss = loss
#
#         for iter in range(max_iter):
#             if self.is_stop_requested():
#                 return None
#
#             #p[0] = apply_matirf_operator(P, u[0] + gamma * (1 - lambda_reg) * Htg)  # data fidelity prox
#             p[0] = apply_matirf_operator(P, u[0] + gamma * Htg)  # NEW
#             p[1] = self.apply_prox_regularization(u[1], prox_ops, reg, lambda_reg, rho=rho)  # regularization prox
#             #p[1] = self.apply_prox_regularization(u[1], prox_ops, reg, gamma*lambda_reg, rho=rho)
#             p[2] = prox_ops.prox_positivity(u[2])  # positivity constraint prox
#
#
#             # updates the ppxa terms and f
#             f, u = self.ppxa_inner_step(p, u, f, weights, lambda_relax)
#
#             if iter % K == K - 1:
#                 loss = loss_computer(f)
#                 self._print(
#                     f"iter {iter + 1:4d} \nloss={loss.item():.3e} | "
#                     f"lambda_relax={lambda_relax:.2e} | "
#                     f"dloss={loss - prev_loss:+.3e}"
#                 )
#                 # prev_loss is the loss K iterations before
#                 if loss - prev_loss > 0:
#                     lambda_relax = lambda_relax / 2  # divides lambda_relax by 2
#                 elif loss - prev_loss > - EPS:  # the stopping criterion is met
#                     self._print("\nThe stopping criterion EPS has been met.")
#                     self._print(f"Execution en {time.time() - t0} s.")
#                     return f.clamp(min=0.)
#                 prev_loss = loss
#
#         self._print("\nThe maximum iterations number has been reached.")
#         self._print(f"Execution en {time.time() - t0} s.")
#         return f.clamp(min=0.)
#
#     @staticmethod
#     def ppxa_inner_step(p, u, f, weights, lambda_relax=1.9):
#         """
#         Inner step of the ppxa algorithm using different weights
#         p_{k} = \sum w_i p_{i,k}
#         u_{i,k+1} = u_{i,k} + \lambda ( 2 p_{k} - x_k - p_{i,k} )
#         x_{k+1} = x_k + \lambda (p_k - x_k)
#         """
#         # compute the weighted sum of the proximal operators:
#         pl = torch.zeros_like(f)
#         for j in range(len(p)):
#             pl += weights[j] * p[j]
#         # updates the ppxa terms
#         for j in range(len(u)):
#             u[j] = u[j] + lambda_relax * (2.0 * pl - f - p[j])
#         # update the solution:
#         f += lambda_relax * (pl - f)
#         return f, u
#
#     @staticmethod
#     def apply_prox_regularization(f, prox_ops, reg, lambda_reg, rho=0.6):
#         if reg == "no regularization":
#             return f
#         elif reg == "L1":
#             return prox_ops.prox_l1(f, lambda_reg)
#         elif reg == "Tikhonov":
#             return prox_ops.prox_tikhonov(f, lambda_reg)
#         elif reg == "Tikhonov Boulanger":
#             return prox_ops.prox_tikhonov_boulanger(f, lambda_reg)
#         elif reg == "TV":
#             return prox_ops.prox_tv(f, lambda_reg)
#         elif reg == "TV normalized":
#             return prox_ops.prox_tv_normalized(f, lambda_reg)
#         elif reg == "Hessian Frobenius":
#             return prox_ops.prox_hessian_frobenius(f, lambda_reg)
#         elif reg == "SHV":
#             return prox_ops.prox_shv(f, lambda_reg, rho=rho)
#         else:
#             raise ValueError(f"Unknown regularization method '{reg}'.")
#
