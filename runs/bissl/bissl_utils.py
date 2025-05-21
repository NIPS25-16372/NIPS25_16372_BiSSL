from typing import Callable
import torch

from source.bissl.ij_solvers import IJGradCalc, cg_solver

from runs.bissl.config import ArgsBiSSLDefaults


def get_hinv_solver(
    args: ArgsBiSSLDefaults,
    lower_backbone: torch.nn.Module,
    lower_criterion: torch.nn.Module | Callable,
) -> IJGradCalc:
    match args.hinv_solver:
        case "cg":
            ij_grad_calc = IJGradCalc(
                solver=cg_solver,
                parameters=tuple(lower_backbone.parameters()),
                criterion=lower_criterion,
                lam_dampening=args.cg_lam_dampening,
                solver_kwargs=dict(
                    iter_num=args.cg_iter_num,
                    verbose=bool(args.cg_verbose),
                ),
            )
        case _:
            raise KeyError

    return ij_grad_calc
