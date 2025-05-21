from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Tuple,
    Optional,
    Union,
)
import torch
import torch.distributed


class IJGradCalc:
    """
    IJGradCalc is a class designed to compute the implicit jacobian (IJ) vector product
    using a specified solver. It calculates the Hessian vector product and applies
    dampening to the lambda parameter.

    Attributes:
        solver (Callable[..., Tuple[torch.Tensor, ...]]): A solver function that computes
            the IJ vector product.
        parameters (Tuple[torch.nn.Parameter, ...]): A tuple of model parameters.
        criterion (Callable[[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]):
            A loss function used to compute gradients.
        lam_dampening (float): A dampening factor applied to the lambda parameter.
        solver_kwargs_dict (Dict[str, Any]): Additional keyword arguments for the solver.

    Methods:
        __call__(lower_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 lam: Union[float, Tuple[float, ...]] = 1.0) -> Tuple[torch.Tensor, ...]:
            Computes the IJ vector product for the given input and vectors.
    """

    def __init__(
        self,
        solver: Callable[..., Tuple[torch.Tensor, ...]],
        parameters: Tuple[torch.nn.Parameter, ...],
        criterion: Callable[
            [torch.Tensor | Tuple[torch.Tensor, torch.Tensor]], torch.Tensor
        ],
        lam_dampening: float = 0.0,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.criterion = criterion
        self.parameters = parameters
        self.lam_dampening = lam_dampening
        self.solver = solver

        if solver_kwargs is not None:
            self.solver_kwargs_dict: Dict[Any, Any] = dict(solver_kwargs)
        else:
            self.solver_kwargs_dict = {}

    def _accumulate_vectors(
        self, vecs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Accumulates gradients across all processes in distributed.
        """
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            vecs = tuple(v.contiguous() for v in vecs)
            # Ensure all tensors in vecs are contiguous
            for v in vecs:
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            return tuple(
                v.div(torch.distributed.get_world_size()).detach() for v in vecs
            )
        return vecs

    def __call__(
        self,
        lower_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        vecs: Tuple[torch.Tensor, ...],
        lam: Union[float, Tuple[float, ...]] = 1.0,
    ) -> Tuple[torch.Tensor, ...]:

        assert all(vec.shape == par.shape for vec, par in zip(vecs, self.parameters))
        # turns lambda par
        if isinstance(lam, float):
            lam = tuple(lam for _ in range(len(vecs)))
        elif isinstance(lam, (tuple, list)):
            pass
        else:
            raise TypeError("lam must be a float or a tuple/list of floats")

        # Ensures to accumulate gradients across all processes in distributed training
        vecs = self._accumulate_vectors(vecs)

        # Function which calculates hessian vector product, and returns this product along
        # with the other term, such that the output is the inverse IJ vector product
        def hpid_v_p(vs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            # Clears the cache to avoid memory leaks
            torch.cuda.empty_cache()

            # Gets first order gradients
            grads: Iterator[torch.Tensor] = iter(
                torch.autograd.grad(
                    self.criterion(lower_input),
                    self.parameters,
                    create_graph=True,
                )
            )

            # Inner product of the gradients and given input vectors
            # prod is a tuple of scalar tensors
            prods: Tuple[torch.Tensor, ...] = tuple(
                grad.mul(v).sum() for grad, v in zip(grads, vs)
            )

            # Hessian vector product
            hvps: Tuple[torch.Tensor, ...] = torch.autograd.grad(prods, self.parameters)  # type: ignore

            # Accumulate gradients across all processes in distributed training.
            # Should be correct due to the liniarity of the hessian vector product...
            hvps = self._accumulate_vectors(hvps)

            return tuple(
                vmul + hvp.div_(lam_par + self.lam_dampening)
                for vmul, hvp, lam_par in zip(vs, hvps, lam)
            )

        return self.solver(hpid_v_p, vecs, **self.solver_kwargs_dict)


def cg_solver(
    mvp_fn: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    vecs: Tuple[torch.Tensor, ...],
    layerwise_solve: bool = True,
    iter_num: int = 10,
    verbose: bool = False,
) -> Tuple[torch.Tensor, ...]:

    xs = tuple(torch.zeros_like(vec) for vec in vecs)
    rs = tuple(vec for vec in vecs)
    ds = tuple(r.clone() for r in rs)

    if layerwise_solve:
        # Initialising norm(r), avoiding the necessity for calculating this value twice in a loop
        new_rdotrs = tuple(torch.sum(r**2) for r in rs)

        for _ in range(iter_num):
            Hds = mvp_fn(ds)

            rdotrs = tuple(r_copy for r_copy in new_rdotrs)

            alphas = tuple(
                rdotr / (torch.sum(d * Hd) + 1e-12)
                for rdotr, d, Hd in zip(rdotrs, ds, Hds)
            )

            xs = tuple(x + alpha * d for x, d, alpha in zip(xs, ds, alphas))
            rs = tuple(r - alpha * Hd for r, Hd, alpha in zip(rs, Hds, alphas))

            new_rdotrs = tuple(torch.sum(r**2) for r in rs)
            betas = tuple(
                new_rdotr / (rdotr + 1e-12)
                for new_rdotr, rdotr in zip(new_rdotrs, rdotrs)
            )

            ds = tuple(r + beta * d for r, d, beta in zip(rs, ds, betas))

            if verbose:
                print(new_rdotrs)
                print("")
    else:
        # Initialising norm(r), avoiding the neccesity for calculating this value twice in a loop
        new_rdotr = sum(torch.sum(r**2) for r in rs)

        for _ in range(iter_num):
            rdotr = new_rdotr

            Hds = mvp_fn(ds)

            alpha = sum(rdotr / (torch.sum(d * Hd) + 1e-12) for d, Hd in zip(ds, Hds))

            xs = tuple(x + alpha * d for x, d in zip(xs, ds))
            rs = tuple(r - alpha * Hd for r, Hd in zip(rs, Hds))

            new_rdotr = sum(torch.sum(r**2) for r in rs)
            beta = new_rdotr / (rdotr + 1e-12)

            ds = tuple(r + beta * d for r, d in zip(rs, ds))

            if verbose:
                print(new_rdotr)
                print("")

    return tuple(torch.nan_to_num(x) for x in xs)
