import torch
from torch.autograd import Function
from joblib import Parallel, delayed
import numpy as np
from numba import njit

# ======= Funções auxiliares otimizadas =======

@njit
def central_diff(f_plus, f_minus, eps):
    return (f_plus - f_minus) / (2.0 * eps)

def parallel_eval(func_name, yi_np, d, opt_func, delta):
    if delta is not None:
        yi_np = yi_np + delta
    log = opt_func.OptimizationLog_Original()
    if func_name == 'Cost':
        return opt_func.Cost_Original(yi_np, d, log, 3)
    elif func_name == 'gT':
        return opt_func.gT_Original(yi_np, d, 0, log)
    elif func_name == 'g_TempLog':
        return -opt_func.g_TempLog_Original(yi_np, d)
    elif func_name == 'jac_gT':
        return opt_func.jac_gT_Original(yi_np, d, 0, log)
    elif func_name == 'jac_TempLog':
        return opt_func.jac_TempLog_Original(yi_np, d)
    else:
        raise ValueError(f"Unknown function {func_name}")

# ======= Classes Autograd Otimizadas =======

class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        costs = [
            opt_func.Cost_Original(yi, d, opt_func.OptimizationLog_Original(), 3)
            for yi in y.detach().cpu().numpy()
        ]
        return y.new_tensor(costs)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt_func
        device, dtype = y.device, y.dtype
        batch, n = y.shape

        eps = torch.stack([
            torch.tensor(opt_func.eps_definition_F3_Original(yi.cpu().numpy(), d), device=device, dtype=dtype)
            for yi in y
        ], dim=0)

        jobs = []
        for i in range(batch):
            yi_np = y[i].detach().cpu().numpy()
            for k in range(n):
                delta = np.zeros_like(yi_np); delta[k] = eps[i, k].item()
                jobs.append(('Cost', yi_np, d, opt_func, delta))
                jobs.append(('Cost', yi_np, d, opt_func, -delta))

        results = Parallel(n_jobs=-1)(
            delayed(parallel_eval)(*args) for args in jobs
        )

        grads = []
        idx = 0
        for i in range(batch):
            grad_i = torch.zeros(n, device=device, dtype=dtype)
            for k in range(n):
                f_plus = results[idx]
                f_minus = results[idx+1]
                diff = central_diff(np.array(f_plus), np.array(f_minus), np.array(eps[i, k].cpu()))
                grad_i[k] = diff
                idx += 2
            grads.append(grad_i)

        grad_y = torch.stack(grads, dim=0)
        return grad_y * grad_output.unsqueeze(1), None, None

class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func

        outputs = [
            opt_func.gT_Original(yi, d, 0, opt_func.OptimizationLog_Original())
            for yi in y.detach().cpu().numpy()
        ]
        return torch.stack([
            torch.tensor(o, device=y.device, dtype=y.dtype)
            for o in outputs
        ], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt_func
        device, dtype = y.device, y.dtype
        batch, n = y.shape

        eps = torch.full_like(y, 1e-6)

        jobs = []
        for i in range(batch):
            yi_np = y[i].detach().cpu().numpy()
            for k in range(n):
                delta = np.zeros_like(yi_np); delta[k] = eps[i, k].item()
                jobs.append(('gT', yi_np, d, opt_func, delta))
                jobs.append(('gT', yi_np, d, opt_func, -delta))

        results = Parallel(n_jobs=-1)(
            delayed(parallel_eval)(*args) for args in jobs
        )

        grads = []
        idx = 0
        for i in range(batch):
            go_i = grad_output[i]
            grad_i = torch.zeros(n, device=device, dtype=dtype)
            for k in range(n):
                f_plus = np.array(results[idx])
                f_minus = np.array(results[idx+1])
                diff = central_diff(f_plus, f_minus, np.array(eps[i, k].cpu()))
                grad_i[k] = (go_i * torch.tensor(diff, device=device)).sum()
                idx += 2
            grads.append(grad_i)

        grad_y = torch.stack(grads, dim=0)
        return grad_y, None, None

class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt = opt_func

        outputs = [
            -opt_func.g_TempLog_Original(yi, d)
            for yi in y.detach().cpu().numpy()
        ]
        return torch.stack([
            torch.tensor(o, device=y.device, dtype=y.dtype)
            for o in outputs
        ], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt
        device, dtype = y.device, y.dtype
        batch, n = y.shape

        eps = torch.full_like(y, 1e-6)

        jobs = []
        for i in range(batch):
            yi_np = y[i].detach().cpu().numpy()
            for k in range(n):
                delta = np.zeros_like(yi_np); delta[k] = eps[i, k].item()
                jobs.append(('g_TempLog', yi_np, d, opt_func, delta))
                jobs.append(('g_TempLog', yi_np, d, opt_func, -delta))

        results = Parallel(n_jobs=-1)(
            delayed(parallel_eval)(*args) for args in jobs
        )

        grads = []
        idx = 0
        for i in range(batch):
            go_i = grad_output[i]
            grad_i = torch.zeros(n, device=device, dtype=dtype)
            for k in range(n):
                f_plus = np.array(results[idx])
                f_minus = np.array(results[idx+1])
                diff = central_diff(f_plus, f_minus, np.array(eps[i, k].cpu()))
                grad_i[k] = (go_i * torch.tensor(diff, device=device)).sum()
                idx += 2
            grads.append(grad_i)

        grad_y = torch.stack(grads, dim=0)
        return grad_y, None, None

class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt = opt_func

        outputs = [
            opt_func.jac_gT_Original(yi, d, 0, opt_func.OptimizationLog_Original())
            for yi in y.detach().cpu().numpy()
        ]
        return torch.stack([
            torch.tensor(o, device=y.device, dtype=y.dtype)
            for o in outputs
        ], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt
        device, dtype = y.device, y.dtype
        batch, n = y.shape

        eps = torch.full_like(y, 1e-6)

        jobs = []
        for i in range(batch):
            yi_np = y[i].detach().cpu().numpy()
            for k in range(n):
                delta = np.zeros_like(yi_np); delta[k] = eps[i, k].item()
                jobs.append(('jac_gT', yi_np, d, opt_func, delta))
                jobs.append(('jac_gT', yi_np, d, opt_func, -delta))

        results = Parallel(n_jobs=-1)(
            delayed(parallel_eval)(*args) for args in jobs
        )

        grads = []
        idx = 0
        for i in range(batch):
            go_i = grad_output[i]
            grad_i = torch.zeros(n, device=device, dtype=dtype)
            for k in range(n):
                Jp = np.array(results[idx])
                Jm = np.array(results[idx+1])
                diff = central_diff(Jp, Jm, np.array(eps[i, k].cpu()))
                grad_i[k] = (go_i * torch.tensor(diff, device=device)).sum()
                idx += 2
            grads.append(grad_i)

        grad_y = torch.stack(grads, dim=0)
        return grad_y, None, None

class JacTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt = opt_func

        outputs = [
            opt_func.jac_TempLog_Original(yi, d)
            for yi in y.detach().cpu().numpy()
        ]
        return torch.stack([
            torch.tensor(o, device=y.device, dtype=y.dtype)
            for o in outputs
        ], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return torch.zeros_like(y), None, None
