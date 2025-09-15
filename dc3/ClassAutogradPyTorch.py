import torch
from torch.autograd import Function
import numpy as np

# Utilitário: central difference (opcional)
def _finite_diff_forward(fn, x, eps_vec, out_dim=None):
    """
    fn: callable(x_np) -> R^m (1D array)
    x: 1D np.array
    eps_vec: (n,) array de passos
    Retorna (f0, J_fd) onde J_fd shape (n, m)
    """
    f0 = fn(x)
    f0 = np.asarray(f0, dtype=np.float64)
    n = x.shape[0]
    m = f0.shape[0] if f0.ndim > 0 else 1
    if out_dim is not None and m != out_dim:
        # opcional validar
        pass
    J = np.empty((n, m), dtype=np.float64)
    # Perturba todas as direções (forward)
    eye = np.eye(n, dtype=np.float64)
    Xp = x[None, :] + eye * eps_vec[:, None]  # (n, n)
    f_plus = [fn(Xp[i]) for i in range(n)]
    f_plus = np.asarray(f_plus, dtype=np.float64)  # (n, m)
    diff = (f_plus - f0) / eps_vec[:, None]
    J[:] = diff
    return f0, J

class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        y_np = y.detach().cpu().numpy()
        costs = np.array([float(opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3))
                          for yi in y_np], dtype=np.float64)
        return torch.from_numpy(costs).to(device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        y_np = y.detach().cpu().numpy()
        B, n = y_np.shape
        grads = np.zeros((B, n), dtype=np.float64)
        for i in range(B):
            yi = y_np[i]
            eps = opt_func.eps_definition_F3(yi, d)
            # aceitar eps escalar ou vetor
            if np.isscalar(eps):
                eps_vec = np.full(n, eps, dtype=np.float64)
            else:
                eps_vec = np.asarray(eps, dtype=np.float64)
                if eps_vec.shape[0] != n:
                    eps_vec = np.full(n, float(eps), dtype=np.float64)
            f0 = float(opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3))
            f_plus = []
            for k in range(n):
                yk = yi.copy()
                yk[k] += eps_vec[k]
                f_plus.append(float(opt_func.Cost(yk, d, opt_func.OptimizationLog(), 3)))
            f_plus = np.asarray(f_plus, dtype=np.float64)
            grads[i] = (f_plus - f0) / eps_vec
        grad_tensor = torch.from_numpy(grads).to(device=y.device, dtype=y.dtype)
        return grad_tensor * grad_output.unsqueeze(1), None, None

class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        y_np = y.detach().cpu().numpy()
        outputs = [opt_func.gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np]
        outs = np.asarray(outputs, dtype=np.float64)
        ctx.out_dim = outs.shape[1] if outs.ndim == 2 else 1
        return torch.from_numpy(outs).to(device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        y_np = y.detach().cpu().numpy()
        go_np = grad_output.detach().cpu().numpy()
        B, n = y_np.shape
        grads = np.zeros((B, n), dtype=np.float64)
        eps = 1e-6
        eye = np.eye(n, dtype=np.float64) * eps
        for i in range(B):
            yi = y_np[i]
            f0 = opt_func.gT(yi, d, 0, opt_func.OptimizationLog())
            f0 = np.asarray(f0, dtype=np.float64)
            Yp = yi[None, :] + eye  # (n, n)
            f_plus = [opt_func.gT(Yp[k], d, 0, opt_func.OptimizationLog()) for k in range(n)]
            f_plus = np.asarray(f_plus, dtype=np.float64)          # (n, m)
            J = (f_plus - f0) / eps                                # (n, m)
            grads[i] = J @ go_np[i]                                 # (n,)
        return torch.from_numpy(grads).to(device=y.device, dtype=y.dtype), None, None

class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        y_np = y.detach().cpu().numpy()
        vals = np.array([-opt_func.g_TempLog(yi, d) for yi in y_np], dtype=np.float64)  # (B, m)
        return torch.from_numpy(vals).to(device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        y_np = y.detach().cpu().numpy()
        go_np = grad_output.detach().cpu().numpy()
        B, n = y_np.shape
        # Infer m
        f0_all = np.array([-opt_func.g_TempLog(yi, d) for yi in y_np], dtype=np.float64)  # (B, m)
        B2, m = f0_all.shape
        eps = 1e-6
        grads = np.zeros((B, n), dtype=np.float64)
        for k in range(n):
            y_plus = y_np.copy()
            y_plus[:, k] += eps
            f_plus = np.array([-opt_func.g_TempLog(yi, d) for yi in y_plus], dtype=np.float64)  # (B,m)
            df = (f_plus - f0_all) / eps
            # chain rule: sum_j go_ij * df_ij
            grads[:, k] = np.sum(go_np * df, axis=1)
        return torch.from_numpy(grads).to(device=y.device, dtype=y.dtype), None, None

class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np]  # (m,n)
        arr = np.asarray(outs, dtype=np.float64)  # (B,m,n)
        return torch.from_numpy(arr).to(device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        y_np = y.detach().cpu().numpy()
        go_np = grad_output.detach().cpu().numpy()  # (B,m,n)
        B, n = y_np.shape
        grads = np.zeros((B, n), dtype=np.float64)
        eps = 1e-6
        for i in range(B):
            yi = y_np[i]
            J0 = opt_func.jac_gT(yi, d, 0, opt_func.OptimizationLog())  # (m,n)
            for k in range(n):
                yk = yi.copy()
                yk[k] += eps
                Jp = opt_func.jac_gT(yk, d, 0, opt_func.OptimizationLog())
                dJ = (Jp - J0) / eps
                grads[i, k] = np.sum(go_np[i] * dJ)
        return torch.from_numpy(grads).to(device=y.device, dtype=y.dtype), None, None

class JacTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_TempLog(yi, d) for yi in y_np]   # (m,n)
        arr = np.asarray(outs, dtype=np.float64)
        return torch.from_numpy(arr).to(device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Retorna zero -> sem gradiente através da jacobiana
        (y,) = ctx.saved_tensors
        return torch.zeros_like(y), None, None