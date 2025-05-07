import torch
from torch.autograd import Function
import numpy as np

# ========= COST ==========

class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func

        y_np = y.detach().cpu().numpy()
        costs = np.empty(y_np.shape[0])

        for i, yi in enumerate(y_np):
            log = opt_func.OptimizationLog()
            costs[i] = opt_func.Cost(yi, d, log, 3)

        return torch.tensor(costs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        device, dtype = y.device, y.dtype
        y_np = y.detach().cpu().numpy()

        batch, n = y_np.shape
        grads = np.empty((batch, n), dtype=np.float32)

        for i in range(batch):
            yi = y_np[i]
            eps = opt_func.eps_definition_F3(yi, d)
            f0 = opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3)

            for k in range(n):
                y_plus = yi.copy()
                y_plus[k] += eps[k]
                f_plus = opt_func.Cost(y_plus, d, opt_func.OptimizationLog(), 3)
                grads[i, k] = (f_plus - f0) / eps[k]

        return torch.tensor(grads, device=device, dtype=dtype) * grad_output.unsqueeze(1), None, None

# ========= GT ==========



class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt_func = d, opt_func

        y_np = y.detach().cpu().numpy()
        outputs = np.array([opt_func.gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np])
        return torch.tensor(outputs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        print('ACESSOU GTAUTGRAD')
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func

        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, D = y_np.shape
        grads = np.zeros_like(y_np, dtype=np.float64)
        eps = 1e-6

        for i in range(B):
            yi = y_np[i]
            go_i = grad_out_np[i]
            f0 = opt_func.gT(yi, d, 0, opt_func.OptimizationLog())  # shape (D,)

            # Perturbações: gera D vetores com perturbação em uma dimensão
            y_plus = yi[None, :] + np.eye(D) * eps  # shape (D, D)
            
            # Avaliações para todas as perturbações
            f_plus = np.array([opt_func.gT(y_pert, d, 0, opt_func.OptimizationLog()) for y_pert in y_plus])

            # Derivadas numéricas: shape (D, D)
            jacobian = (f_plus - f0) / eps

            # Produto: (D,) = (D, D) @ (D,)
            grads[i] = jacobian @ go_i

        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None



class GTAutograd_Original(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt_func = d, opt_func
        y_np = y.detach().cpu().numpy()
        outputs = [opt_func.gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np]
        return torch.tensor(outputs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, D = y_np.shape
        grads = np.empty_like(y_np)

        for i in range(B):
            yi = y_np[i]
            go_i = grad_out_np[i]
            eps = np.full(D, 1e-6, dtype=np.float64)
            f0 = opt_func.gT(yi, d, 0, opt_func.OptimizationLog())

            for k in range(D):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                f_plus = opt_func.gT(y_plus, d, 0, opt_func.OptimizationLog())
                grads[i, k] = np.dot(go_i, (f_plus - f0) / eps[k])

        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None

# ========= GTempLog ==========

class GTempLogAutograd_Original(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        y_np = y.detach().cpu().numpy()
        results = [-opt_func.g_TempLog(yi, d) for yi in y_np]
        return torch.tensor(results, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt
        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, D = y_np.shape
        grads = np.empty_like(y_np)

        for i in range(B):
            yi = y_np[i]
            go_i = grad_out_np[i]
            eps = np.full(D, 1e-6)
            f0 = -opt_func.g_TempLog(yi, d)

            for k in range(D):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                f_plus = -opt_func.g_TempLog(y_plus, d)
                grads[i, k] = np.dot(go_i, (f_plus - f0) / eps[k])

        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None
    
    
class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func

        y_np = y.detach().cpu().numpy()
        results = np.array([-opt_func.g_TempLog(yi, d) for yi in y_np])  # (B, D)
        return torch.tensor(results, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt

        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()  # (B, D)

        B, D = y_np.shape
        eps = 1e-6
        grads = np.zeros_like(y_np)

        f0 = np.array([-opt_func.g_TempLog(yi, d) for yi in y_np])  # (B, D)

        for k in range(D):
            y_plus = y_np.copy()
            y_plus[:, k] += eps

            f_plus = np.array([-opt_func.g_TempLog(yi, d) for yi in y_plus])  # (B, D)
            df = (f_plus - f0) / eps  # (B, D)

            grads[:, k] = np.sum(grad_out_np * df, axis=1)  # Reduz para (B,)

        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None



# ========= JacGT ==========



class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np]
        return torch.tensor(outs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt
        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, n = y_np.shape
        grads = np.empty_like(y_np)

        for i in range(B):
            yi = y_np[i]
            go_i = grad_out_np[i]
            eps = np.full(n, 1e-6)
            J0 = opt_func.jac_gT(yi, d, 0, opt_func.OptimizationLog())

            for k in range(n):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                Jp = opt_func.jac_gT(y_plus, d, 0, opt_func.OptimizationLog())
                grads[i, k] = np.sum(go_i * (Jp - J0) / eps[k])

        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None

# ========= JacTempLog ==========

class JacTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_TempLog(yi, d) for yi in y_np]
        return torch.tensor(outs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return torch.zeros_like(y), None, None
