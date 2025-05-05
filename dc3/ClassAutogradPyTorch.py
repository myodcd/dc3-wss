import torch
from torch.autograd import Function
import numpy as np

class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func

        y_np = y.detach().cpu().numpy()
        costs = []
        for yi in y_np:
            log = opt_func.OptimizationLog_Original()
            cost = opt_func.Cost_Original(yi, d, log, 3)
            costs.append(cost)

        return y.new_tensor(costs)

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        device, dtype = y.device, y.dtype

        y_np = y.detach().cpu().numpy()
        batch, n = y.shape
        grads = []

        for i in range(batch):
            yi = y_np[i]
            eps = opt_func.eps_definition_F3_Original(yi, d)
            f0 = opt_func.Cost_Original(yi, d, opt_func.OptimizationLog_Original(), 3)
            grad_i = np.zeros(n)

            for k in range(n):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                y_minus = yi.copy(); y_minus[k] -= eps[k]

                f_plus = opt_func.Cost_Original(y_plus, d, opt_func.OptimizationLog_Original(), 3)
                f_minus = opt_func.Cost_Original(y_minus, d, opt_func.OptimizationLog_Original(), 3)

                grad_i[k] = (f_plus - f_minus) / (2 * eps[k])

            grads.append(torch.tensor(grad_i, device=device, dtype=dtype))

        grad_y = torch.stack(grads, dim=0)
        return grad_y * grad_output.unsqueeze(1), None, None


class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        device, dtype = y.device, y.dtype
        y_np = y.detach().cpu().numpy()

        outs = [opt_func.gT_Original(yi, d, 0, opt_func.OptimizationLog_Original()) for yi in y_np]
        out = torch.stack([torch.as_tensor(o, device=device, dtype=dtype) for o in outs])
        ctx.save_for_backward(y)
        ctx.d, ctx.opt_func = d, opt_func
        return out

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func
        device, dtype = y.device, y.dtype

        y_np = y.detach().cpu().numpy()
        B, D = y.shape
        grads = []

        for i in range(B):
            yi = y_np[i]
            go_i = grad_output[i]
            eps = np.full(D, 1e-6)
            grad_i = np.zeros(D)

            for k in range(D):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                y_minus = yi.copy(); y_minus[k] -= eps[k]

                f_plus = opt_func.gT_Original(y_plus, d, 0, opt_func.OptimizationLog_Original())
                f_minus = opt_func.gT_Original(y_minus, d, 0, opt_func.OptimizationLog_Original())

                grad_i[k] = np.dot(go_i.cpu().numpy(), (f_plus - f_minus) / (2 * eps[k]))

            grads.append(torch.tensor(grad_i, device=device, dtype=dtype))

        return torch.stack(grads), None, None


class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        device, dtype = y.device, y.dtype
        y_np = y.detach().cpu().numpy()
        outs = [-opt_func.g_TempLog_Original(yi, d) for yi in y_np]
        out = torch.stack([torch.as_tensor(o, device=device, dtype=dtype) for o in outs])
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        return out

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt
        device, dtype = y.device, y.dtype

        y_np = y.detach().cpu().numpy()
        B, D = y.shape
        grads = []

        for i in range(B):
            yi = y_np[i]
            go_i = grad_output[i].cpu().numpy()
            eps = np.full(D, 1e-6)
            grad_i = np.zeros(D)

            for k in range(D):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                y_minus = yi.copy(); y_minus[k] -= eps[k]

                f_plus = -opt_func.g_TempLog_Original(y_plus, d)
                f_minus = -opt_func.g_TempLog_Original(y_minus, d)

                grad_i[k] = np.dot(go_i, (f_plus - f_minus) / (2 * eps[k]))

            grads.append(torch.tensor(grad_i, device=device, dtype=dtype))

        return torch.stack(grads), None, None


class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_gT_Original(yi, d, 0, opt_func.OptimizationLog_Original()) for yi in y_np]
        return torch.stack([torch.as_tensor(o, device=y.device, dtype=y.dtype) for o in outs])

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt
        device, dtype = y.device, y.dtype
        y_np = y.detach().cpu().numpy()

        B, n = y.shape
        grads = []

        for i in range(B):
            yi = y_np[i]
            go_i = grad_output[i].cpu().numpy()
            eps = np.full(n, 1e-6)
            grad_i = np.zeros(n)

            for k in range(n):
                y_plus = yi.copy(); y_plus[k] += eps[k]
                y_minus = yi.copy(); y_minus[k] -= eps[k]

                Jp = opt_func.jac_gT_Original(y_plus, d, 0, opt_func.OptimizationLog_Original())
                Jm = opt_func.jac_gT_Original(y_minus, d, 0, opt_func.OptimizationLog_Original())

                grad_i[k] = np.sum(go_i * (Jp - Jm) / (2 * eps[k]))

            grads.append(torch.tensor(grad_i, device=device, dtype=dtype))

        return torch.stack(grads), None, None


class JacTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        y_np = y.detach().cpu().numpy()
        out = [opt_func.jac_TempLog_Original(yi, d) for yi in y_np]
        return torch.stack([torch.as_tensor(o, device=y.device, dtype=y.dtype) for o in out])

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return torch.zeros_like(y), None, None
