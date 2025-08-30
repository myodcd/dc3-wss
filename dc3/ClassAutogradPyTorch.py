import torch
from torch.autograd import Function
import numpy as np
from joblib import Parallel, delayed
import time

# ========= COST ==========


class CostAutograd_ORIGINAL(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func

        y_np = y.detach().cpu().numpy()
        costs = np.array(
            [float(opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3)) for yi in y_np]
        )

        return torch.tensor(costs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func

        y_np = y.detach().cpu().numpy()
        batch, n = y_np.shape
        grads = np.zeros_like(y_np, dtype=np.float64)

        for i in range(batch):
            yi = y_np[i]
            eps = opt_func.eps_definition_F3(yi, d)

            f0 = float(opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3))

            # Cria perturbações
            perturb = np.eye(n) * eps
            y_pert = yi[None, :] + perturb

            f_plus = np.array(
                [
                    float(opt_func.Cost(y_pert[j], d, opt_func.OptimizationLog(), 3))
                    for j in range(n)
                ]
            )

            grads[i] = (f_plus - f0) / eps

        grad_tensor = torch.tensor(grads, device=y.device, dtype=y.dtype)
        return grad_tensor * grad_output.unsqueeze(1), None, None


class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func

        # Converte para NumPy apenas para chamar a rotina externa
        y_np = y.detach().cpu().numpy()
        costs = np.array(
            [float(opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3)) for yi in y_np]
        )

        return torch.tensor(costs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func

        y_np = y.detach().cpu().numpy()
        batch, n = y_np.shape
        grads = np.zeros_like(y_np, dtype=np.float64)

        for i in range(batch):
            yi = y_np[i]
            eps = opt_func.eps_definition_F3(yi, d)  # escalar ou vetor

            f0 = float(opt_func.Cost(yi, d, opt_func.OptimizationLog(), 3))

            # Perturbações unitárias
            perturb = np.eye(n) * eps
            y_pert = yi[None, :] + perturb

            f_plus = np.array(
                [
                    float(opt_func.Cost(y_pert[j], d, opt_func.OptimizationLog(), 3))
                    for j in range(n)
                ]
            )

            grads[i] = (f_plus - f0) / eps

        grad_tensor = torch.from_numpy(grads).to(device=y.device, dtype=y.dtype)
        return grad_tensor * grad_output.unsqueeze(1), None, None


# ========= GT ==========


class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        # time_start_gtautgrad_forward = time.time()
        ctx.save_for_backward(y)
        ctx.d, ctx.opt_func = d, opt_func

        y_np = y.detach().cpu().numpy()
        outputs = np.array(
            [opt_func.gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np]
        )
        # print('GTAUTGRAD_FORWARD', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start_gtautgrad_forward)))
        # print('%%%%%%%%%')
        return torch.tensor(outputs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # CALL_COUNTER['GTAutograd'] += 1
        # time_start_gtautgrad = time.time()
        # print(f'begin ACESSOU GTAUTGRAD ({CALL_COUNTER["GTAutograd"]} acesso(s))')
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt_func

        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, D = y_np.shape
        grads = np.zeros_like(y_np, dtype=np.float64)
        eps = 1e-6

        for i in range(B):
            yi = y_np[i]
            go_i = grad_out_np[i]
            f0 = opt_func.gT(yi, d, 0, opt_func.OptimizationLog())

            y_plus = yi[None, :] + np.eye(D) * eps
            f_plus = np.array(
                [
                    opt_func.gT(y_pert, d, 0, opt_func.OptimizationLog())
                    for y_pert in y_plus
                ]
            )
            jacobian = (f_plus - f0) / eps
            grads[i] = jacobian @ go_i

        # print('GTAUTGRAD backward', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start_gtautgrad)))
        # print('%%%%%%%%%')
        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None


# ========= GTempLog ==========


class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        # time_start_gtemp = time.time()
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func

        y_np = y.detach().cpu().numpy()
        results = np.array([-opt_func.g_TempLog(yi, d) for yi in y_np])
        # print('GTempLogAutograd_FORWARD', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start_gtemp)))
        # print('%%%%%%%%%')
        return torch.tensor(results, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):

        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt

        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, D = y_np.shape
        eps = 1e-6
        grads = np.zeros_like(y_np)

        f0 = np.array([-opt_func.g_TempLog(yi, d) for yi in y_np])

        for k in range(D):
            y_plus = y_np.copy()
            y_plus[:, k] += eps
            f_plus = np.array([-opt_func.g_TempLog(yi, d) for yi in y_plus])
            df = (f_plus - f0) / eps
            grads[:, k] = np.sum(grad_out_np * df, axis=1)
        return torch.tensor(grads, device=y.device, dtype=y.dtype), None, None


# ========= JacGT ==========


class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        # time_start_jacgt = time.time()
        ctx.save_for_backward(y)
        ctx.d, ctx.opt = d, opt_func
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_gT(yi, d, 0, opt_func.OptimizationLog()) for yi in y_np]
        return torch.tensor(outs, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d, opt_func = ctx.d, ctx.opt
        y_np = y.detach().cpu().numpy()
        grad_out_np = grad_output.detach().cpu().numpy()

        B, n = y_np.shape
        grads = np.empty_like(y_np)

        for i in range(B):

            time_loop_1 = time.time()

            yi = y_np[i]
            go_i = grad_out_np[i]
            eps = np.full(n, 1e-6)
            J0 = opt_func.jac_gT(yi, d, 0, opt_func.OptimizationLog())

            for k in range(n):
                time_loop_2 = time.time()
                y_plus = yi.copy()
                y_plus[k] += eps[k]
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
        (y,) = ctx.saved_tensors
        return torch.zeros_like(y), None, None
