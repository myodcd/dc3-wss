import torch
from torch.autograd import Function
import OptimAuxFunctionsV2 as opt_func
import numpy as np

class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        # y: [batch, n_vars]
        ctx.d = d
        ctx.opt_func = opt_func
        ctx.save_for_backward(y)

        costs = []
        # computa custo para cada amostra, via API externa
        for yi in y.detach().cpu().numpy():
            log = opt_func.CostOptimizationLog()
            costs.append(opt_func.Cost(yi, d, log, 3))
        # retorna tensor já no device e dtype de y
        out = y.new_tensor(costs)          # shape [batch]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt_func

        epsilon = 1e-6
        grad_est = y.new_zeros(y.shape)    # [batch, n_vars]

        # diferenças finitas para cada elemento de y
        for b in range(y.shape[0]):
            yb = y[b]                       # [n_vars]
            base = yb.detach().cpu().numpy()
            for i in range(yb.shape[0]):
                y_plus  = base.copy()
                y_minus = base.copy()
                y_plus[i]  += epsilon
                y_minus[i] -= epsilon

                log1 = opt_func.CostOptimizationLog()
                f_plus  = opt_func.Cost(y_plus,  d, log1, 3)
                log2 = opt_func.CostOptimizationLog()
                f_minus = opt_func.Cost(y_minus, d, log2, 3)

                # ∂f/∂y_i ≈ (f_plus - f_minus)/2ε
                df_dyi = (f_plus - f_minus) / (2 * epsilon)
                # aplica cadeia: grad_est[b,i] = grad_output[b] * df_dyi
                grad_est[b, i] = grad_output[b] * df_dyi

        # devolve (grad_y, None, None)
        return grad_est, None, None


class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        y_np = y.detach().cpu().numpy()
        outs = []
        for i in range(y_np.shape[0]):
            log_tank = opt_func.TanksOptimizationLog()
            outs.append(opt_func.gT(y_np[i], d, 0, log_tank))
        out = torch.tensor(outs, device=y.device, dtype=y.dtype)
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt_func

        epsilon = 1e-6
        # prealoca tensor de gradientes [batch, n_vars]
        grad_est = y.new_zeros(y.shape)

        # para cada amostra no batch
        for b in range(y.shape[0]):
            base = y[b].detach().cpu().numpy()
            # para cada variável
            for j in range(y.shape[1]):
                y_plus  = base.copy()
                y_minus = base.copy()
                y_plus[j]  += epsilon
                y_minus[j] -= epsilon

                log1 = opt_func.TanksOptimizationLog()
                f_plus  = torch.as_tensor(opt_func.gT(y_plus,  d, 0, log1),
                                          dtype=y.dtype)
                log2 = opt_func.TanksOptimizationLog()
                f_minus = torch.as_tensor(opt_func.gT(y_minus, d, 0, log2),
                                          dtype=y.dtype)

                # ∂g/∂y_j ≈ (f_plus - f_minus) / (2ε)   → vetor [n_tanks]
                dg_dyj = (f_plus - f_minus) / (2 * epsilon)
                # cadeia: grad_output[b] · dg_dyj
                grad_est[b, j] = torch.dot(grad_output[b], dg_dyj)

        return grad_est, None, None


class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        batch, n = y.shape
        eps = 1e-6
        jac_pos = y.new_zeros(batch, n, n)
        y_np = y.detach().cpu().numpy()
        for b in range(batch):
            base = y_np[b]
            for j in range(n):
                plus, minus = base.copy(), base.copy()
                plus[j]  += eps
                minus[j] -= eps
                log1, log2 = opt_func.TanksOptimizationLog(), opt_func.TanksOptimizationLog()
                f_p = torch.as_tensor(opt_func.gT(plus,  d, 0, log1),
                                      dtype=y.dtype, device=y.device)
                f_m = torch.as_tensor(opt_func.gT(minus, d, 0, log2),
                                      dtype=y.dtype, device=y.device)
                jac_pos[b, :, j] = (f_p - f_m) / (2*eps)
        ctx.save_for_backward(jac_pos)
        return torch.cat([jac_pos, -jac_pos], dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        (jac_pos,) = ctx.saved_tensors
        n = jac_pos.shape[1]
        go_pos = grad_output[:, :n, :]
        go_neg = grad_output[:, n:, :]
        # soma go_pos⋅J_posᵀ e subtrai go_neg⋅J_posᵀ
        grad_input = (
            torch.bmm(go_pos.transpose(1,2), jac_pos).sum(dim=2)
          - torch.bmm(go_neg.transpose(1,2), jac_pos).sum(dim=2)
        )
        return grad_input, None, None
