import torch
from torch.autograd import Function
import OptimAuxFunctionsV2 as opt_func
import numpy as np
import utils

class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        # y: [batch, n_vars], requires_grad=True
        # salvamos o y pra usar no backward
        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func

        # rodamos sua API externa — OK, aqui usamos NumPy, mas em seguida
        # no backward vamos reaproximar a derivada em PyTorch.
        costs = []
        for yi in y.detach().cpu().numpy():
            log = opt_func.OptimizationLog_Original()
            costs.append(opt_func.Cost_Original(yi, d, log, 3))
        # Saída: tensor *novo* mas que conectaremos no backward
        return y.new_tensor(costs)  # [batch]

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: ∂L/∂cost, shape [batch]
        Queremos ∂cost/∂y numericamente e depois aplicar regra da cadeia:
          ∂L/∂y = (∂L/∂cost)[:,None] * (∂cost/∂y)
        """
        y, = ctx.saved_tensors
        d = ctx.d
        opt_func = ctx.opt_func
        device, dtype = y.device, y.dtype

        # passo 1: construir eps para cada coord.
        # — você já tinha algo parecido em eps_definition_F3
        eps_list = []
        for yi in y:
            eps_np = opt_func.eps_definition_F3_Original(yi.cpu().numpy(), d)
            eps_list.append(torch.tensor(eps_np, device=device, dtype=dtype))
        eps = torch.stack(eps_list, dim=0)  # [batch, n_vars]

        batch, n = y.shape
        grads = []
        # passo 2: diferença central em PyTorch
        for i in range(batch):
            yi = y[i]
            eps_i = eps[i]
            # f0
            f0 = torch.tensor(
                opt_func.Cost_Original(yi.detach().cpu().numpy(), d, opt_func.OptimizationLog_Original(), 3),
                device=device, dtype=dtype
            )
            # grad por coord
            grad_i = torch.zeros_like(yi)
            for k in range(n):
                δ = torch.zeros_like(yi); δ[k] = eps_i[k]
                f_plus = torch.tensor(
                    opt_func.Cost_Original((yi + δ).detach().cpu().numpy(), d, opt_func.OptimizationLog_Original(), 3),
                    device=device, dtype=dtype
                )
                f_minus = torch.tensor(
                    opt_func.Cost_Original((yi - δ).detach().cpu().numpy(), d, opt_func.OptimizationLog_Original(), 3),
                    device=device, dtype=dtype
                )
                grad_i[k] = (f_plus - f_minus) / (2 * eps_i[k])
            grads.append(grad_i)
        # [batch, n_vars]
        grad_y = torch.stack(grads, dim=0)

        # aplicação da regra da cadeia
        # grad_output: [batch] → [batch,1]
        grad_y = grad_y * grad_output.unsqueeze(1)

        # Não há gradiente para d nem opt_func
        return grad_y, None, None


class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        # y: [B, D]
        device, dtype = y.device, y.dtype
        y_np = y.detach().cpu().numpy()

        outs = []
        for yi in y_np:
            log = opt_func.OptimizationLog_Original()
            outs.append(opt_func.gT_Original(yi, d, 0, log))  # array [m]

        # monta tensor [B, m] no mesmo device/dtype de y
        out = torch.stack([
            torch.as_tensor(o, device=device, dtype=dtype)
            for o in outs
        ], dim=0)

        ctx.save_for_backward(y)
        ctx.d = d
        ctx.opt_func = opt_func
        return out

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        d = ctx.d
        optf = ctx.opt_func
        device, dtype = y.device, y.dtype

        B, D = y.shape
        # você pode usar o mesmo eps que em toda parte:
        eps = torch.full_like(y, 1e-6)

        grad_list = []
        for i in range(B):
            yi = y[i]
            go_i = grad_output[i]             # [m]
            eps_i = eps[i]                    # [D]
            grad_i = torch.zeros_like(yi)     # [D]

            for k in range(D):
                delta = torch.zeros_like(yi)
                delta[k] = eps_i[k]

                # compute f(y + δ)
                with torch.no_grad():
                    yi_p = (yi + delta).cpu().numpy()
                    logp = optf.OptimizationLog_Original()
                    fp_np = optf.gT_Original(yi_p, d, 0, logp)
                    fp = torch.as_tensor(fp_np, device=device, dtype=dtype)

                    yi_m = (yi - delta).cpu().numpy()
                    logm = optf.OptimizationLog_Original()
                    fm_np = optf.gT_Original(yi_m, d, 0, logm)
                    fm = torch.as_tensor(fm_np, device=device, dtype=dtype)

                diff = (fp - fm) / (2 * eps_i[k])
                # produto interno go_i · diff
                grad_i[k] = (go_i * diff).sum()

            grad_list.append(grad_i)

        grad_y = torch.stack(grad_list, dim=0)  # [B, D]
        return grad_y, None, None


class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        """
        y: Tensor [B, D]
        d:     dados do problema
        opt_func: módulo com g_TempLog_Original()
        """
        device, dtype = y.device, y.dtype
        y_np = y.detach().cpu().numpy()

        # 1) chama a versão NumPy
        outs = []
        for yi in y_np:
            # g_TempLog_Original retorna um numpy array de tamanho p
            outs.append(-opt_func.g_TempLog_Original(yi, d))

        # 2) empacota num tensor [B, p] no mesmo device/dtype de y
        with torch.no_grad():
            out = torch.stack([
                torch.as_tensor(o, device=device, dtype=dtype)
                for o in outs
            ], dim=0)

        # 3) salva contexto para backward
        ctx.save_for_backward(y)
        ctx.d   = d
        ctx.opt = opt_func
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [B, p]
        retorna: grad_y [B, D], e None para d e opt_func
        """
        y, = ctx.saved_tensors
        d    = ctx.d
        optf = ctx.opt
        device, dtype = y.device, y.dtype

        B, p = grad_output.shape
        D    = y.size(1)

        # passo de finite difference (pode usar a sua função eps)
        eps = torch.full_like(y, 1e-6)

        grads = []
        for i in range(B):
            yi   = y[i]           # [D]
            go_i = grad_output[i] # [p]
            eps_i = eps[i]        # [D]

            grad_i = torch.zeros(D, device=device, dtype=dtype)
            for k in range(D):
                delta = torch.zeros_like(yi)
                delta[k] = eps_i[k]

                # f(y + δ)
                with torch.no_grad():
                    fpos_np = -optf.g_TempLog(
                        (yi + delta).cpu().numpy(), d
                    )
                    fneg_np = -optf.g_TempLog(
                        (yi - delta).cpu().numpy(), d
                    )
                fpos = torch.as_tensor(fpos_np, device=device, dtype=dtype)
                fneg = torch.as_tensor(fneg_np, device=device, dtype=dtype)

                # diferença central
                diff = (fpos - fneg) / (2 * eps_i[k])  # [p]
                # produto vetor–jacobian
                grad_i[k] = (go_i * diff).sum()

            grads.append(grad_i)

        grad_y = torch.stack(grads, dim=0)  # [B, D]
        return grad_y, None, None


class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, optf):
        # 1) salva entrada
        ctx.save_for_backward(y)
        ctx.d   = d
        ctx.opt = optf

        # 2) chama a versão NumPy original para cada amostra
        y_np = y.detach().cpu().numpy()
        outs = [optf.jac_gT_Original(yi, d, 0, optf.OptimizationLog_Original())
                for yi in y_np]

        # 3) empacota em tensor [B, m, n]
        out = torch.stack([
            torch.as_tensor(o, device=y.device, dtype=y.dtype)
            for o in outs
        ], dim=0)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        y,     = ctx.saved_tensors
        d       = ctx.d
        optf    = ctx.opt
        device, dtype = y.device, y.dtype

        B, m, n = grad_output.shape
        eps      = torch.full_like(y, 1e-6)

        grad_list = []
        for i in range(B):
            yi   = y[i]
            go_i = grad_output[i]  # [m,n]
            eps_i = eps[i]         # [n]

            gi = torch.zeros(n, device=device, dtype=dtype)
            for l in range(n):
                δ = torch.zeros_like(yi); δ[l] = eps_i[l]

                # diferenças centrais usando a versão NumPy original
                with torch.no_grad():
                    Jp = optf.jac_gT_Original(
                        (yi + δ).cpu().numpy(), d, 0, optf.OptimizationLog_Original()
                    )
                    Jm = optf.jac_gT_Original(
                        (yi - δ).cpu().numpy(), d, 0, optf.OptimizationLog_Original()
                    )

                Jp = torch.as_tensor(Jp, device=device, dtype=dtype)
                Jm = torch.as_tensor(Jm, device=device, dtype=dtype)

                diff = (Jp - Jm) / (2 * eps_i[l])  # [m,n]
                gi[l] = (go_i * diff).sum()

            grad_list.append(gi)

        grad_y = torch.stack(grad_list, dim=0)  # [B,n]
        return grad_y, None, None


class JacTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, opt_func):
        """
        y: [B, n_vars]
        returns: [B, p] – cada linha é a jac_TempLog_Original para uma amostra
        """
        # Salva y para usar no backward
        ctx.save_for_backward(y)
        ctx.d   = d
        ctx.opt = opt_func

        # Chama a implementação NumPy original
        y_np = y.detach().cpu().numpy()
        outs = [opt_func.jac_TempLog_Original(yi, d) for yi in y_np]
        # Empacota como tensor
        out = torch.stack([
            torch.as_tensor(o, device=y.device, dtype=y.dtype)
            for o in outs
        ], dim=0)  # shape [B, p]
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/d(out), shape [B, p]
        O gradiente d(out)/d(y) é zero (JacTempLog independe de y),
        então dL/dy = grad_output @ 0 = 0.
        """
        y, = ctx.saved_tensors
        # Retorna um tensor de zeros para y, e None para d e opt_func
        return torch.zeros_like(y), None, None


