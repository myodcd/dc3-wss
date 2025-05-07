import epamodule as em

import torch
from torch.autograd import Function
import numpy as np
from epamodule import ENopen, ENclose, ENsolveH, ENgetnodevalue, ENsetnodevalue
# ============== EPANET WRAPPER ==============
class EPANETWrapper:
    def __init__(self, inp_path):
        self.handle = ENopen(inp_path, rptfile=None, binary=None)

    def update_demands(self, y_np):
        for idx, val in enumerate(y_np, start=1):
            ENsetnodevalue(self.handle, idx, ENgetnodevalue, float(val))

    def solve(self):
        ENsolveH(self.handle)

    def get_pressures(self, n):
        return np.array([ENgetnodevalue(self.handle, i, ENgetnodevalue)
                         for i in range(1, n+1)])

    def close(self):
        ENclose(self.handle)

    def batch_sim(self, Y_batch):
        out = []
        for y in Y_batch:
            self.update_demands(y)
            self.solve()
            out.append(self.get_pressures(len(y)))
        return np.stack(out)

# ========== AUXILIARY: VECTORIZED FINITE-DIFF ==========
def finite_diff_batch(sim_fn, Y, eps=1e-6):
    B, D = Y.shape
    f0 = sim_fn(Y)            # [B, C] or [B]
    # Expand for each feature
    Yexp = Y.unsqueeze(1).repeat(1, D, 1)    # [B, D, D]
    idx = torch.arange(D)
    Yexp[:, idx, idx] += eps
    Yflat = Yexp.view(B*D, D)
    fout = sim_fn(Yflat)
    if fout.dim() == 1:
        C = 1
        fplus = fout.view(B, D)
        df = (fplus - f0.unsqueeze(1)) / eps
        return df.unsqueeze(1)  # [B,1,D]
    else:
        C = fout.shape[1]
        fplus = fout.view(B, D, C)
        df = (fplus - f0.unsqueeze(1)) / eps
        return df.permute(0,2,1)  # [B, C, D]

# ========== CUSTOM AUTOGRAD FUNCTIONS ==========
class CostAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, wrapper):
        Y = y.detach().cpu().numpy()
        pressures = wrapper.batch_sim(Y)         # [B, D]
        costs = np.sum((pressures - d.cpu().numpy())**2, axis=1)
        ctx.save_for_backward(y, d)
        ctx.wrapper = wrapper
        return y.new_tensor(costs)

    @staticmethod
    def backward(ctx, grad_output):
        y, d = ctx.saved_tensors
        wrapper = ctx.wrapper
        def sim_cost(Yt):
            press = wrapper.batch_sim(Yt.detach().cpu().numpy())
            c = np.sum((press - d.cpu().numpy())**2, axis=1)
            return torch.from_numpy(c).to(Yt.device)
        df = finite_diff_batch(lambda Yt: sim_cost(Yt).unsqueeze(1), y)
        grad_y = df.squeeze(1) * grad_output.unsqueeze(1)
        return grad_y, None, None

class GTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, wrapper):
        Y = y.detach().cpu().numpy()
        # exemplo: gT = pressures
        pressures = wrapper.batch_sim(Y)         # [B, D]
        ctx.save_for_backward(y)
        return y.new_tensor(pressures)

    @staticmethod
    def backward(ctx, grad_output):
        # sem segunda ordem
        y, = ctx.saved_tensors
        return torch.zeros_like(y), None, None

class GTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, wrapper):
        Y = y.detach().cpu().numpy()
        pressures = wrapper.batch_sim(Y)         # [B, D]
        vals = -np.sum(np.log(pressures + 1e-6), axis=1)
        ctx.save_for_backward(y)
        return y.new_tensor(vals)

    @staticmethod
    def backward(ctx, grad_output):
        # sem segunda ordem
        y, = ctx.saved_tensors
        return torch.zeros_like(y), None, None

class JacGTAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, wrapper):
        # jacobiano de GT w.r.t y
        out = GTAutograd.apply(y, d, wrapper)     # [B, D]
        ctx.save_for_backward(y, d, wrapper)
        return finite_diff_batch(lambda Yt: GTAutograd.apply(Yt, d, wrapper), y)

    @staticmethod
    def backward(ctx, grad_output):
        # sem terceira ordem
        y, _, _ = ctx.saved_tensors
        return torch.zeros_like(y), None, None, None

class JacTempLogAutograd(Function):
    @staticmethod
    def forward(ctx, y, d, wrapper):
        # jacobiano de GTempLog
        ctx.save_for_backward(y, d, wrapper)
        # retorna shape [B, 1, D]
        return finite_diff_batch(lambda Yt: GTempLogAutograd.apply(Yt, d, wrapper), y)

    @staticmethod
    def backward(ctx, grad_output):
        y, _, _ = ctx.saved_tensors
        return torch.zeros_like(y), None, None, None