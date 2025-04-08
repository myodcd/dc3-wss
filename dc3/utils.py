import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
from scipy.optimize import approx_fprime
import math

import numpy as np

# import osqp
from qpth.qp import QPFunction

# import cyipopt
# from scipy.linalg import svd
# from scipy.sparse import csc_matrix

import hashlib
import scipy.io as spio
import time
import EPANET_API as EPA_API

import OptimAuxFunctionsV2 as opt_func

from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("xpu" if torch.xpu.is_available() else DEVICE)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError("{value} is not a valid boolean value")


def my_hash(string):
    return hashlib.sha1(bytes(string, "utf-8")).hexdigest()


###################################################################

# PROBLEM DC_WSS

###################################################################


class Problem_DC_WSS:
    def __init__(self, d, x, valid_frac=0.0833, test_frac=0.0833):

        self._d = d
        self._num_dc = d.num_dc
        self._X = torch.tensor(x)
        self._xdim = self._X.shape[1]
        self._ydim = self._X.shape[1]
        self._num = len(x)
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._device = None
        self._qty_samples = self._X.shape[0]

    @property
    def device(self):
        return self._device

    @property
    def X(self):
        return self._X

    @property
    def d(self):
        return self._d

    @property
    def num_dc(self):
        return self._num_dc

    @property
    def num(self):
        return self._num

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[: int(self.train_frac * self.num)]

    @property
    def validX(self):
        return self.X[
            int(self.train_frac * self.num) : int(
                (self.num * (self.train_frac + self.valid_frac))
            )
        ]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)) :]

    @property
    def qty_samples(self):
        return self._qty_samples


    def custom_temporal_penalty(self, Y, alpha_overlap=10.0):
        qty = Y.shape[1] // 2
        starts = Y[:, :qty]
        durations = Y[:, qty:]

        margin = 0.1
        overlap_penalty = 0.0

        for i in range(qty - 1):
            end_i = starts[:, i] + durations[:, i] + margin
            next_start = starts[:, i + 1]
            diff = end_i - next_start  # Quanto ultrapassa

            # Penalidade só se houver sobreposição
            penalty = torch.relu(diff)
            overlap_penalty += penalty.mean()

        return alpha_overlap * overlap_penalty


    # def Cost
    def obj_fn(self, y):  # ,d, pumps, tanks, pipes, valves, timeInc):
        # COM EFICIÊNCIA

        log_cost=opt_func.CostOptimizationLog()
        
        with torch.no_grad():
            y_np = y.detach().cpu().numpy()
            
            cost_list = [opt_func.Cost(i, self.d, log_cost, 3) for i in y_np]
            cost_tensor = torch.tensor(cost_list, dtype=torch.float32, device=y.device)

        # Armazenar o custo (para loss), e depois aplicar gradiente manual via backward hook
        cost_tensor.requires_grad_(True)        

        return cost_tensor

    def gT(self, x, y):
        log_tank = opt_func.TanksOptimizationLog()

        #with torch.no_grad():
        y_np = y.detach().cpu().numpy()
        gt_list = [opt_func.gT(i, self.d, 0, log_tank) for i in y_np]
        
        return torch.tensor(gt_list, dtype=torch.float32, device=y.device)

    def g_TempLog(self, x): 
        #with torch.no_grad():
        x_np = x.detach().cpu().numpy()
        g_templog_list = [opt_func.g_TempLog(i, self.d) for i in x_np]
        
        return torch.tensor(g_templog_list, dtype=torch.float32, device=x.device)


    def jac_gT(self, y):
        log_tank = opt_func.TanksOptimizationLog()

        #with torch.no_grad():
        y_np = y.detach().cpu().numpy()
        jac_gt_list = [opt_func.jac_gT(i, self.d, 0, log_tank) for i in y_np]
        
        #jac_gt_list_partial = partial(jac_gt_list)

        jac_list_pos = torch.tensor(jac_gt_list, dtype=torch.float32, device=y.device)
        jac_list_neg = -jac_list_pos.clone()                

        jac_list = torch.cat((jac_list_pos, jac_list_neg), dim=1)
        
        return jac_list

        
    def jac_TempLog(self, x):
        #with torch.no_grad():
        x_np = x.detach().cpu().numpy()
        jac_templog_list = opt_func.jac_TempLog(x_np, self.d)
        
        return torch.tensor(jac_templog_list, dtype=torch.float32, device=x.device)


    def ineq_dist(self, x, y):

        ineq_dist = self.ineq_resid(x, y)

        return ineq_dist

    # ineq_dist
    def ineq_resid(self, x, y):

        d = self.d
        n_min = d.hmin[0]  # 2
        n_max = d.hmax[0]  # 8

        #n_min = 2
        #n_max = 7


        gT = self.gT(x, y)

        # Constraint: [gt - Nmax] <= 0 [samples x 10]
        gt_up = gT - n_max

        # Constraint: [Nmin - gT] <= 0 [samples x 10]
        gt_down = n_min - gT

        # [samples x 5]
        g_TempLog = self.g_TempLog(y)

        # [samples x 25]
        return torch.cat([gt_up, gt_down, g_TempLog], dim=1)

    def ineq_jac(self, Y):

        # [samples x 20 x 10]
        jac_gT = self.jac_gT(Y)

        # [5 x 10]
        jac_TempLog = self.jac_TempLog(Y)

        # [samples x 5 x 10]
        jac_combined = torch.cat([jac_gT, jac_TempLog.unsqueeze(0).repeat(jac_gT.shape[0], 1, 1)], dim=1)




        return jac_combined

    def ineq_grad(self, x, y):
        # [samples x 25] ineq_resid = ineq_dist
        ineq_dist_relu = torch.clamp(self.ineq_dist(x, y),0)
        
        # [samples x 1 x 25]
        ineq_dist_expanded = ineq_dist_relu.unsqueeze(1)  

        # [samples X 25 X 10]
        ineq_jac = self.ineq_jac(y)        

        return torch.matmul(ineq_dist_expanded, ineq_jac).squeeze(1)
    
    
        
    def process_output(self, X, out):
        qty = out.shape[1] // 2

        start_times = out[:, :qty] * 23.9
        durations = out[:, qty:] * (5.0 - 0.1) + 0.1

        end_times = start_times + durations
        over_limit = end_times > 23.9
        durations[over_limit] = 23.9 - start_times[over_limit]

        start_times = torch.clamp(start_times, 0, 23.9)
        durations = torch.clamp(durations, 0.1, 5)

        return torch.cat([start_times, durations], dim=1)


###################################################################

# PROBLEM NON LINEAR

###################################################################


class Problem_Non_Linear:
    def __init__(self, X, valid_frac=0.0833, test_frac=0.0833):

        self._X = torch.tensor(X)
        self._xdim = X.shape[1]
        self._ydim = 2
        self._num = X.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._neq = 1
        self._nineq = 1
        self._device = None

        self.partial_vars = 0
        self.other_vars = 1

    @property
    def device(self):
        return self._device

    @property
    def X(self):
        return self._X

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def trainX(self):
        return self.X[: int(self.train_frac * self.num)]

    @property
    def validX(self):
        return self.X[
            int(self.train_frac * self.num) : int(
                (self.num * (self.train_frac + self.valid_frac))
            )
        ]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)) :]

    def __str__(self):
        return "Problem_Non_Linear-{}-{}-{}-{}".format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, x):

        x1 = x[:, 0]
        x2 = x[:, 1]

        return x1 * ((x1 - x2) ** 2 + (x1 - 2)) + 5

    def eq_resid(self, x, y):

        x1 = x[:, 0]
        x2 = x[:, 1]

        return x1**2 / 2 + 1.5 * x2**2 - 1.2

    def ineq_resid(self, x):

        x1 = x[:, 0]
        x2 = x[:, 1]

        return 0.75 * x1**2 + 0.25 * x2**2 - 0.5

    def ineq_dist(self, x, y):
        """
        Clampe os resíduos de desigualdade.
        """
        resids = self.ineq_resid(x)

        resids = resids.unsqueeze(1)

        return torch.clamp(resids, 0)

    def eq_grad(self, x, y):
        """
        Gradiente do resíduo de igualdade.
        Derivadas parciais:
        """
        y1 = y[:, 0].unsqueeze(1)
        y2 = y[:, 1].unsqueeze(1)

        x1 = x[:, 0].unsqueeze(1)
        x2 = x[:, 1].unsqueeze(1)

        grad_x1 = x1 * y1
        grad_x2 = (3 * x2) * y2

        grad = torch.cat((grad_x1, grad_x2), dim=1)

        return grad

    def ineq_grad(self, x, y):
        """
        Gradiente do resíduo de desigualdade.
        Derivadas parciais:
        """
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Calcula a distância clamped
        dist = self.ineq_dist(x, y)  # Tamanho esperado: [25]

        grad_x1 = 1.5 * x1
        grad_x2 = 0.5 * x2

        y1 = y[:, 0].unsqueeze(1)
        y2 = y[:, 1].unsqueeze(1)

        grad_x1 = y1 * grad_x1.unsqueeze(1)
        grad_x2 = y2 * grad_x2.unsqueeze(1)

        grad_x1_scaled = grad_x1 * dist
        grad_x2_scaled = grad_x2 * dist

        grad = torch.cat((grad_x1_scaled, grad_x2_scaled), dim=1)

        grad = torch.clamp(grad, 0)

        return grad

    def ineq_partial_grad_old(self, X, Y):
        # Resíduo ajustado (clamp para respeitar desigualdades)
        x1 = X[:, 0]
        x2 = X[:, 1]
        grad = self.ineq_dist(X, Y).squeeze(1)
        # grad = self.ineq_dist(x1, x2)

        # Inicialização do tensor para gradientes
        Y = torch.zeros(X.shape[0], X.shape[1], device=self.device)

        Y[:, 0] = grad * 1.5 * x1  # Gradiente para x1
        Y[:, 1] = grad * 0.5 * x2  # Gradiente para x2

        # Retornar gradientes ajustados
        return Y

    def ineq_partial_grad(self, X, Y):
        # Assumindo que as duas variáveis são "parciais"
        grad_x1 = 1.5 * X[:, 0]  # Derivada em relação a x1
        grad_x2 = 0.5 * X[:, 1]  # Derivada em relação a x2

        grad = torch.stack([grad_x1, grad_x2], dim=1)

        # A parte efetiva do gradiente pode ser calculada diretamente
        grad_effective = 2 * torch.clamp(
            Y - 0.5, 0
        )  # Ajuste para garantir valores não negativos

        # Atualizando Y com base no gradiente
        Y = torch.zeros(X.shape[0], X.shape[1], device=self.device)
        Y[:, 0] = grad_effective[:, 0]  # Atualiza para x1
        Y[:, 1] = grad_effective[:, 1]  # Atualiza para x2

        return Y

    def process_output(self, X, Y):
        return Y

    def complete_partial(self, X, Z):

        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)

        Y[:, self.partial_vars] = Z.squeeze(1)
        for i in range(Y.shape[0]):
            z = Z[i, 0]
            term = 1.2 - (z**2 / 2)
            x2 = torch.sqrt(
                torch.clamp((2 / 3) * term, min=0)
            )  # Clamp para evitar valores inválidos

            Y[i, self.other_vars] = x2

        return Y
