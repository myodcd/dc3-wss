import torch
import torch.nn as nn

import torch.nn.functional as F

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

from torch.autograd import Function
import ClassAutogradPyTorch as autograd_pt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("xpu" if torch.xpu.is_available() else DEVICE)


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



def parser_tt_to_td(y):

    if y.shape[1] % 2:
        y_last_column = y[:,-1:]
        y = y[:,:-1]
        
    n, cols = y.shape
    qty = cols // 2  # Quantidade de pares (hora, duração)

    # Divide as colunas em horas e durações
    horarios = y[:, :qty]  # Primeira metade (horas)
    duracoes = y[:, qty:]  # Segunda metade (durações)

    # Intercala horas e durações
    y_intercalado = torch.empty_like(y)
    y_intercalado[:, ::2] = horarios
    y_intercalado[:, 1::2] = duracoes
        
    
    y_intercalado = torch.cat([y_intercalado, y_last_column], dim=1)  if y.shape[1] % 2  else y_intercalado

    return y_intercalado

def parser_td_to_tt(y):

    n, cols = y.shape
    qty = cols // 2  # Quantidade de pares (hora, duração)

    # Seleciona as colunas de horas e durações
    horarios = y[:, ::2]  # Colunas pares (horas)
    duracoes = y[:, 1::2]  # Colunas ímpares (durações)

    # Concatena horas e durações
    y_separado = torch.cat([horarios, duracoes], dim=1)

    return y_separado   

    
###################################################################

# PROBLEM DC_WSS

###################################################################


class Problem_DC_WSS:
    def __init__(self, d, x, valid_frac=0.0833, test_frac=0.0833):

        self._d = d
        self._X = torch.tensor(x)
        self._xdim = self._X.shape[1]
        self._ydim = self._X.shape[1]
        self._num = self._X.shape[0]
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._device = None
        self._qty_samples = self._X.shape[0]
        self.hmin = 2
        self.hmax = 8
        self.timemin = 0.001
        self.timemax = 23.9999
        self.tariff_value = [0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094]
        self.tariff_time = [2, 4, 1, 2, 3, 12]

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



 
    
    # def Cost
    def obj_fn(self, y, args):  # ,d, pumps, tanks, pipes, valves, timeInc):
        # COM EFICIÊNCIA

        y_parsed = parser_td_to_tt(y) if args['vector_format'] == 'td-td' else y


        log = opt_func.CostOptimizationLog()

        result = torch.tensor([opt_func.Cost(i, self.d, log, 3) for i in y_parsed])
        
        return result

    # def Cost
    def obj_fn_com_autograd(self, y):  # ,d, pumps, tanks, pipes, valves, timeInc):
        # COM EFICIÊNCIA

        result = autograd_pt.CostAutograd.apply(y, self.d, opt_func)

        return result


    def gT(self, y, args):

        #print('y depois parser ', y[0])
        #print('- - - - ')

        log = opt_func.TanksOptimizationLog()

        result = torch.tensor([opt_func.gT(i, self.d, 0, log) for i in y])
        
        #print('resultado gt[0] ', result[0])
        #print('- - - - ')
        #result = parser_tt_to_td(result) if  args['vector_format'] == 'td-td' else result
        
        
        return result

    def gT_com_autograd(self, y, limit=True):
        result = autograd_pt.GTAutograd.apply(y, self.d, opt_func)

        #if limit:
        #    result = torch.clamp(result, 2, 8)

        return result

    def g_TempLog(self, y):

        g_templog_list = [-1 * opt_func.g_TempLog(i, self.d) for i in y]

        return torch.stack(g_templog_list)



    def g_x(self, y, args, numpy=False ):
        
        
        #y = parser_td_to_tt(y) if args['vector_format'] == 'td-td' else y
        #y = y[:,:3]
        
        y_np = y
        
        if numpy:
            y_np = y.detach().numpy() if isinstance(y, torch.Tensor) else y
        
        return y_np


    def ineq_dist(self, x, y, args):

        ineq_dist = torch.relu(self.ineq_resid(x, y, args))
        
        #print('ineq_dist relu', ineq_dist[0])
        
        return ineq_dist

    def get_yvars(self, y, args):


        #print('y antes parser ', y[0])
        #print('- - - - ')
        
        y_parsed = parser_td_to_tt(y) if args['vector_format'] == 'td-td' else y

        g_tl = self.gT(y_parsed, args)
        g_tlog = self.g_TempLog(y)
        g_x = self.g_x(y_parsed, args)

        return g_tl, g_tlog, g_x


    # ineq_dist
    def ineq_resid(self, x, y, args):

        gt, g_tlog, g_x = self.get_yvars(y, args)

   
        gt_grater = gt - self.hmax
        gt_minor = self.hmin - gt
        
        g_x_grater = g_x - self.timemax
        g_x_minor = self.timemin - g_x

        #print('shape ineq_dist ', torch.cat(
        #    [gt_grater, gt_minor, g_x_grater, g_x_minor, g_tlog], dim=1
        #)[0].shape)
        
        

        resids = torch.cat(
            [gt_grater, gt_minor, g_x_grater, g_x_minor, g_tlog], dim=1
        )
        
        
        #print('Ineq Dist ', resids[0])
        #print('- - - - ')      

        return resids

    def jac_gT_before_parser(self, y, args):


        log = opt_func.TanksOptimizationLog()

        jac = torch.tensor([opt_func.jac_gT(i, self.d, 0, log) for i in y])
        

        jac_pos = jac
        jac_neg = -jac.clone()

        return torch.cat([jac_pos, jac_neg], dim=1)  # [batch, 20, 10]
    
    def jac_gT(self, y, args):

        #y = parser_td_to_tt(y) if args['vector_format'] == 'td-td' else y

        log = opt_func.TanksOptimizationLog()

        jac = torch.tensor([opt_func.jac_gT(i, self.d, 0, log) for i in y])
        
        #print('jac_gT_before_parser ', jac[0])
        #print('- - - - ') 
        
        jac = torch.stack([parser_tt_to_td(i) for i in jac]) if args['vector_format'] == 'td-td' else jac

        
        #print('jac_gT_after_parser ', jac[0])
        #print('- - - - ') 
        
        jac_pos = jac
        jac_inv = -jac.clone()
        
        all_jac_gT = torch.cat([jac_pos, jac_inv], dim=1)  # [batch, 20, 10]

        #print('all_jac_gT ', all_jac_gT[0].shape)
        #print('jac_gT_pos_neg ', all_jac_gT[0])
        #print('- - - - ') 

        return all_jac_gT # [batch, 20, 10]

    def jac_gT_com_autograd(self, y):

        result = autograd_pt.JacGTAutograd.apply(y, self.d, opt_func)

        return result

    def jac_TempLog(self, y):

        jac_templog_list = opt_func.jac_TempLog(y, self.d)
        
        #print('jac_TempLog ', jac_templog_list[0])
        #print('- - - - ') 
        
        return jac_templog_list


    def jac_x(self, y, args):
        
        #y = y[:,:3]
        
        eps_aux = [opt_func.eps_definition_F3(i, self.d) for i in y]

        y_np = y.detach().numpy() if isinstance(y, torch.Tensor) else y    
        eps_aux = torch.stack(eps_aux, dim=0)
        eps_aux = eps_aux.detach().numpy() if isinstance(eps_aux, torch.Tensor) else eps_aux
        
        jac_x_list = torch.tensor([approx_fprime(y_np[i], self.g_x, eps_aux[i], *(1, args)) for i in range(len(y_np))])
        
        
        jac_x_pos = jac_x_list
        jac_x_neg = -jac_x_list.clone()
        
        #print('jac_x ', torch.cat([jac_x_pos, jac_x_neg], dim=1).shape)        
        #print('jac_x ', torch.cat([jac_x_pos, jac_x_neg], dim=1))
        #print('- - - - ') 
        
        return torch.cat([jac_x_pos, jac_x_neg], dim=1)  # [batch, 20, 10]
       
    def ineq_jac(self, y,args):


        y_parsed = parser_td_to_tt(y) if args['vector_format'] == 'td-td' else y

        # jac_gT: [batch, 5, 10] (supondo que você tenha 5 ciclos e 10 variáveis)
        jac_gT = self.jac_gT(y_parsed, args)  # Output de jac_gT, forma: [batch, 5, 10]

        # jac_TempLog: [batch, 5], será ajustado para [batch, 5, 1]
        jac_TempLog = self.jac_TempLog(y)  # Output de jac_TempLog, forma: [batch, 5]
        # jac_TempLog = jac_TempLog.unsqueeze(2)

        jac_TempLog = jac_TempLog.unsqueeze(0).repeat(
            jac_gT.shape[0], 1, 1
        )  # Forma ajustada: [batch, 5, 1]


        jac_x = self.jac_x(y_parsed, args)  # [batch, 5]

        jac_combined = torch.cat([jac_gT, jac_x, jac_TempLog], dim=1)

        return jac_combined

    def ineq_grad(self, x, y, args):
        # [samples x 25]
        ineq_dist = self.ineq_dist(x, y, args)
        ineq_jac = self.ineq_jac(y, args)  # [batch x n_constraints x n_vars]
                                
        return ineq_jac.transpose(1,2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)

            
    def process_output(self, X, out):
        
        out2 = nn.Sigmoid()(out)  # Normaliza a saída para [0, 1]
        
        qty = out2.shape[1] // 2  # Quantidade de duty cycles

        # Escala os tempos normalizados para [0, 23.8] e durações para [0.1, 5.0]
        start_times = out2[:, :qty] * 23.8
        durations = out2[:, qty:] * (5.0 - 0.1) + 0.1

        # Ordena os horários de início e ajusta as durações na mesma ordem
        start_times_sorted, sorted_idx = torch.sort(start_times, dim=1)
        durations_sorted = durations.gather(1, sorted_idx)

        # Garante que os ciclos não se sobreponham
        adjusted_start_times = [start_times_sorted[:, 0:1]]
        for i in range(1, qty):
            prev_end = adjusted_start_times[i - 1] + durations_sorted[:, i - 1:i]
            min_start = prev_end + 0.001  # Pequeno intervalo de segurança
            current_start = torch.max(start_times_sorted[:, i:i+1], min_start)
            adjusted_start_times.append(current_start)

        adjusted_start_times = torch.cat(adjusted_start_times, dim=1)

        # Garante que nenhum ciclo ultrapasse o final do dia
        max_end = adjusted_start_times + durations_sorted
        durations_sorted = torch.where(max_end > 23.8, 23.8 - adjusted_start_times, durations_sorted)

        # Trunca valores finais para garantir domínio válido
        adjusted_start_times = torch.clamp(adjusted_start_times, 0, 23.8)
        durations_sorted = torch.clamp(durations_sorted, 0.1, 5.0)

        return torch.cat([adjusted_start_times, durations_sorted], dim=1)


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


###################################################################

# PROBLEM NON LINEAR - 2 INEQUALITIES

###################################################################


class Problem_Non_Linear_2Ineq:
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
        return "Problem_Non_Linear_2ineq-{}-{}-{}-{}".format(
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

    def ineq_g1(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return 0.75 * x1**2 + 0.25 * x2**2 - 0.5

    def ineq_g2(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return x1 * x2 - 0.3

    def ineq_dist(self, x, y):
        """
        Clampe os resíduos de desigualdade das duas restrições.
        """
        resids_g1 = torch.clamp(self.ineq_g1(x), 0)
        resids_g2 = torch.clamp(self.ineq_g2(x), 0)

        resids = torch.stack([resids_g1, resids_g2], dim=1)  # shape: [batch, 2]
        return resids

    def eq_grad(self, x, y):
        """
        Gradiente do resíduo de igualdade.
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
        Gradiente combinado dos resíduos de desigualdade g1 e g2.
        """
        x1 = x[:, 0].unsqueeze(1)
        x2 = x[:, 1].unsqueeze(1)

        # Distâncias (resíduos clampados)
        dists = self.ineq_dist(x, y)  # shape: [batch, 2]

        # y pesos para cada restrição e variável
        y1 = y[:, 0].unsqueeze(1)
        y2 = y[:, 1].unsqueeze(1)

        # Gradientes para g1
        grad_g1_x1 = 1.5 * x1
        grad_g1_x2 = 0.5 * x2
        grad_g1 = torch.cat(
            (y1 * grad_g1_x1 * dists[:, 0:1], y2 * grad_g1_x2 * dists[:, 0:1]), dim=1
        )

        # Gradientes para g2
        grad_g2_x1 = x2
        grad_g2_x2 = x1
        grad_g2 = torch.cat(
            (y1 * grad_g2_x1 * dists[:, 1:2], y2 * grad_g2_x2 * dists[:, 1:2]), dim=1
        )

        grad = grad_g1 + grad_g2
        grad = torch.clamp(grad, 0)
        return grad

    def ineq_partial_grad(self, X, Y):
        """
        Versão combinada parcial: aplica gradientes para g1 e g2 separadamente.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        dists = self.ineq_dist(X, Y)  # shape: [batch, 2]

        # Gradientes para g1
        grad_g1_x1 = 1.5 * x1 * dists[:, 0]
        grad_g1_x2 = 0.5 * x2 * dists[:, 0]

        # Gradientes para g2
        grad_g2_x1 = x2 * dists[:, 1]
        grad_g2_x2 = x1 * dists[:, 1]

        # Soma dos gradientes
        Y_out = torch.zeros_like(X)
        Y_out[:, 0] = grad_g1_x1 + grad_g2_x1
        Y_out[:, 1] = grad_g1_x2 + grad_g2_x2

        return Y_out

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
