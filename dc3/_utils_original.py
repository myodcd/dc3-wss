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

from torch.autograd import Function

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
        #self.hmin = d.hmin[0]
        #self.hmax = d.hmax[0]
        #self.tar_beg = d.tar_beg
        #self.tariff_value = d.tariff_value
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


    # def Cost
    def obj_fn(self, y):  # ,d, pumps, tanks, pipes, valves, timeInc):
        # COM EFICIÊNCIA

        log_cost=opt_func.CostOptimizationLog()

        y_np = y
        cost_list = [opt_func.Cost(i, self.d, log_cost, 3) for i in y_np]
        cost_tensor = torch.tensor(cost_list, requires_grad=True)

        return cost_tensor


    def gT(self, x, y):
        log_tank = opt_func.TanksOptimizationLog()

        #with torch.no_grad():
        #y_np = y.detach().cpu().numpy()
        y_np = y
        gt_list = [opt_func.gT(i, self.d, 0, log_tank) for i in y_np]
        gt_list = torch.stack(gt_list)
        
        #gt_list_min_max = gt_list
        
        #gt_list = torch.clamp(gt_list,2,8)
        
        return gt_list


    def g_tankLevel(self, y):
        log_tank = opt_func.TanksOptimizationLog()
        #y_np = y.detach().cpu().numpy()
        
        y_np = y

        
        g_tl = [opt_func.gT(i, self.d, 0, log_tank) for i in y_np]
        #gt_list = torch.stack(gt_list)
        
        #gt_up = gt_list - n_max
        #gt_down = n_min - gt_list
        
        
        return torch.stack(g_tl)
 
    def g_TempLog(self, x): 
        #with torch.no_grad():
        #x_np = x.detach().cpu().numpy()
        x_np = x
                
        g_templog_list = [opt_func.g_TempLog(i,self.d) for i in x_np]
                         
        return torch.stack(g_templog_list)

    def g_overlap(self, y):
        horarios = y[:, :5]
        duracoes = y[:, 5:]
        overlap = torch.zeros(y.shape[0], 5, device=y.device)

        for i in range(5):
            h_i = horarios[:, i]
            f_i = h_i + duracoes[:, i]
            for j in range(i + 1, 5):
                h_j = horarios[:, j]
                f_j = h_j + duracoes[:, j]

                ini_sobre = torch.maximum(h_i, h_j)
                fim_sobre = torch.minimum(f_i, f_j)
                overlap_ij = torch.clamp(fim_sobre - ini_sobre, min=0.0)  # nova variável temporária

                overlap[:, i] = torch.where(overlap_ij > 0.0, torch.tensor(1.0, device=y.device), overlap[:, i])

            return overlap  # [n x 5]

    def g_tariff(self, y):
        """
        Calcula as penalidades de acionamento em horários de tarifa alta.
        Entrada: y [batch_size, 10] → [h0..h4, d0..d4]
        Saída: penalidades [batch_size, 5]
        """
        duracao_tarifas = torch.tensor([2, 4, 1, 2, 3, 12], device=y.device)
        valores_tarifas = torch.tensor([0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094], device=y.device)

        inicio_tarifas = torch.cumsum(torch.cat([torch.tensor([0.0], device=y.device), duracao_tarifas[:-1]]), dim=0)
        fim_tarifas = inicio_tarifas + duracao_tarifas

        horarios = y[:, :5]         # [batch, 5]
        duracoes = y[:, 5:]         # [batch, 5]
        fim_horarios = horarios + duracoes

        tarifa_limite = 0.13
        penalidade_alta = 1000.0

        mask_tarifas = valores_tarifas >= tarifa_limite
        inicio_altas = inicio_tarifas[mask_tarifas]
        fim_altas = fim_tarifas[mask_tarifas]

        h_ini = horarios.unsqueeze(2)      # [batch, 5, 1]
        h_fim = fim_horarios.unsqueeze(2)  # [batch, 5, 1]
        t_ini = inicio_altas.view(1, 1, -1)  # [1, 1, k]
        t_fim = fim_altas.view(1, 1, -1)     # [1, 1, k]

        sobre_ini = torch.maximum(h_ini, t_ini)
        sobre_fim = torch.minimum(h_fim, t_fim)
        dur_sobreposta = torch.clamp(sobre_fim - sobre_ini, min=0.0)

        penalidades = dur_sobreposta * penalidade_alta
        penal_total = penalidades.sum(dim=2)  # [batch, 5]

        return penal_total



    def ineq_dist(self, x, y):

        ineq_dist = self.ineq_resid(x, y)

        return ineq_dist



    def get_yvars(self,y):
        d = self.d
        g_tl = self.g_tankLevel(y)
        g_tlog = self.g_TempLog(y)
        g_o = self.g_overlap(y)  # [n x 5]
        g_tf = self.g_tariff(y)  # [n x 5]
        
        
        return g_tl, g_tlog, g_o, g_tf, d

    # ineq_dist
    def ineq_resid(self, x, y):
        
        
        
        g_tl, g_tlog, g_o, g_tf, d = self.get_yvars(y)

        resids = torch.cat([g_tl - d.hmax[0],d.hmin[0] - g_tl,g_tlog, g_tf]
            , dim=1)  # [n x 25]

        #gt_up, gt_down = self.g_tankLevel(x, y)
        
        #g_templog = torch.clamp(self.g_TempLog(y),0)
        
        #overlap = self.g_overlap(y)  # [n x 5]
        
        # torch.round((y *100)/100)
        
        #tariff = self.g_tariff(y)  # [n x 5]
        
        

        # Ajuste para garantir que o total seja 35 restrições (somando 10 + 5 + 10 + 5 + 5)
        return resids
    
            
    def jac_gT_teste(self, y):
        
        log_tank = opt_func.TanksOptimizationLog()
        
        class JacGTFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_input, d, log_tank):
                y_np = y_input.detach().cpu().numpy()
                output_list = [opt_func.jac_gT(i, d, 0, log_tank) for i in y_np]
                output = torch.tensor(output_list, dtype=y_input.dtype, device=y_input.device)
                
                # Salva para o backward (se possível — depende do que você pode fazer)
                ctx.save_for_backward(y_input)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                y_input, = ctx.saved_tensors



                # Aqui você precisa colocar a derivada de jac_gT em relação a y.
                # Se isso não for possível, você pode usar um truque como identidade:
                grad_input = grad_output  # ou grad_output * some_jacobian_estimation
                return grad_input, None, None  # uma entrada para cada argumento da forward
        


        jac_gt_tensor = JacGTFunction.apply(y, self.d, log_tank)


    def jac_gT(self, y):
  
        log_tank = opt_func.TanksOptimizationLog()

        y_np = y
        
        jac_gt_list = [ opt_func.jac_gT(i, self.d, 0, log_tank)  for i in y_np]

        jac_list_pos = torch.tensor(jac_gt_list, device=y.device)
        
        jac_list_neg = -jac_list_pos.clone()                

        jac_list = torch.cat((jac_list_pos, jac_list_neg), dim=1)
        
        
        return jac_list


    def jac_TempLog(self, x):

        x_np = x
        jac_templog_list = torch.clamp(opt_func.jac_TempLog(x_np, self.d),0)
        
        jac_templog_list = torch.tensor(jac_templog_list, device=x.device)
        
        return jac_templog_list






    def g_tankLevel_original(self, x, y):
        log_tank = opt_func.TanksOptimizationLog()
        #y_np = y.detach().cpu().numpy()
        y_np = y
        n_min = 2
        n_max = 8
        
        gt_list = [opt_func.gT(i, self.d, 0, log_tank) for i in y_np]
        gt_list = torch.stack(gt_list)
        
        gt_list_min_max = torch.clamp(gt_list,n_min,n_max)
                
        
        gt_up = gt_list_min_max - n_max
        gt_down = n_min - gt_list_min_max

        return gt_up, gt_down





    def g_tariff_old(self, y):
        duracao_tarifas = torch.tensor([2, 4, 1, 2, 3, 12], device=y.device)
        valores_tarifas = torch.tensor([0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094], device=y.device)

        inicio_tarifas = torch.cumsum(torch.cat([torch.tensor([0.0], device=y.device), duracao_tarifas[:-1]]), dim=0)
        fim_tarifas = inicio_tarifas + duracao_tarifas

        horarios = y[:, :5]
        duracoes = y[:, 5:]

        penalidade_base = 1000.0  # Penalidade proporcional

        restricoes = []

        for i in range(5):
            h_ini = horarios[:, i]
            h_fim = h_ini + duracoes[:, i]
            penal = torch.zeros_like(h_ini)

            for j in range(len(valores_tarifas)):
                t_ini = inicio_tarifas[j]
                t_fim = fim_tarifas[j]
                tarifa = valores_tarifas[j]

                sobre_ini = torch.maximum(h_ini, t_ini)
                sobre_fim = torch.minimum(h_fim, t_fim)
                duracao_sobreposta = torch.clamp(sobre_fim - sobre_ini, min=0.0)

                penal += duracao_sobreposta * tarifa * penalidade_base

            restricoes.append(penal.unsqueeze(1))

        return torch.cat(restricoes, dim=1)  # [n x 5]







    def jac_overlap(self,horarios, duracoes):
        
        n = horarios.shape[0]
        jac_sobreposicoes = torch.zeros((n, 5, 5), device=horarios.device)

        for i in range(5):
            for j in range(i + 1, 5):
                # Horário de início e término das bombas i e j
                h_i = horarios[:, i]
                h_j = horarios[:, j]
                f_i = h_i + duracoes[:, i]
                f_j = h_j + duracoes[:, j]

                # Cálculo da sobreposição
                sobre_ini = torch.maximum(h_i, h_j)
                sobre_fim = torch.minimum(f_i, f_j)
                overlap = torch.clamp(sobre_fim - sobre_ini, min=0.0)

                # Derivadas parciais para o jacobiano
                d_overlap_d_hi = torch.where(overlap > 0, -1.0 * (sobre_ini < f_i), 0.0)
                d_overlap_d_hj = torch.where(overlap > 0, -1.0 * (sobre_ini < f_j), 0.0)
                d_overlap_d_duracao_i = torch.where(overlap > 0, 1.0 * (sobre_fim > h_i), 0.0)
                d_overlap_d_duracao_j = torch.where(overlap > 0, 1.0 * (sobre_fim > h_j), 0.0)

                # Preencher o jacobiano com as derivadas
                jac_sobreposicoes[:, i, j] = d_overlap_d_hi
                jac_sobreposicoes[:, j, i] = d_overlap_d_hj
                jac_sobreposicoes[:, i, i] = d_overlap_d_duracao_i
                jac_sobreposicoes[:, j, j] = d_overlap_d_duracao_j

        return jac_sobreposicoes


    def jac_tariff(self, y):

        
        y = y.clone().detach().requires_grad_(True)
        penal = self.g_tariff(y)  # [batch_size, 5]

        batch_size = y.shape[0]
        jacobian = torch.zeros((batch_size, 5, 10), device=y.device)

        for i in range(5):
            grads = torch.autograd.grad(
                outputs=penal[:, i],
                inputs=y,
                grad_outputs=torch.ones_like(penal[:, i]),
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            jacobian[:, i, :] = grads

        return jacobian


    def ineq_jac(self, Y):
        jac_gT = self.jac_gT(Y)  # [n x 20 x 10]
        jac_TempLog = self.jac_TempLog(Y)  # [5 x 10]

        # Jacobiano das tarifas (restrições de tarifas)
        jac_tariff = self.jac_tariff(Y)  # [n x 10]
        jac_tariff = jac_tariff.unsqueeze(1).repeat(1, 5, 1)  # [n x 5 x 10]
        
        # Ajustar jac_TempLog para que tenha a forma [n x 5 x 10]
        # Repetir jac_TempLog para cada amostra em n
        if jac_tariff.dim() == 2:
            jac_tariff = jac_tariff.unsqueeze(1).repeat(1, 5, 1)  # [n x 5 x 10]
        elif jac_tariff.dim() == 3:
            pass  # já está correto
        else:
            raise ValueError(f"jac_tariff com dimensões inesperadas: {jac_tariff.shape}")        
        # Jacobiano das sobreposições
        jac_overlap = self.jac_overlap(Y[:, :5], Y[:, 5:])  # [n x 5 x 5]
        jac_overlap = jac_overlap.repeat(1, 1, 2) 
        # Concatenando ao longo da dimensão 1 (a segunda dimensão das matrizes)
        #jac_combined = torch.cat([jac_gT, jac_TempLog, jac_tariff, jac_overlap], dim=1)  # [n x 25 x 10]

        jac_combined = torch.cat([jac_gT, jac_TempLog, jac_tariff], dim=1)  # [n x 25 x 10]


        return jac_combined

        
    def ineq_jac_original(self, Y):

        # [samples x 20 x 10]
        jac_gT = self.jac_gT(Y)

        # [5 x 10]
        jac_TempLog = self.jac_TempLog(Y)

        # [samples x 5 x 10]
        jac_combined = torch.cat([jac_gT, jac_TempLog.unsqueeze(0).repeat(jac_gT.shape[0], 1, 1)], dim=1)

        return jac_combined

    def ineq_grad_original(self, x, y):
        # [samples x 25] ineq_resid = ineq_dist
        ineq_dist_relu = torch.clamp(self.ineq_dist(x, y),0)
        
        # [samples x 1 x 25]
        ineq_dist_expanded = ineq_dist_relu.unsqueeze(1)  

        # [samples X 25 X 10]
        ineq_jac = self.ineq_jac(y)
        
        return torch.matmul(ineq_dist_expanded, ineq_jac).squeeze(1)


    def ineq_grad(self, x, y):
        # [samples x 25]
        ineq_resid = self.ineq_dist(x, y)

        # Máscara booleana onde há violação (valor > 0)
        

        
        #mask = (ineq_resid > 0).float()  # [samples x 25]

        # Jacobiano das restrições (samples x 25 x 10)
        ineq_jac = self.ineq_jac(y)  # [batch x n_constraints x n_vars]

        # Aplica a máscara no jacobiano: zera os gradientes onde não há violação
        #masked_jac = ineq_jac * mask.unsqueeze(2)  # Broadcasting
        #masked_jac = masked_jac.float()

        # Produto do resíduo clamped com o jacobiano mascarado
        grad = torch.matmul(ineq_resid.unsqueeze(1), ineq_jac).squeeze(1)  # [batch x 10]

#        return 2 * torch.clamp(grad,0)
    
        return 2 * grad
    
    #    return torch.clamp(grad,0)
    


    def process_output(self, X, out):
        qty = out.shape[1] // 2

        start_times = out[:, :qty] * 23.8
        durations = out[:, qty:] * (5.0 - 0.1) + 0.1

        # Ordena os tempos de início
        start_times_sorted, sorted_idx = torch.sort(start_times, dim=1)
        durations_sorted = torch.gather(durations, 1, sorted_idx)

        # Corrige sobreposições impondo espaçamento de 0.001
        adjusted_start_times = start_times_sorted.clone()

        for i in range(1, qty):
            prev_end = adjusted_start_times[:, i - 1] + durations_sorted[:, i - 1]
            min_start = prev_end + 0.001  # Espaço mínimo obrigatório

            current_start = adjusted_start_times[:, i]
            adjusted_value = torch.maximum(current_start, min_start)

            # Atualiza o tensor ajustado
            adjusted_start_times = torch.cat([
                adjusted_start_times[:, :i],
                adjusted_value.unsqueeze(1),
                adjusted_start_times[:, i+1:]
            ], dim=1)

        # Corrige durações para não ultrapassar o tempo máximo
        end_times = adjusted_start_times + durations_sorted
        over_limit = end_times > 23.8
        durations_sorted = torch.where(over_limit, 23.8 - adjusted_start_times, durations_sorted)

        # Reaplica os clamps
        adjusted_start_times = torch.clamp(adjusted_start_times, 0, 23.8)
        durations_sorted = torch.clamp(durations_sorted, 0.1, 5)

        return torch.cat([adjusted_start_times, durations_sorted], dim=1)

    def process_output_original(self, X, out):
        qty = out.shape[1] // 2

        start_times = out[:, :qty] * 23.8
        durations = out[:, qty:] * (5.0 - 0.1) + 0.1

        # Ordena os tempos de início
        start_times_sorted, sorted_idx = torch.sort(start_times, dim=1)
        durations_sorted = torch.gather(durations, 1, sorted_idx)

        # Corrige sobreposições de forma segura (sem in-place)
        adjusted_start_times = start_times_sorted.clone()

        for i in range(1, qty):
            prev_end = adjusted_start_times[:, i - 1] + durations_sorted[:, i - 1]
            adjusted_start_times = torch.cat([
                adjusted_start_times[:, :i],
                torch.max(adjusted_start_times[:, i:i+1], prev_end.unsqueeze(1)),
                adjusted_start_times[:, i+1:]
            ], dim=1)

        # Corrige durações para não ultrapassar o tempo máximo
        end_times = adjusted_start_times + durations_sorted
        over_limit = end_times > 23.8
        durations_sorted = torch.where(over_limit, 23.8 - adjusted_start_times, durations_sorted)

        # Reaplica os clamps
        adjusted_start_times = torch.clamp(adjusted_start_times, 0, 23.8)
        durations_sorted = torch.clamp(durations_sorted, 0.1, 5)

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
        grad_g1 = torch.cat((y1 * grad_g1_x1 * dists[:, 0:1],
                            y2 * grad_g1_x2 * dists[:, 0:1]), dim=1)

        # Gradientes para g2
        grad_g2_x1 = x2
        grad_g2_x2 = x1
        grad_g2 = torch.cat((y1 * grad_g2_x1 * dists[:, 1:2],
                            y2 * grad_g2_x2 * dists[:, 1:2]), dim=1)

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
