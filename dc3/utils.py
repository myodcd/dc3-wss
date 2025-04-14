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
        cost_tensor = torch.tensor(cost_list, dtype=torch.float32, requires_grad=True)

        return cost_tensor


    def gT(self, x, y):
        log_tank = opt_func.TanksOptimizationLog()

        #with torch.no_grad():
        y_np = y.detach().cpu().numpy()
        #y_np = y
        gt_list = [opt_func.gT(i, self.d, 0, log_tank) for i in y_np]
        gt_list = torch.stack(gt_list)
        #gt_list_min_max = gt_list
        
        gt_list_min_max = torch.clamp(gt_list,2,8)
        
        return gt_list_min_max


 
    def g_TempLog(self, x): 
        #with torch.no_grad():
        #x_np = x.detach().cpu().numpy()
        x_np = x
        g_templog_list = [opt_func.g_TempLog(i,self.d) for i in x_np]
        
        
        return torch.stack(g_templog_list)



    def finite_difference(self, f, x, h=1e-4, method='central'):
        
        log_tank = opt_func.TanksOptimizationLog()
        
        x = x.detach().clone().requires_grad_(False)
        n = x.numel()
        x = x.view(-1)
        f0 = f(x, self.d, 0, log_tank)
        f0_shape = f0.shape if isinstance(f0, torch.Tensor) else torch.Size([])
        is_scalar = len(f0_shape) == 0

        if is_scalar:
            grad = torch.zeros_like(x)
        else:
            grad = torch.zeros(f0.numel(), x.numel())

        for i in range(n):
            x_i = x.clone()

            if method == 'forward':
                x_i[i] += h
                f1 = f(x_i,self.d, 0, log_tank)
                diff = (f1 - f0) / h
            elif method == 'backward':
                x_i[i] -= h
                f1 = f(x_i, self.d, 0, log_tank)
                diff = (f0 - f1) / h
            elif method == 'central':
                x_forward = x.clone(); x_forward[i] += h
                x_backward = x.clone(); x_backward[i] -= h
                f_forward = f(x_forward, self.d, 0, log_tank)
                f_backward = f(x_backward, self.d, 0, log_tank)
                diff = (f_forward - f_backward) / (2 * h)
            else:
                raise ValueError("Method must be 'forward', 'backward', or 'central'")

            if is_scalar:
                grad[i] = diff
            else:
                grad[:, i] = diff.view(-1)

        return grad.view(f0_shape + x.shape) if not is_scalar else grad.view(x.shape)


        



    def jac_gT(self, y):
        log_tank = opt_func.TanksOptimizationLog()

        y_np = y.detach().cpu().numpy()
        
        jac_gt_list = [ opt_func.jac_gT(i, self.d, 0, log_tank)  for i in y_np]

        jac_list_pos = torch.tensor(jac_gt_list, dtype=torch.float32, device=y.device)
        
        jac_list_neg = -jac_list_pos.clone()                

        jac_list = torch.cat((jac_list_pos, jac_list_neg), dim=1)
        
        
        return jac_list
    
    
    def numerical_jacobian(self,func, y, eps=1e-3):

        y = y.detach().clone()
        n = y.numel()
        y = y.view(-1)
        f0 = func(y)  # f(y)
        m = f0.numel()
        J = torch.zeros(m, n)

        for i in range(n):
            y_perturb = y.clone()
            y_perturb[i] += eps
            f1 = func(y_perturb)
            J[:, i] = (f1 - f0) / eps

        return J

    
    def jac_gt_batch(self, y_batch, data_sys, log_tank, opt_func):

        jacobianos = []
        

        for y_i in y_batch:
            #y_i = y_i.detach().clone().requires_grad_(True)

            
            J = self.numerical_jacobian(lambda y_:   opt_func.gT(y_, data_sys, 0, log_tank), y_i)
            
            
            
            jacobianos.append(J)
        return torch.stack(jacobianos)  # shape: (n, 10, 10)


        

    def jac_gT_v2(self, y):
        log_tank = opt_func.TanksOptimizationLog()

        #with torch.no_grad():
        #y_np = y.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        
        ############################
        #eps_aux = [opt_func.eps_definition_F3(i,self.d) for i in y_np]
        
        #jac_temp = [self.finite_difference(opt_func.gT, i) for i in y] 
        
        
        jac = self.jac_gt_batch(y, self.d, log_tank, opt_func)
        
        ############################
        
        #jac_gt_list = [opt_func.jac_gT(i, self.d, 0, log_tank) for i in y_np]
        #jac_gt_list_partial = partial(jac_gt_list)


        #jac_gt_list_new = ''


        jac_list_pos = jac
        
        jac_list_neg = -jac_list_pos.clone()                

        jac_list = torch.cat((jac_list_pos, jac_list_neg), dim=1)
        
        
        return jac_list

        
    def jac_TempLog(self, x):
        #with torch.no_grad():
        #x_np = x.detach().cpu().numpy()
        x_np = x
        jac_templog_list = torch.clamp(opt_func.jac_TempLog(x_np, self.d),0)
        
        jac_templog_list = torch.tensor(jac_templog_list, dtype=torch.float32, device=x.device)
        
        return jac_templog_list


    def ineq_dist(self, x, y):

        ineq_dist = self.ineq_resid(x, y)

        return ineq_dist

    def ineq_resid_v2(self, x, y):
        d = self.d
        n_min = 2  # Limite mínimo dos tanques
        n_max = 8  # Limite máximo dos tanques

        gT = self.gT(x, y)  # [n x 10]
        gt_up = gT - n_max  # [n x 10]
        gt_down = n_min - gT # [n x 10]


        
        # Tarifas
        duracao_tarifas = torch.tensor([2, 4, 1, 2, 3, 12], device=y.device)
        valores_tarifas = torch.tensor([0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094], device=y.device)

        inicio_tarifas = torch.cumsum(torch.cat([torch.tensor([0.0], device=y.device), duracao_tarifas[:-1]]), dim=0)
        fim_tarifas = inicio_tarifas + duracao_tarifas

        horarios = y[:, :5]  # [n x 5]
        duracoes = y[:, 5:]  # [n x 5]

        g_TempLog = torch.clamp(self.g_TempLog(y),0)

        # Penalização se a bomba for acionada em horários com tarifas mais caras
        tarifa_limite = 0.15  # Limite de tarifa acima do qual será penalizado
        penalidade_alta = 500.0  # Penalidade para acionamentos em horários caros

        restricao_tarifa_alta = []
        for i in range(5):
            h_ini = horarios[:, i]
            h_fim = h_ini + duracoes[:, i]

            penal = torch.zeros_like(h_ini)
            for j in range(len(valores_tarifas)):
                if valores_tarifas[j] >= tarifa_limite:  # Se a tarifa for maior que o limite
                    t_ini = inicio_tarifas[j]
                    t_fim = fim_tarifas[j]

                    # Verificando a sobreposição entre os horários da bomba e a tarifa
                    sobre_ini = torch.maximum(h_ini, t_ini)
                    sobre_fim = torch.minimum(h_fim, t_fim)
                    duracao_sobreposta = torch.clamp(sobre_fim - sobre_ini, min=0.0)

                    # Acumulando a penalidade pela sobreposição de tempo
                    penal += duracao_sobreposta * penalidade_alta

            restricao_tarifa_alta.append(penal.unsqueeze(1))

        restricao_tarifa_alta = torch.cat(restricao_tarifa_alta, dim=1)  # [n x 5]

        # Restrição de sobreposição (ajuste para 5 pares de bombas)
        sobreposicoes = torch.zeros(y.shape[0], 5, device=y.device)  # [n x 5]
        for i in range(5):
            h_i = horarios[:, i]
            f_i = h_i + duracoes[:, i]
            for j in range(i + 1, 5):
                h_j = horarios[:, j]
                f_j = h_j + duracoes[:, j]

                # Calculando a sobreposição entre os tempos de dois acionamentos
                ini_sobre = torch.maximum(h_i, h_j)
                fim_sobre = torch.minimum(f_i, f_j)
                overlap = torch.clamp(fim_sobre - ini_sobre, min=0.0)  # [n]

                # Se houver sobreposição, a restrição de sobreposição será 1
                sobreposicoes[:, i] = torch.where(overlap > 0.0, torch.tensor(1.0, device=y.device), sobreposicoes[:, i])

        # Ajuste para garantir que o total seja 25 restrições (somando 10 + 5 + 10)
        return torch.cat([gt_up, gt_down, g_TempLog, restricao_tarifa_alta, sobreposicoes], dim=1)


    # ineq_dist
    def ineq_resid(self, x, y):

        n_min = 2  # Limite mínimo dos tanques
        n_max = 8  # Limite máximo dos tanques

        gT = self.gT(x, y) # [samples x 10]

        # Constraint: [gt - Nmax] <= 0 [samples x 10]
        gt_up = gT - n_max

        # Constraint: [Nmin - gT] <= 0 [samples x 10]
        gt_down = n_min - gT

        # [samples x 5]
        g_TempLog = self.g_TempLog(y)

        # [samples x 25]
        return torch.cat([gt_up, gt_down, g_TempLog], dim=1)


    def jacobiano_sobreposicoes_v2(self, horarios, duracoes):
        n = horarios.shape[0]
        jac_sobreposicoes = torch.zeros((n, 10), device=horarios.device)

        for i in range(5):
            for j in range(i + 1, 5):
                h_i = horarios[:, i]
                h_j = horarios[:, j]
                d_i = duracoes[:, i]
                d_j = duracoes[:, j]
                f_i = h_i + d_i
                f_j = h_j + d_j

                sobre_ini = torch.maximum(h_i, h_j)
                sobre_fim = torch.minimum(f_i, f_j)
                overlap = torch.clamp(sobre_fim - sobre_ini, min=0.0)

                mask = (overlap > 0).float()  # [n]

                # Derivadas parciais (negativo onde aumento do valor causa mais sobreposição)
                d_overlap_d_hi = -mask * ((h_i >= h_j).float())  # deslocar h_i para trás reduz sobreposição
                d_overlap_d_hj = -mask * ((h_j >= h_i).float())
                d_overlap_d_di = mask * ((f_i <= f_j).float())
                d_overlap_d_dj = mask * ((f_j <= f_i).float())

                # Acumular nas posições corretas do jacobiano [n, 10]
                jac_sobreposicoes[:, i]     += d_overlap_d_hi
                jac_sobreposicoes[:, j]     += d_overlap_d_hj
                jac_sobreposicoes[:, 5 + i] += d_overlap_d_di
                jac_sobreposicoes[:, 5 + j] += d_overlap_d_dj

        return jac_sobreposicoes


    def jacobiano_sobreposicoes(self,horarios, duracoes):
        
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

    def jac_tarifas(self, y):
        # Tarifas e estrutura de tempo
        tarifa_limite = 0.15  # Só penaliza tarifas acima disso

        duracao_tarifas = torch.tensor([2, 4, 1, 2, 3, 12], device=y.device)
        valores_tarifas = torch.tensor([0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094], device=y.device)

        inicio_tarifas = torch.cumsum(torch.cat([torch.tensor([0.0], device=y.device), duracao_tarifas[:-1]]), dim=0)
        fim_tarifas = inicio_tarifas + duracao_tarifas

        horarios = y[:, :5]   # [n x 5] horários de início
        duracoes = y[:, 5:]   # [n x 5] durações dos acionamentos

        n = horarios.shape[0]
        jac_tarifas = torch.zeros(n, 10, device=y.device)  # [n x 10] ← gradiente w.r.t. y

        for i in range(5):  # para cada bomba
            h_ini = horarios[:, i]  # [n]
            dur = duracoes[:, i]
            h_fim = h_ini + dur

            for j in range(len(valores_tarifas)):  # para cada faixa de tarifa
                tarifa = valores_tarifas[j]
                if tarifa < tarifa_limite:
                    continue  # pula tarifas baratas

                t_ini = inicio_tarifas[j]
                t_fim = fim_tarifas[j]

                sobre_ini = torch.maximum(h_ini, t_ini)
                sobre_fim = torch.minimum(h_fim, t_fim)
                duracao_sobreposta = torch.clamp(sobre_fim - sobre_ini, min=0.0)  # [n]

                penal = duracao_sobreposta * tarifa  # custo de acionamento em horário caro

                # Derivadas:
                mask = (duracao_sobreposta > 0).float()  # [n]

                # ∂penal/∂h_ini = -tarifa se h_ini < t_fim e h_ini > t_ini
                d_penal_d_hi = -tarifa * mask * ((h_ini >= t_ini) & (h_ini <= t_fim)).float()

                # ∂penal/∂dur = tarifa se h_fim < t_fim e h_fim > t_ini
                d_penal_d_dur = tarifa * mask * ((h_fim >= t_ini) & (h_fim <= t_fim)).float()

                # Acumula no jacobiano
                jac_tarifas[:, i] += d_penal_d_hi
                jac_tarifas[:, 5 + i] += d_penal_d_dur

        return jac_tarifas


    def jac_tarifas_old(self, y):
        # Tarifas
        duracao_tarifas = torch.tensor([2, 4, 1, 2, 3, 12], device=y.device)
        valores_tarifas = torch.tensor([0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094], device=y.device)

        inicio_tarifas = torch.cumsum(torch.cat([torch.tensor([0.0], device=y.device), duracao_tarifas[:-1]]), dim=0)
        fim_tarifas = inicio_tarifas + duracao_tarifas

        horarios = y[:, :5]   # [n x 5] (horários de início)
        duracoes = y[:, 5:]   # [n x 5] (durações dos acionamentos)

        # Inicializando o Jacobiano de tarifas (inicialmente zero)
        jac_tarifas = torch.zeros(horarios.shape[0], horarios.shape[1] * 2, device=y.device)  # [n x 10] (horarios + duracoes)

        # Calculando a penalização das tarifas para cada bomba
        for i in range(5):
            h_ini = horarios[:, i]  # [n]
            h_fim = h_ini + duracoes[:, i]

            for j in range(len(valores_tarifas)):
                t_ini = inicio_tarifas[j]
                t_fim = fim_tarifas[j]
                tarifa = valores_tarifas[j]

                sobre_ini = torch.maximum(h_ini, t_ini)
                sobre_fim = torch.minimum(h_fim, t_fim)
                duracao_sobreposta = torch.clamp(sobre_fim - sobre_ini, min=0.0)

                # Calculando a penalização de tarifa
                penal = duracao_sobreposta * tarifa

                # Agora, vamos calcular o Jacobiano da penalização em relação a cada variável (horário e duração)
                # Jacobiano para o horário de início
                jac_ini = torch.zeros_like(h_ini)
                jac_ini += (sobre_fim - sobre_ini) * tarifa

                # Jacobiano para a duração
                jac_dur = torch.zeros_like(h_ini)
                jac_dur += (sobre_fim - sobre_ini) * tarifa  # O cálculo correto aqui

                # Preenchendo o Jacobiano de tarifas com os valores calculados
                jac_tarifas[:, i] += jac_ini  # Jacobiano para horários
                jac_tarifas[:, 5 + i] += jac_dur  # Jacobiano para durações

        return jac_tarifas





    def ineq_jac_v2(self, Y):
        # Jacobiano de gT (restrições de nível)
        jac_gT = self.jac_gT(Y)  # [n x 20 x 10]

        # Jacobiano de TempLog (restrições de temperatura)
        jac_TempLog = self.jac_TempLog(Y)  # [5 x 10]

        # Jacobiano das tarifas (restrições de tarifas)
        jac_tarifas = self.jac_tarifas(Y)  # [n x 10]

        # Ajustar jac_TempLog para que tenha a forma [n x 5 x 10]
        # Repetir jac_TempLog para cada amostra em n
        jac_TempLog = jac_TempLog.unsqueeze(0).repeat(jac_gT.shape[0], 1, 1)  # [n x 5 x 10]

        # Ajustar jac_tarifas para que tenha a forma [n x 5 x 10]
        # Repetir jac_tarifas para que tenha 5 repetições ao longo da segunda dimensão (dim=1)
        jac_tarifas = jac_tarifas.unsqueeze(1).repeat(1, 5, 1)  # [n x 5 x 10]
        
        # Jacobiano das sobreposições
        jac_sobreposicoes = self.jacobiano_sobreposicoes(Y[:, :5], Y[:, 5:])  # [n x 5 x 5]
        jac_sobreposicoes = jac_sobreposicoes.repeat(1, 1, 2) 
        # Concatenando ao longo da dimensão 1 (a segunda dimensão das matrizes)
        jac_combined = torch.cat([jac_gT, jac_TempLog, jac_tarifas, jac_sobreposicoes], dim=1)  # [n x 25 x 10]

        return jac_combined


        
    def ineq_jac(self, Y):

        # [samples x 20 x 10]
        jac_gT = self.jac_gT(Y)

        # [5 x 10]
        jac_TempLog = self.jac_TempLog(Y)

        # [samples x 5 x 10]
        jac_combined = torch.cat([jac_gT, jac_TempLog.unsqueeze(0).repeat(jac_gT.shape[0], 1, 1)], dim=1)

        return jac_combined

    def ineq_grad_v2(self, x, y):
        # [samples x 25] ineq_resid = ineq_dist
        ineq_dist_relu = torch.clamp(self.ineq_dist(x, y),0)
        
        # [samples x 1 x 25]
        ineq_dist_expanded = ineq_dist_relu.unsqueeze(1).type(torch.float32)  

        # [samples X 25 X 10]
        ineq_jac = self.ineq_jac(y).type(torch.float32)
        
        
        
        return torch.matmul(ineq_dist_expanded, ineq_jac).squeeze(1)
    

    def ineq_grad(self, x, y):
        # [samples x 25]
        ineq_resid = self.ineq_dist(x, y)

        # Máscara booleana onde há violação (valor > 0)
        mask = (ineq_resid > 0).float()  # [samples x 25]

        # Jacobiano das restrições (samples x 25 x 10)
        ineq_jac = self.ineq_jac(y)  # [batch x n_constraints x n_vars]

        # Aplica a máscara no jacobiano: zera os gradientes onde não há violação
        masked_jac = ineq_jac * mask.unsqueeze(2)  # Broadcasting

        # Produto do resíduo clamped com o jacobiano mascarado
        grad = torch.matmul(mask.unsqueeze(1), masked_jac).squeeze(1)  # [batch x 10]

        return grad
    

    def process_output(self, X, out):
        qty = out.shape[1] // 2

        start_times = out[:, :qty] * 23.9
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
        over_limit = end_times > 23.9
        durations_sorted = torch.where(over_limit, 23.9 - adjusted_start_times, durations_sorted)

        # Reaplica os clamps
        adjusted_start_times = torch.clamp(adjusted_start_times, 0, 23.9)
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
