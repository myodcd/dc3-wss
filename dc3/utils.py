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
        
        y = y.detach().cpu().numpy()

        cost_list = [opt_func.Cost(i, self.d, log_cost, 3) for i in y]

        return torch.tensor(cost_list, requires_grad=True)

    def gT(self, x, y):
        log_tank = opt_func.TanksOptimizationLog()
        
        y = y.detach().cpu().numpy()

        gt_list = [opt_func.gT(i, self.d, 0, log_tank) for i in y]

        return torch.tensor(gt_list)

    def g_TempLog(self, x):  # tstart(n+1) > tstop(n)  (várias bombas)
        # print('Temporal Logic Const. --> x(start-stop)')

        g_templog_list = [opt_func.g_TempLog(i, self.d) for i in x]

        return torch.tensor(g_templog_list)

    def jac_gT(self, y):

        log_tank = opt_func.TanksOptimizationLog()

        y = y.detach().cpu().numpy()

        jac_gt_list = [opt_func.jac_gT(i, self.d, 0, log_tank) for i in y]
        
        jac_list_pos = torch.tensor(jac_gt_list)
        jac_list_neg = -jac_list_pos.clone()
        
        return torch.cat((jac_list_pos, jac_list_neg), dim=1)
    
    def jac_TempLog(self, x):

        jac_templog_list = opt_func.jac_TempLog(x, self.d)

        return torch.tensor(jac_templog_list)

    def ineq_dist(self, x, y):

        ineq_dist = self.ineq_resid(x, y)

        return ineq_dist

    # ineq_dist
    def ineq_resid(self, x, y):

        d = self.d
        n_min = d.hmin[0]  # 2
        n_max = d.hmax[0]  # 8

        gT = self.gT(x, y)


        #gt_ineq1 = torch.clamp(n_min - gT, min=0)

        # Parte que viola superior: quanto gT está acima de n_max (zero se não estiver)
        #gt_ineq2 = torch.clamp(gT - n_max, min=0)
        
        
        # Constraint: [gt - Nmax] <= 0 [samples x 10]
        gt_ineq_up = gT - n_max

        # Constraint: [Nmin - gT] <= 0 [samples x 10]
        gt_ineq_down = n_min - gT

        # [samples x 5]
        g_TempLog = self.g_TempLog(y)

        # [samples x 25]
        return torch.cat([gt_ineq_up, gt_ineq_down, g_TempLog], dim=1)

    def ineq_jac(self, Y):

        # [samples x 20 x 10]
        jac_gT = self.jac_gT(Y)

        # [5 x 10]
        jac_TempLog = self.jac_TempLog(Y)

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
    
    def process_output_old(self, x, out):
        qty = out.shape[1] // 2
        start_times = out[:, :qty] * 23.9
        durations = out[:, qty:] * (5.0 - 0.1) + 0.1
        schedules, sort_indices = torch.sort(start_times, dim=1)
        return torch.cat([schedules, durations], dim=1)
        
    def process_output(self, X, out):
        qty = out.shape[1] // 2
        # Escala os outputs:
        # - Os 5 primeiros para o intervalo [0, 23.9] (início de funcionamento)
        # - Os 5 últimos para o intervalo [0.1, 5.0] (duração)
        start_times = out[:, :qty] * 23.8
        durations = out[:, qty:] * (5.0 - 0.1) + 0.1

        # Garante que (início + duração) não ultrapasse 23.8 inicialmente
        end_times = start_times + durations
        over_limit = end_times > 23.8
        durations[over_limit] = 23.8 - start_times[over_limit]

        # Aplica clamp para garantir os limites
        start_times = torch.clamp(start_times, 0, 23.8)
        durations = torch.clamp(durations, 0.1, 5)

        # Ordena os horários de cada amostra e reordena as durações de acordo
        schedules, sort_indices = torch.sort(start_times, dim=1)
        durations = torch.gather(durations, 1, sort_indices)

        margin = 0.1
        # Para cada amostra, verifica sobreposição entre intervalos
        for j in range(schedules.shape[0]):
            for i in range(qty - 1):
                # Se o fim do intervalo atual (início + duração) ultrapassar o início da próxima janela...
                if schedules[j, i] + durations[j, i] > schedules[j, i+1]:
                    new_start = schedules[j, i] + durations[j, i] + margin
                    # Ajusta o início da próxima, garantindo não ultrapassar 23.8
                    schedules[j, i+1] = min(new_start, 23.8)
            # Para o último intervalo, garante que (início + duração) não ultrapasse 24 horas
            if schedules[j, -1] + durations[j, -1] > 24:
                durations[j, -1] = 24 - schedules[j, -1] - 0.001

        return torch.cat([schedules, durations], dim=1)

    def process_output_las_last_used(self, X, out):
        # Aplica sigmoide para limitar os valores entre 0 e 1
        #out2 = torch.sigmoid(out)

        # Escala os 5 primeiros valores para o intervalo [0, 23.9] (início de funcionamento)
        start_times = out[:, :5] * 23.8

        # Escala os 5 últimos para o intervalo [0.1, 5.0] (duração)
        durations = out[:, 5:] * (5.0 - 0.1) + 0.1

        # Garante que (início + duração) não ultrapassa 23.9
        end_times = start_times + durations
        over_limit = end_times > 23.8
        durations[over_limit] = 23.8 - start_times[over_limit]

        # Ordena os intervalos para facilitar o controle de sobreposição
        sorted_indices = torch.argsort(start_times, dim=1)
        sorted_starts = torch.gather(start_times, 1, sorted_indices)
        sorted_durations = torch.gather(durations, 1, sorted_indices)
        sorted_ends = sorted_starts + sorted_durations

        # Corrige sobreposição (empurra a próxima bomba para depois do fim da anterior)
        for i in range(1, 5):
            prev_end = sorted_ends[:, i - 1]
            this_start = sorted_starts[:, i]
            shift_needed = (prev_end > this_start)
            # Se houver sobreposição, move o início da bomba atual
            sorted_starts[:, i] = torch.where(shift_needed, prev_end, this_start)
            # Recalcula os tempos de término
            sorted_ends[:, i] = sorted_starts[:, i] + sorted_durations[:, i]
            # Se exceder 23.9, ajusta duração
            over_limit = sorted_ends[:, i] > 23.9
            sorted_durations[:, i] = torch.where(over_limit, 23.8 - sorted_starts[:, i], sorted_durations[:, i])
            sorted_ends[:, i] = sorted_starts[:, i] + sorted_durations[:, i]

        # Retorna início e duração ajustados (você pode concatenar ou retornar separadamente)
        return torch.cat([sorted_starts, sorted_durations], dim=1)


    def process_output_last_used(self, x, out):
        qty = out.shape[1] // 2
        # Escalar: horários em [0,24] e durações em [0,6]
        schedules = out[:, :qty] * 24
        durations = out[:, qty:] * 6

        # Aplica clamp: horários em [0, 23.9] e durações em [0.1, 5]
        schedules = torch.clamp(schedules, 0, 23.9)
        durations = torch.clamp(durations, 0.1, 5)
        # Ordena os horários por amostra
        schedules, _ = torch.sort(schedules, dim=1)

        # Para cada amostra, aplica os ajustes:
        for j in range(schedules.shape[0]):
            # 1. Garante que os horários tenham diferença mínima de 2 (como na generate_time)
            for i in range(1, qty):
                if schedules[j, i] < schedules[j, i-1] + 2:
                    schedules[j, i] = schedules[j, i-1] + 2

            # 2. Ajusta cada duração para que o intervalo não ultrapasse o início da próxima janela
            for i in range(qty - 1):
                # O máximo permitido é a diferença entre o horário atual e o próximo, menos uma margem (0.1)
                max_dur = schedules[j, i+1] - schedules[j, i] - 0.1
                if max_dur < 0.1:
                    max_dur = 0.1
                if durations[j, i] > max_dur:
                    durations[j, i] = max_dur

            # 3. Para o último horário, garante que (horário + duração) não ultrapasse 24
            if schedules[j, -1] + durations[j, -1] > 24:
                durations[j, -1] = 24 - schedules[j, -1] - 0.001

        return torch.cat([schedules, durations], dim=1)        
        

    def process_output_old2(self, x, out):
        qty = out.shape[1] // 2
        schedules = out[:, :qty] * 24
        durations = out[:, qty:] * 6
        
        schedules = torch.clamp(schedules, 0, 23.9)
        schedules, _ = torch.sort(schedules, dim=1)
        
        for i in range(qty):
            if i > 0:
                inicio = schedules[0, i - 1]
                proximo = schedules[0, i]
                duration = durations[0, i - 1]
                
                if inicio + duration >= proximo:
                    schedules[0, i] = inicio + duration + 0.1
                    schedules[0, i] = min(schedules[0, i], 23.9)  # Garantir que não ultrapasse 23.9999

            if i == qty - 1:  # Garantir que a soma do último horário com a duração não ultrapasse 24 horas
                if schedules[0, i] + durations[0, i] > 23.9:
                    durations[0, i] = 23.9 - schedules[0, i]
                    
        
        for j in range(schedules.shape[0]):
            if schedules[j, -1] + durations[j, -1] > 24:
                schedules[j, -1] = 24 - durations[j, -1] - 0.001
        
        #print(schedules)
        #print(durations)
        return torch.cat([schedules, durations], dim=1)
        
        
    def process_output_old(self, X, out):
        # Número de janelas: assume que a 1ª metade de out são horários e a 2ª durações
        qty = out.shape[1] // 2

        # Escalar as saídas: horários para [0,24] e durações para [0,6]
        schedules = out[:, :qty] * 24  
        durations = out[:, qty:] * 6 

        # Garantir que os horários estejam em [0, 23]
        schedules = torch.clamp(schedules, 0, 23)
        # Ordena os horários para garantir ordem crescente (por amostra)
        schedules, _ = torch.sort(schedules, dim=1)

        # Impõe espaçamento mínimo entre os horários
        for i in range(1, qty):
            min_spacing = 2.0 + torch.rand_like(schedules[:, i]) * 0.5
            schedules[:, i] = torch.where(
                schedules[:, i] >= schedules[:, i - 1] + min_spacing,
                schedules[:, i],
                schedules[:, i - 1] + min_spacing
            )
        schedules = torch.clamp(schedules, 0, 23)
        
        # Restrição: o primeiro horário não deve estar entre 20 e 23
        schedules[:, 0] = torch.where(schedules[:, 0] >= 20,
                                    torch.tensor(19.0, device=schedules.device),
                                    schedules[:, 0])
        
        # Calcula o máximo de duração permitido para cada janela para não ultrapassar 24 horas
        max_durations = torch.zeros_like(durations)
        for i in range(qty):
            if i == 0:
                max_durations[:, i] = torch.where(
                    schedules[:, 0] >= 23,
                    torch.tensor(1.0, device=durations.device),
                    schedules[:, 1] - schedules[:, 0]
                )
            elif i == qty - 1:
                max_durations[:, i] = torch.where(
                    schedules[:, qty - 1] >= 23,
                    torch.tensor(1.0, device=durations.device),
                    24.0 - schedules[:, qty - 1]
                )
            else:
                max_durations[:, i] = torch.clamp(schedules[:, i + 1] - schedules[:, i],
                                                min=0.1, max=5.0)
        
        # Garante que as durações estejam no intervalo [0.1, 5] e não excedam os máximos
        durations = torch.clamp(durations, 0.1, 5)
        durations = torch.min(durations, max_durations)
        
        # Ajusta, se necessário, as durações para que 'horário + duração' não ultrapasse 24
        for i in range(qty):
            while torch.any(schedules[:, i] + durations[:, i] > 24):
                durations[:, i] = torch.clamp(durations[:, i] - 0.001, min=0.1)
        
        # Verifica sobreposição dos intervalos e, se necessário, ajusta o início da janela seguinte
        # de forma que (horário atual + duração) <= (próximo horário).
        # O ajuste (acréscimo de 0.1) é aplicado somente se, após o aumento, a soma com a duração
        # da próxima janela não ultrapassar 24 horas.
        for j in range(schedules.shape[0]):  # para cada amostra
            for i in range(qty - 1):
                end_time = schedules[j, i] + durations[j, i]
                if end_time - schedules[j, i+1] > 1e-6:
                    new_val = schedules[j, i+1] + 0.1
                    # Ajusta para não ultrapassar o limite de 24 horas (considerando a duração atual)
                    if new_val + durations[j, i+1] <= 24:
                        schedules[j, i+1] = new_val
            # Reordena os horários para manter a ordem crescente
            schedules[j, :] = torch.sort(schedules[j, :])[0]
        
        return torch.cat([schedules, durations], dim=1)    

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
