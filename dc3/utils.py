import torch
import torch.nn as nn

import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
from scipy.optimize import approx_fprime
import math
import plot_simple as plot_simple
import numpy as np

import datetime

import hashlib
import scipy.io as spio
import time
import EPANET_API as EPA_API

import OptimAuxFunctionsV2 as opt_func

from functools import partial

from torch.autograd import Function
import ClassAutogradPyTorch as autograd_pt
from torch.autograd.functional import jacobian

import utils as utils
import imageio
import os
import shutil

import time
import re
from PIL import Image

import tempfile


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

def prepare_inp_copies(base_inp_path, n_jobs):

    for i in range(n_jobs):
        job_dir = f"/tmp/epanet_job_{i}"
        os.makedirs(job_dir, exist_ok=True)
        shutil.copy(base_inp_path, os.path.join(job_dir, "Bomba-deposito_v1"))

def create_gif_from_plots(nr: int, base_folder: str, output_folder: str, duracao: int = 1000):
    padrao = re.compile(rf"plot_Y_sample_nr{nr}_iteration_.*\.png")
    arquivos = [f for f in os.listdir(base_folder) if padrao.fullmatch(f)]
    if not arquivos:
        print(f"Nenhuma imagem encontrada para nr{nr}")
        return

    def extrai_indice(nome):
        match = re.search(r"iteration_nr(\d+)", nome)
        return int(match.group(1)) if match else 0

    arquivos.sort(key=extrai_indice)
    caminhos = [os.path.join(base_folder, nome) for nome in arquivos]
    imagens = [Image.open(nome) for nome in caminhos]

    output = os.path.join(output_folder, f"gif_tanks_nr{nr}.gif")
    imagens[0].save(
        output,
        save_all=True,
        append_images=imagens[1:],
        duration=duracao,
        loop=0
    )



def create_combined_gif(Y_shape_0, base_folder: str, output_folder: str, duracao: int = 1000):
    imagens_por_nr = []

    for nr in range(Y_shape_0):
        padrao = re.compile(rf"plot_Y_sample_nr{nr}_iteration_.*\.png")
        arquivos = [f for f in os.listdir(base_folder) if padrao.fullmatch(f)]

        def extrai_indice(nome):
            match = re.search(r"iteration_nr(\d+)", nome)
            return int(match.group(1)) if match else 0

        arquivos.sort(key=extrai_indice)
        caminhos = [os.path.join(base_folder, nome) for nome in arquivos]
        imagens = [Image.open(caminho) for caminho in caminhos]
        imagens_por_nr.append(imagens)

    num_quadros = min(len(imgs) for imgs in imagens_por_nr)
    quadros_comb = []

    for i in range(num_quadros):
        imagens_lado_a_lado = [imagens_por_nr[nr][i] for nr in range(Y_shape_0)]
        altura_comum = min(img.height for img in imagens_lado_a_lado)
        imagens_redim = [img.resize((int(img.width * altura_comum / img.height), altura_comum)) for img in imagens_lado_a_lado]

        largura_total = sum(img.width for img in imagens_redim)
        combinado = Image.new('RGB', (largura_total, altura_comum))
        x_offset = 0
        for img in imagens_redim:
            combinado.paste(img, (x_offset, 0))
            x_offset += img.width

        quadros_comb.append(combinado)

    if quadros_comb:
        output = os.path.join(output_folder, f"gif_combined_all_nr.gif")
        quadros_comb[0].save(
            output,
            save_all=True,
            append_images=quadros_comb[1:],
            duration=duracao,
            loop=0
        )




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


    def obj_fn_Original(self, y, args):
        
        log_cost = opt_func.OptimizationLog()
        
        y = y.detach().cpu().numpy()
        
        result = torch.tensor([opt_func.Cost(i, self.d, log_cost, 3) for i in y])
        
        return result
    
    
    def obj_fn_Autograd(self, y, args):
        
        #start_time = time.time()
                
        #log_cost = opt_func.OptimizationLog()
        
        #y = y.detach().cpu().numpy()
        
        #result = torch.tensor([opt_func.Cost(i, self.d, log_cost, 3) for i in y])

        result = autograd_pt.CostAutograd.apply(y, self.d, opt_func)
        
        #print('COST_AUTOGRAD', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
        
        #print('%%%%%%%%%')
        
        return result
    
    def gT_Original(self, y):
        log_cost = opt_func.OptimizationLog()
        y_t = y  # preserva dtype/device
        y_np = y_t.detach().cpu().numpy()
        gt_list = []
        for i in y_np:
            gt_list.append(opt_func.gT(i, self.d, 0, log_cost))
        gt = torch.tensor(gt_list, dtype=y_t.dtype, device=y_t.device)
        return gt
    
    def gT_Autograd(self, y, args):

        result = autograd_pt.GTAutograd.apply(y, self.d, opt_func)

        return result


    def g_TempLog_Original(self, y):
        
     
        y = y.detach().cpu().numpy()
        
        g_templog_list = torch.tensor([-opt_func.g_TempLog(i, self.d) for i in y])

        return g_templog_list

    def g_TempLog_Autograd(self, y):

        g_templog_list = autograd_pt.GTempLogAutograd.apply(y, self.d, opt_func)

        return g_templog_list


    
    def g_x_Autograd(self, y, args, numpy=False):

        return y

    def ineq_dist(self, x, y, args):

        ineq_dist = torch.relu(self.ineq_resid(x, y, args))
        
        return ineq_dist

    def get_yvars(self, y, args):

        #g_tl = self.gT_Autograd(y, args)
        g_tl = self.gT_Original(y)
        g_tlog = self.g_TempLog_Original(y)
        #g_tlog = self.g_TempLog_Autograd(y)
        g_x = self.g_x_Autograd(y, args)

        return g_tl, g_x, g_tlog

    # ineq_dist
    def ineq_resid(self, x, y, args):

        gt, g_x, g_tlog = self.get_yvars(y, args)

        gt_max = gt - self.hmax
        gt_min = self.hmin - gt

        g_x_max = g_x - self.timemax
        g_x_min = self.timemin - g_x

        resids = torch.cat([gt_max, gt_min, g_x_max, g_x_min, g_tlog], dim=1)


        #for i in range(y.shape[0]):
            
        #    plot_simple.plot_simple(y[i].cpu().detach().numpy(), 000, args, 0, #n_sample=1,title_comment=f'gt_max: {gt_max[i]} \n gt_min: {gt_min[i]}')
                                
            



        return resids

    def jac_gT_Original(self, y):

        #start_time = time.time()

        log_cost = opt_func.OptimizationLog()
        
        y = y.detach().cpu().numpy()
        
        jac = torch.tensor([ opt_func.jac_gT(i, self.d, 0, log_cost)  for i in y])

        jac_pos = jac
        jac_inv = -jac.clone()

        result = torch.cat([jac_pos, jac_inv], dim=1)  # [batch, 20, 10]
        
        #print('JAC_GT_ORIGINAL', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

        return result  # [batch, 20, 10]


    def jac_gT_Autograd(self, y, args):
        
        jac = autograd_pt.JacGTAutograd.apply(y, self.d, opt_func)

        jac_pos = jac
        jac_inv = -jac.clone()

        result = torch.cat([jac_pos, jac_inv], dim=1)  # [batch, 20, 10]

        return result  # [batch, 20, 10]

    
    def jac_TempLog_Autograd(self, y):

        result = autograd_pt.JacTempLogAutograd.apply(y, self.d, opt_func)

        return result
    
    def jac_TempLog_Original(self, y):
        
        #start_time = time.time()
        
        y = y.detach().cpu().numpy()
        
        result = torch.tensor([opt_func.jac_TempLog(i, self.d) for i in y])

        #print('JAC_TEMPLOG_ORIGINAL', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

        return result
    
    def jac_x_Autograd(self, y, args):

        #start_time = time.time()

        y = y.clone().requires_grad_(True)  # [B, D]
        B, D = y.shape

        def g_vec(y_i):

            out = self.g_x_Autograd(y_i.unsqueeze(0), args)
            return out.squeeze(0)  # [C]

        # 3) loop sobre o batch (pode ser lento, mas preserva grafo)
        jac_list = []
        for b in range(B):
            # jacobian retorna tensor [C, D]
            Jb = jacobian(g_vec, y[b], create_graph=True, strict=False)
            jac_list.append(Jb)

        J_pos = torch.stack(jac_list, dim=0)
        J_neg = -J_pos        

        result = torch.cat([J_pos, J_neg], dim=1)

        return result

    
    def ineq_jac(self, y, args):

        #jac_gT = self.jac_gT_Autograd(y, args)
        
        jac_gT = self.jac_gT_Original(y)
        
        #jac_TempLog = self.jac_TempLog_Autograd(y) 

        jac_TempLog = -self.jac_TempLog_Original(y)

        jac_x = self.jac_x_Autograd(y, args) 

        
        result = torch.cat([jac_gT, jac_x, jac_TempLog], dim=1)


        return result

    def ineq_grad(self, x, y, args):
        ineq_dist = self.ineq_dist(x, y, args)
        ineq_jac = self.ineq_jac(y, args)  # [batch x n_constraints x n_vars]

        return ineq_jac.transpose(1, 2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)



    def process_output_original(self, X, out):
        out2 = nn.Sigmoid()(out)  # Normaliza a saída para [0, 1]

        qty = out2.shape[1] // 2  # Quantidade de duty cycles

        # Escala os tempos normalizados para [0, 23.8] e durações para [0.1, 5.0]
        start_times = out2[:, :qty] * 23.9
        durations = out2[:, qty:] * (5.0 - 0.001) + 0.001

        # Ordena os horários de início e ajusta as durações na mesma ordem
        start_times_sorted, sorted_idx = torch.sort(start_times, dim=1)
        durations_sorted = durations.gather(1, sorted_idx)

        # Ajusta os horários de início para evitar sobreposição
        adjusted_start_times = [start_times_sorted[:, 0:1]]
        for i in range(1, qty):
            prev_end = adjusted_start_times[i - 1] + durations_sorted[:, i - 1 : i]
            current_start = torch.max(
                start_times_sorted[:, i : i + 1], prev_end + 0.001
            )  # Adiciona intervalo mínimo de 0.05
            adjusted_start_times.append(current_start)

        adjusted_start_times = torch.cat(adjusted_start_times, dim=1)

        # Garante que nenhum ciclo ultrapasse o final do dia
        max_end = adjusted_start_times + durations_sorted
        durations_sorted = torch.where(
            max_end > 23.9, 23.9 - adjusted_start_times, durations_sorted
        )

        # Trunca valores finais para garantir domínio válido
        #adjusted_start_times = torch.clamp(adjusted_start_times, 0, 23.8)
        #durations_sorted = torch.clamp(durations_sorted, 0.001, 5.0)

        return torch.cat([adjusted_start_times, durations_sorted], dim=1)


# ...existing code...
    def process_output_original_parametric1(self, X, out):
        """
        Versão diferenciável sem uso de sort/max:
        - Garante ordem via somas cumulativas de gaps positivos.
        - Durações em [0.001, 5.0].
        - Ajusta (comprime) inícios se último término exceder 23.9 sem alterar durações.
        """
        DAY_MAX = 23.9
        MIN_DUR = 0.001
        MAX_DUR = 5.0
        MIN_GAP = 0.001

        B, D = out.shape
        qty = D // 2
        raw_gaps = out[:, :qty]
        raw_durs = out[:, qty:]

        # Gaps positivos (softplus) + mínimo
        gaps = F.softplus(raw_gaps) + MIN_GAP                # [B, qty]

        # Durações com limite superior (sigmoid) + mínimo
        durations = torch.sigmoid(raw_durs) * (MAX_DUR - MIN_DUR) + MIN_DUR  # [B, qty]

        # Inícios cumulativos
        start_times = torch.cumsum(gaps, dim=1)  # crescente

        # Ajuste se extrapolar o dia (escala só os inícios)
        last_start = start_times[:, -1]
        last_end = last_start + durations[:, -1]
        # fator que garante last_start_scaled + last_duration <= DAY_MAX
        alpha = (DAY_MAX - durations[:, -1]) / (last_start + 1e-6)
        alpha = torch.clamp(alpha, max=1.0).unsqueeze(1)  # não expande se já dentro
        start_times = start_times * alpha  # comprime intervalos mantendo proporções

        return torch.cat([start_times, durations], dim=1)
# ...existing code...


    def process_output_original_parametric2(self, X, out):
        """
        Parametrização suave diferenciável sem sort/max:
        Estrutura do vetor out (dim = 2 * qty):
            - Primeira metade (qty):
                * índice 0: parâmetro para primeiro início (>=0)
                * índices 1..qty-1: parâmetros para gaps após cada ciclo (entre o fim do anterior e o início do próximo)
            - Segunda metade (qty): parâmetros de duração
        Construção:
            start_0 = softplus(raw_start_0)
            gaps_end_i = softplus(raw_gap_i) + MIN_GAP
            durations_i = sigmoid(raw_dur_i) * (MAX_DUR - MIN_DUR) + MIN_DUR
            increments_i = durations_{i} + gaps_end_{i+1}  (para i = 0..qty-2)
            start_k = start_0 + cumsum(increments_0..k-1)
        Garantias:
            - Ordem crescente dos inícios
            - Não sobreposição (cada incremento inclui duração anterior + gap)
            - Compressão proporcional se último término > DAY_MAX (mantém não sobreposição)
        """
        DAY_MAX = 23.9
        MIN_DUR = 0.001
        MAX_DUR = 5.0
        MIN_GAP = 0.001

        B, D = out.shape
        qty = D // 2
        raw_pos = out[:, :qty]          # primeiro início + (qty-1) gaps-end
        raw_dur = out[:, qty:]          # durações

        # Durações em faixa
        durations = torch.sigmoid(raw_dur) * (MAX_DUR - MIN_DUR) + MIN_DUR  # [B, qty]

        # Primeiro início (>=0) e gaps (>= MIN_GAP)
        first_start = F.softplus(raw_pos[:, 0:1])                           # [B,1]
        gaps_end = F.softplus(raw_pos[:, 1:]) + MIN_GAP if qty > 1 else raw_pos.new_zeros((B,0))

        if qty == 1:
            start_times = first_start  # só um ciclo
        else:
            # increments = duração_i + gap_end_i  (para i=0..qty-2)
            increments = durations[:, :-1] + gaps_end  # [B, qty-1]
            cum_increments = torch.cumsum(increments, dim=1)                # [B, qty-1]
            # start_0 já é first_start
            start_rest = first_start + cum_increments                       # [B, qty-1]
            start_times = torch.cat([first_start, start_rest], dim=1)       # [B, qty]

        # Último término
        last_end = start_times[:, -1] + durations[:, -1]
        overflow = last_end > DAY_MAX

        if overflow.any():
            # Total span sem incluir duração final: span = start_last
            # Precisamos escalar first_start e increments para caber antes do término final
            # total_span = start_last + duration_last
            start_last = start_times[:, -1]
            total_span = start_last + durations[:, -1]
            scale = (DAY_MAX - durations[:, -1]) / (start_last + 1e-8)      # [B]
            scale = torch.where(overflow, torch.clamp(scale, max=1.0), torch.ones_like(scale))
            scale = scale.unsqueeze(1)

            # Reconstroi starts escalando first_start e increments proporcionalmente
            first_start_scaled = first_start * scale
            if qty > 1:
                increments_scaled = increments * scale
                cum_increments_scaled = torch.cumsum(increments_scaled, dim=1)
                start_rest_scaled = first_start_scaled + cum_increments_scaled
                start_times = torch.cat([first_start_scaled, start_rest_scaled], dim=1)
            else:
                start_times = first_start_scaled

        return torch.cat([start_times, durations], dim=1)
# ...existing code...



# ...existing code...
    def process_output_original_parametric3(self, X, out):
        """
        Parametrização suave diferenciável sem sort/max:
        - Durações sem teto superior (>= MIN_DUR).
        - Ordem e não sobreposição garantidas por gaps cumulativos.
        - Se ultrapassar DAY_MAX, escala inícios e durações proporcionalmente.
        """
        DAY_MAX = 23.9
        MIN_DUR = 0.001
        MIN_GAP = 0.001

        B, D = out.shape
        qty = D // 2
        raw_pos = out[:, :qty]   # [B, qty] -> first_start e gaps
        raw_dur = out[:, qty:]   # [B, qty] -> durações

        # Durações sem limite máximo (>= MIN_DUR)
        durations = F.softplus(raw_dur) + MIN_DUR  # [B, qty]

        # Primeiro início (>=0) e gaps (>= MIN_GAP)
        first_start = F.softplus(raw_pos[:, 0:1])                         # [B,1]
        gaps_end = F.softplus(raw_pos[:, 1:]) + MIN_GAP if qty > 1 else raw_pos.new_zeros((B, 0))

        if qty == 1:
            start_times = first_start
        else:
            # increments = dur_i + gap_{i+1}  (para i = 0..qty-2)
            increments = durations[:, :-1] + gaps_end                     # [B, qty-1]
            start_rest = first_start + torch.cumsum(increments, dim=1)    # [B, qty-1]
            start_times = torch.cat([first_start, start_rest], dim=1)     # [B, qty]

        # Ajuste para caber no horizonte: escala homogênea preserva não sobreposição
        last_end = start_times[:, -1] + durations[:, -1]                  # [B]
        alpha = torch.clamp(DAY_MAX / (last_end + 1e-8), max=1.0).unsqueeze(1)  # [B,1]
        start_times = start_times * alpha
        durations = durations * alpha

        return torch.cat([start_times, durations], dim=1)
# ...existing code...


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


    def obj_fn_Original(self, x, args):

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

    def ineq_dist(self, x, y, args):
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

    def ineq_grad(self, x, y, args):
        """
        Gradiente do resíduo de desigualdade.
        Derivadas parciais:
        """
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Calcula a distância clamped
        dist = self.ineq_dist(x, y, args)  # Tamanho esperado: [25]

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

    def ineq_partial_grad_old(self, X, Y, args):
        # Resíduo ajustado (clamp para respeitar desigualdades)
        x1 = X[:, 0]
        x2 = X[:, 1]
        grad = self.ineq_dist(X, Y, args).squeeze(1)
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

    def obj_fn_Original(self, x, args):
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

    def ineq_dist(self, x, y, args):
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

    def ineq_grad(self, x, y, args):
        """
        Gradiente combinado dos resíduos de desigualdade g1 e g2.
        """
        x1 = x[:, 0].unsqueeze(1)
        x2 = x[:, 1].unsqueeze(1)

        # Distâncias (resíduos clampados)
        dists = self.ineq_dist(x, y, args)  # shape: [batch, 2]

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
        dists = self.ineq_dist(X, Y, args)  # shape: [batch, 2]

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
