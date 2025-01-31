import torch
import torch.nn as nn
#from torch.autograd import Function
torch.set_default_dtype(torch.float64)
from scipy.optimize import approx_fprime
import math

import numpy as np
#import osqp
from qpth.qp import QPFunction
#import cyipopt
#from scipy.linalg import svd
#from scipy.sparse import csc_matrix

import hashlib
#from copy import deepcopy
import scipy.io as spio
import time

#from pypower.api import case57
#from pypower.api import opf, makeYbus
#from pypower import idx_bus, idx_gen, ppoption

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()

def h_tmin(d,htank,timeInc,min): # Reduzir os niveis do deposito de tmin em tmin 
    tmin=min # minutos
    h=np.transpose(htank)

    if(d.flag_t_inicio!=0):
        time_tmin_seg1=np.arange(0,(d.flag_t_inicio*60*60), tmin*60)
        time_tmin_seg2=np.arange(d.flag_t_inicio*60*60,(24*60*60)+tmin*60, tmin*60)
        time_tmin_seg=np.concatenate((time_tmin_seg2,time_tmin_seg1),axis=0)
    else:
        time_tmin_seg=np.arange(0,(24*60*60), (tmin*60))
    h_tmin=np.ones(len(time_tmin_seg))*999
    
    for i in range(0,len(time_tmin_seg)): #Guardar os valores cujo tempo são multiplos 
        idx = (np.where(timeInc['StartTime']==time_tmin_seg[i]))[0]
        if (idx.size != 0):
            h_tmin[i]=h[idx[0]]
   
    idx_zero=(np.where(h_tmin==999))[0] #h_15min=999 --> Não existe incremento que termine com tempo multiplo de x minutos
    if (idx_zero.size != 0):
         for i in range(0,len(idx_zero)):
                idx_end=set((np.where(time_tmin_seg[idx_zero[i]]<=timeInc['EndTime']))[0])
                idx_start=set((np.where(time_tmin_seg[idx_zero[i]]>=timeInc['StartTime']))[0])
                aux=list(idx_end.intersection(idx_start))
                if(len(aux)!=0):
                    h_tmin[idx_zero[i]]=h[aux[0]+1] 
                else:
                    print('ERROR: WATER LEVEL NOT FOUND ->'+str((time_tmin_seg[idx_zero[i]]/3600)))                     
    return h_tmin


# CODIGO ORIGINAL

# def eps_definition_F3(x,d): #definição do eps para a 3a formulação (DC) + VSPs
#     # PERTURBAÇÃO DEPENDE DO VALOR DA VARIAVEL
#     epsF_i=d.epsF_i # % de perturbação para inicio
#     epsF_d=d.epsF_d # % de perturbação para duração
#     eps = np.zeros(len(x))
# 
#     for p in range(0,d.n_pumps):
#         x_p=x[d.dc_pos[p]:d.dc_pos[p+1]] # operação por bomba
#         n_dc=d.n_dc[p]
#         eps_aux=np.zeros(len(x_p))
#         ### definição de perturbação para inicio de DC ###
#         for i in range(0,n_dc): 
#             inicio, dur = x_p[i], x_p[i + n_dc]
#             next = x_p[i+1] if i < n_dc - 1 else 24
#             flagR_i = 0
# 
#             if(inicio + (max(inicio,1)*epsF_i) + dur <= next): # progressivas standard com eps = max(inicio,1)*epsF_i
#                 eps_aux[i]=max(inicio,1)*epsF_i
#             else: 
#                 dif=next-inicio-dur
#                 if(dif >= d.dif_DC - (1/(60*60))): #6e-4): # progressivas com a diferença entre DCs
#                     eps_aux[i]=dif
#                 else: # regressivas
#                     pre = x_p[i-1] + x_p[i-1+n_dc] if i > 0 else 0  # definição da variavel pre = fim do dc anterior
# 
#                     if(inicio - (max(inicio,1)*epsF_i) >= pre):                
#                         eps_aux[i]=-max(inicio,1)*epsF_i # regressiva standard com eps=-max(inicio,1)*epsF_i 
#                     else:
#                         flagR_i=1 # Não se pode aplicar a regressiva standard sem sobrepor DCs
#                         print('starting time: standard progressive and regressive not applied')
#                         print('prev: '+str(pre)+'; inicio: '+str(inicio)+'; fim: '+str(inicio+dur)+'; next '+str(next))
# 
#             if(flagR_i==1): # Não se pode aplicar a regressiva standard sem sobrepor DCs    
#                 dif=(inicio-pre) 
#                 if(dif>= d.dif_DC - (1/(60*60))): # regressiva com eps igual a diferente entre dc's
#                     eps_aux[i]=-dif
#                 else:
#                     eps_aux[i]=max(inicio,1)*epsF_i # sobrepor para a frente -> supostamente isto não acontece     
#                     print('ERROR: DC overlapping for starting time') 
# 
#         ### definição de perturbação para duração de DC ###
#         for j in range(n_dc,len(x_p)): 
#             inicio, dur = x_p[j - n_dc], x_p[j]
#             next = x_p[j + 1 - n_dc] if j < len(x_p) - 1 else 24
#             flagR_d = 0
#             if(dur + (max(dur,1)*epsF_d) + inicio <= next): # progressiva standard com eps=max(dur,1)*epsF_d
#                 eps_aux[j]=max(dur,1)*epsF_d
#             else:
#                 dif=next - (inicio+dur)
#                 if(dif>= d.dif_DC - (1/(60*60))): # progressivas com a diferença entre DCs
#                     eps_aux[j]=dif 
#                 else: # regressivas
#                     if(dur - max(dur,1)*epsF_d >= 0):                
#                         eps_aux[j]=-max(dur,1)*epsF_d # regressiva standard com eps=-max(dur,1)*epsF_d
#                     else:
#                         flagR_d=1 # Não se pode aplicar a regressiva standard 
#                         print('duration: standard progressive and regressive not applied')                        
#                         print('inicio: '+str(inicio)+'; fim: '+str(inicio+dur)+'; next: '+str(next))
# 
#             if(flagR_d==1): # dif. regressiva para duração 
#                 # eps_aux[j] = -dur if dur >= (d.dif_DC - 1/(60*60)) else max(inicio, 1) * epsF_d
#                 if(dur >= d.dif_DC - (1/(60*60))): # regressivas com eps = -dur
#                     eps_aux[j]= -dur 
#                 else:
#                     eps_aux[j]=max(inicio,1)*epsF_d # sobrepor para a frente -> supostamente isto não acontece     
#                     print('ERROR: DC overlapping -> duration is to low to regressive')
#         
#         eps[d.dc_pos[p]:d.dc_pos[p+1]] = eps_aux
#     
#     #retificar perturbações maiores que 5 minutos
#     idx1=np.where(eps > 5/60)
#     if(len(idx1[0])!=0): eps[idx1[0]]=5/60
#     idx2=np.where(eps < - 5/60)
#     if(len(idx2[0])!=0): eps[idx2[0]]=-5/60
# 
#     if(len(x)>d.dc_pos[d.n_pumps]): #caso hajam VSPS
# 
#         for v in range(d.dc_pos[d.n_pumps],len(x)):
#             val = max(x[v], 1) * d.eps_VSP
#             if(x[v] + val < d.lim_VSP[1]): #forward
#                 eps[v] = val
#             else: #regressive
#                 eps[v] = - val
#                 if(x[v] - val < d.lim_VSP[0]):
#                     print('ERROR: VSP eps not respecting boundaries') 
#             if(val<0.0001):
#                 print('WARNING: VPS eps is too low!') 
#     return eps 

def eps_definition_F3(x_matrix, d): # definição do eps para a 3a formulação (DC) + VSPs
    # PERTURBAÇÃO DEPENDE DO VALOR DA VARIAVEL
    epsF_i = d.epsF_i # % de perturbação para inicio
    epsF_d = d.epsF_d # % de perturbação para duração
    eps_matrix = np.zeros_like(x_matrix)

    for p in range(d.n_pumps):
        x_p = x_matrix[p]  # operação por bomba
        n_dc = d.n_dc[p]
        eps_aux = np.zeros_like(x_p)
        
        ### definição de perturbação para inicio de DC ###
        for i in range(n_dc):
            inicio, dur = x_p[i], x_p[i + n_dc]
            next = x_p[i + 1] if i < n_dc - 1 else 24
            flagR_i = 0

            if inicio + (max(inicio, 1) * epsF_i) + dur <= next: # progressivas standard com eps = max(inicio,1)*epsF_i
                eps_aux[i] = max(inicio, 1) * epsF_i
            else:
                dif = next - inicio - dur
                if dif >= d.dif_DC - (1 / (60 * 60)): # progressivas com a diferença entre DCs
                    eps_aux[i] = dif
                else: # regressivas
                    pre = x_p[i - 1] + x_p[i - 1 + n_dc] if i > 0 else 0  # definição da variavel pre = fim do dc anterior

                    if inicio - (max(inicio, 1) * epsF_i) >= pre:
                        eps_aux[i] = -max(inicio, 1) * epsF_i # regressiva standard com eps=-max(inicio,1)*epsF_i 
                    else:
                        flagR_i = 1 # Não se pode aplicar a regressiva standard sem sobrepor DCs
                        print('starting time: standard progressive and regressive not applied')
                        print(f'prev: {pre}; inicio: {inicio}; fim: {inicio + dur}; next {next}')

            if flagR_i == 1: # Não se pode aplicar a regressiva standard sem sobrepor DCs    
                dif = inicio - pre
                if dif >= d.dif_DC - (1 / (60 * 60)): # regressiva com eps igual a diferente entre dc's
                    eps_aux[i] = -dif
                else:
                    eps_aux[i] = max(inicio, 1) * epsF_i # sobrepor para a frente -> supostamente isto não acontece     
                    print('ERROR: DC overlapping for starting time')

        ### definição de perturbação para duração de DC ###
        for j in range(n_dc, len(x_p)):
            inicio, dur = x_p[j - n_dc], x_p[j]
            next = x_p[j + 1 - n_dc] if j < len(x_p) - 1 else 24
            flagR_d = 0
            if dur + (max(dur, 1) * epsF_d) + inicio <= next: # progressiva standard com eps=max(dur,1)*epsF_d
                eps_aux[j] = max(dur, 1) * epsF_d
            else:
                dif = next - (inicio + dur)
                if dif >= d.dif_DC - (1 / (60 * 60)): # progressivas com a diferença entre DCs
                    eps_aux[j] = dif
                else: # regressivas
                    if dur - max(dur, 1) * epsF_d >= 0:
                        eps_aux[j] = -max(dur, 1) * epsF_d # regressiva standard com eps=-max(dur,1)*epsF_d
                    else:
                        flagR_d = 1 # Não se pode aplicar a regressiva standard 
                        print('duration: standard progressive and regressive not applied')
                        print(f'inicio: {inicio}; fim: {inicio + dur}; next: {next}')

            if flagR_d == 1: # dif. regressiva para duração 
                if dur >= d.dif_DC - (1 / (60 * 60)): # regressivas com eps = -dur
                    eps_aux[j] = -dur
                else:
                    eps_aux[j] = max(inicio, 1) * epsF_d # sobrepor para a frente -> supostamente isto não acontece     
                    print('ERROR: DC overlapping -> duration is too low to regressive')

        eps_matrix[p] = eps_aux

    # retificar perturbações maiores que 5 minutos
    eps_matrix = np.clip(eps_matrix, -5 / 60, 5 / 60)

#    if x_matrix.shape[1] > d.dc_pos[d.n_pumps]: # caso hajam VSPS
#        for v in range(d.dc_pos[d.n_pumps], x_matrix.shape[1]):
#            val = max(x_matrix[:, v], 1) * d.eps_VSP
#            eps_matrix[:, v] = np.where(x_matrix[:, v] + val < d.lim_VSP[1], val, -val)
#            if np.any(x_matrix[:, v] - val < d.lim_VSP[0]):
#                print('ERROR: VSP eps not respecting boundaries')
#            if np.any(val < 0.0001):
#                print('WARNING: VPS eps is too low!')

    return eps_matrix

def h_red3_acordeao(x_matrix, htank, timeInc, d, n_points):
    # Transpor matriz para garantir alinhamento correto
    h = np.transpose(htank)
    
    n_pumps = len(x_matrix)
    h_min_list = []
    
    for p in range(n_pumps):
        x = x_matrix[p]  # Acessa a linha p da matriz x_matrix
        n_arranques = int(len(x) / 2)
        time_seg = np.zeros(n_arranques * (2 + n_points) + 1)
        
        idx = 0
        for i in range(n_arranques):
            if d.flag_t_inicio != 0:
                x_i = max(0, min(24, x[i]))
                start = (x_i + d.flag_t_inicio) * 3600
                start = start - 24 * 3600 if start >= 24 * 3600 else start
                
                end_at = (x_i + x[i + n_arranques] + d.flag_t_inicio) * 3600
                end_dv = (x_i + x[i + n_arranques]) * 3600
                
                if end_dv > 24 * 3600:
                    end = d.flag_t_inicio
                else:
                    end = end_at - 24 * 3600 if end_at >= 24 * 3600 else end_at
                
                total_time_hor = d.flag_t_inicio * 3600
            else:
                start = x[i] * 3600
                end = (x[i] + x[i + n_arranques]) * 3600
                total_time_hor = 24 * 3600

            time_seg[idx] = math.floor(start + 0.5)
            idx += 1

            if n_points != 0:
                deltaT = (x[i + n_arranques] * 3600) / (n_points + 1)
                for k in range(1, n_points + 1):
                    interv = start + k * deltaT
                    interv = interv - 24 * 3600 if interv >= 24 * 3600 else interv
                    time_seg[idx] = math.floor(interv + 0.5)
                    idx += 1

            time_seg[idx] = math.floor(end + 0.5)
            idx += 1

        h_min = np.ones(len(time_seg)) * 999
        time_seg[-1] = total_time_hor

        for i in range(0, len(time_seg) - 1, 2 + n_points):
            idx = np.where(timeInc['StartTime'] == time_seg[i])[0]
            if idx.size != 0:
                h_min[i] = h[idx]

            if n_points != 0:
                for k in range(i + 1, i + 1 + n_points):
                    idx2 = np.where(timeInc['StartTime'] == time_seg[k])[0]
                    if idx2.size != 0:
                        h_min[k] = h[idx2[0]]

        for i in range(1 + n_points, len(time_seg) - 1, 2 + n_points):
            idx = np.where(timeInc['EndTime'] == time_seg[i])[0]
            if idx.size != 0:
                h_min[i] = h[idx[0] + 1]

        idx = np.where(timeInc['EndTime'] == time_seg[-1])[0]
        if idx.size != 0:
            h_min[-1] = h[idx[0] + 1]
        
        idx_zero = np.where(h_min == 999)[0]
        if idx_zero.size != 0:
            for i in idx_zero:
                idx_end = set(np.where(time_seg[i] <= timeInc['EndTime'])[0])
                idx_start = set(np.where(time_seg[i] >= timeInc['StartTime'])[0])
                aux = list(idx_end.intersection(idx_start))
                if len(aux) != 0:
                    h_min[i] = h[aux[0] + 1]
                else:
                    if total_time_hor <= time_seg[i] <= total_time_hor + 3600:
                        h_min[i] = h[-1]
                        print(f'ERROR: WATER LEVEL EXTRAPOLATED -> {time_seg[i] / 3600}')
                    else:
                        print(f'ERROR: WATER LEVEL NOT FOUND -> {time_seg[i] / 3600}')
        
        h_min_list.append(h_min)
    
    return np.array(h_min_list)

def round_x(x,d): #arredondar aos segundos    
    x_round=[]
    for p in range(d.n_pumps):
        x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
        x_aux=[math.floor(x_p[i]*3600 + 0.5) for i in range(0,d.n_dc[p])]
        for i in range(d.n_dc[p],2*d.n_dc[p]):
            x_aux.append(math.floor(x_aux[i-d.n_dc[p]] + (x_p[i]*3600) + 0.5))
        x_round=x_round+x_aux

    # x_round=[math.floor(x[i]*3600 + 0.5) for i in range(len(x))]    
    if(len(x)>2*np.sum(d.n_dc)): #VSP
        x_vel=x[d.dc_pos[d.n_pumps]:len(x)]
        vel_pos=np.concatenate(([0],np.cumsum(d.n_dc)))
        for p in range(0,d.n_pumps):
            x_v=x_vel[vel_pos[p]:vel_pos[p+1]]
            x_aux=[round(x_v[i], 4) for i in range(0,d.n_dc[p])]
            x_round=x_round+x_aux

    return x_round


def linear_interpolation(x_values, y_values, x_prime):
    for i in range(len(x_values) - 1):
        if x_values[i] <= x_prime <= x_values[i + 1]:
            # Aplicar interpolação linear
            y_prime = y_values[i] + (y_values[i + 1] - y_values[i]) * (x_prime - x_values[i]) / (x_values[i + 1] - x_values[i])
            return (y_prime/100)
        
    raise ValueError("x_prime está fora do intervalo dos pontos fornecidos.")
        
def h_red2(x,htank,timeInc,d): #h no inicio e fim de cada arranque - F2 + 24h
    h=np.transpose(htank)
    time_seg=np.zeros(int(len(x)*2)+1)
    t_inic=[7,11,15,19,0,2,4]*np.array([60*60])
    #t_inic=np.concatenate(([0],np.cumsum(d.t_hor_s_2F)))
    idx=0
    for i in range(0,len(x)):
        start=t_inic[i]
        end=(t_inic[i]+(x[i]*d.t_hor_s_2F[i]))
        if(start%1<0.5):
            time_seg[idx]=round(start)
            idx+=1
        else:
            time_seg[idx]=math.ceil(start)
            idx+=1
        if(end%1<0.5):
            time_seg[idx]=round(end)
            idx+=1
        else:
            time_seg[idx]=math.ceil(end)
            idx+=1
    time_seg[len(time_seg)-1]=7*(60*60)
    
    h_min=np.ones(len(time_seg))*999
    for i in range(0,len(time_seg)-1,2): # guardar valores no inicio do duty cycle
        idx = (np.where(timeInc['StartTime']==time_seg[i]))[0]
        if (idx.size != 0):
            h_min[i]=h[idx[0]]
    
    for i in range(1,len(time_seg)-1,2): # guardar valores no final do duty cycle
        idx = (np.where(timeInc['EndTime']==time_seg[i]))[0]
        if (idx.size != 0):
                h_min[i]=h[idx[0]+1]


    idx = (np.where(timeInc['EndTime']==time_seg[len(time_seg)-1]))[0] #guardar valores nas 24h
    if (idx.size != 0):
        h_min[len(time_seg)-1]=h[idx[0]+1]    

    idx_zero=(np.where(h_min==999))[0] #h_min=999 --> Não existe incremento com o tempo que queriamos
    if (idx_zero.size != 0):
        for i in range(0,len(idx_zero)):
            idx_end=set((np.where(time_seg[idx_zero[i]]<=timeInc['EndTime']))[0])
            idx_start=set((np.where(time_seg[idx_zero[i]]>=timeInc['StartTime']))[0])
            aux=list(idx_end.intersection(idx_start))
            if(len(aux)!=0):
                h_min[idx_zero[i]]=h[aux[0]+1] 
            else:
                print('ERROR: WATER LEVEL NOT FOUND ->'+str((time_seg[idx_zero[i]]/3600)))                     
    return h_min

###################################################################

# PROBLEM DC-WSS

###################################################################


class Problem_DC_WSS:
    def __init__(self, d,pumps,tanks,pipes,valves,timeInc,controls_epanet, x, valid_frac=0.0833, test_frac=0.0833):
        
        
        self._pumps = pumps
        self._tanks = tanks
        self._pipes = pipes
        self._valves = valves
        self._timeInc = timeInc
        self._controls_epanet = controls_epanet
        self._d = d
        

        self._costTariff = 0

        self._num_dc = d.num_dc
        
        self._X = torch.tensor(x)
        self._xdim = len(x)
        self._ydim = len(x)
        self._num = len(x)
        self._neq = 0
        self._ineq = 2
        self._nknowns = 0
        self._valid_frac = valid_frac
        self._test_frac = test_frac
        self._device = None
        

    @property
    def device(self):
        return self._device
    
    @property
    def X(self):
        return self._X

    @property
    def pumps(self):
        return self._pumps
    
    @property
    def tanks(self):
        return self._tanks
    
    @property
    def pipes(self):
        return self._pipes
    
    @property
    def valves(self):
        return self._valves
    
    @property
    def timeInc(self):
        return self._timeInc
    
    @property
    def controls_epanet(self):
        return self._controls_epanet
    
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
    def neq(self):
        return self._neq
    
    @property
    def ineq(self):
        return self._ineq

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

    

    def Cost(self, x ,d, pumps, tanks, pipes, valves, timeInc):
            # COM EFICIÊNCIA
        cost_pump=[]
        for p in range (0,d.n_pumps):
            cp=0
            for i in range (0,len(timeInc['StartTime'])):
                tariffpriceInc=(timeInc['duration'][i]/3600)*timeInc['Tariff'][i]
                if(pumps["pump"+str(p)+"_sp"][i]!=0 and pumps["pump"+str(p)+"_sp"][i]!=1):
                    eff=linear_interpolation(d.eff_flow_points,d.eff_points,pumps["pump"+str(p)+"_q"][i])
                    n2=1-(1-eff)*((1/pumps["pump"+str(p)+"_sp"][i])**0.1)
                    up=(abs(pumps["pump"+str(p)+"_h"][i]) * (pumps["pump"+str(p)+"_q"][i]) * 9.81)/1000 # Q-> m3/s ; H->m ; P-> kW
                    # up=(pumps["pump"+str(p)+"_p"][i]*eff) -> o de cima dá valores mais próximos
                    cost1=tariffpriceInc*up/n2
                else:
                    cost1=(tariffpriceInc*pumps["pump"+str(p)+"_p"][i])

                cp+=cost1
            cost_pump.append(cp)        
        CostT=sum(cost_pump)

        return CostT




            
    def gT(self, x,d,id,tanks, timeInc): #Lower and Higher Water Level 

        g1 = np.array([])
        for i in range(0,len(d.dc_pos)-1):
            ini=d.dc_pos[i]
            fin=d.dc_pos[i+1]
            g1_aux=h_red3_acordeao(x[ini:fin],tanks['tank'+str(id)+'_h'],timeInc,d,d.n_points_tank[id])
            g1_aux = np.ravel(g1_aux) 
            g1=np.concatenate((g1,g1_aux[0:len(g1_aux)-1])) # h no inicio e fim de cada arranque 

        g1=np.concatenate((g1, [g1_aux[-1]])) #Acrescentar as 24h

        
        return g1        



    def g_TempLog(self, x_matrix, d): #tstart(n+1) > tstop(n)  (várias bombas)
        # print('Temporal Logic Const. --> x(start-stop)')
        g5 = []
        for p in range(0, d.n_pumps): #d.n_pumps
            g5_F33 = np.zeros(d.n_dc[p])
            x_p = x_matrix[p]  # Acessa a linha p da matriz x_matrix
            
            if d.n_dc[p] != 1:
                for i in range(0, d.n_dc[p] - 1):
                    g5_F33[i] = x_p[i + 1] - (x_p[i] + x_p[i + d.n_dc[p]])
                g5_F33[i + 1] = 24 - (x_p[d.n_dc[p] - 1] + x_p[int(2 * d.n_dc[p] - 1)]) # garantir que a ultima duração não é superior a T
            else:
                g5_F33[0] = 24 - (x_p[d.n_dc[p] - 1] + x_p[int(2 * d.n_dc[p] - 1)]) # garantir que a ultima duração não é superior a T
                    
            g5 = np.concatenate((g5, g5_F33))

        # print(x_matrix)
        # print(g5)
        return g5
    
        
    def jac_TempLog(self, x,d):
        # eps_aux=AF.eps_definition_F3(x,d) 
        # jac=approx_fprime(x, g5_F3, eps_aux)
        
        n_var_pump=np.multiply(d.n_dc,2) #numero de variaveis por bomba
        for p in range(0,d.n_pumps):
            matriz1 = np.zeros((d.n_dc[p], d.n_dc[p]), dtype=int)  
            matriz2 = np.zeros((d.n_dc[p], d.n_dc[p]), dtype=int)  
            for i in range(0,d.n_dc[p]):
                matriz1[i][i] = -1.  
                matriz2[i][i] = -1.  
                if(i!=d.n_dc[p]-1):
                    matriz1[i][i+1] = 1.  
            jac_aux=np.concatenate((matriz1,matriz2), axis=1)
            
            if(d.n_pumps!=1):
                if(p==0):
                    matriz_d=np.zeros((d.n_dc[p], sum(n_var_pump[p+1:len(n_var_pump)])), dtype=int)
                    jac=np.concatenate((jac_aux,matriz_d), axis=1)
                
                elif(p==d.n_pumps-1):
                    matriz_a=np.zeros((d.n_dc[p], sum(n_var_pump[0:p])), dtype=int)
                    jac1=np.concatenate((matriz_a,jac_aux), axis=1)
                    jac=np.concatenate((jac,jac1), axis=0)  

                else:                            
                    matriz_a=np.zeros((d.n_dc[p], sum(n_var_pump[0:p])), dtype=int)
                    matriz_d=np.zeros((d.n_dc[p], sum(n_var_pump[p+1:len(n_var_pump)])), dtype=int)
                    jac1=np.concatenate((matriz_a,jac_aux,matriz_d), axis=1)
                    jac=np.concatenate((jac,jac1), axis=0)                               
            else:
                jac=jac_aux

        if(len(x)>d.dc_pos[d.n_pumps]): #VSP
            jac_vsp=np.zeros((len(jac), sum(d.n_dc)), dtype=int)  
            jac=np.concatenate((jac,jac_vsp),axis=1)
        
        # mod=np.linalg.norm(jac)
        return jac


    def jac_gT(self, x, d):
        
        eps_aux = torch.tensor(eps_definition_F3(x, d))
        jac = eps_aux
        #jac = approx_fprime(x, self.gT, eps_aux)
        
        return jac

#    CODIGO ORIGINAL
#    def g_TempLog(self, x,d): #tstart(n+1) > tstop(n)  (várias bombas)
#        # print('Temporal Logic Const. --> x(start-stop)')
#        g5=[]
#        for p in range(0,d.n_pumps): #d.n_pumps
#            g5_F33=np.zeros(d.n_dc[p])
#            x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
#            
#            if(d.n_dc[p]!=1):
#                for i in range(0,d.n_dc[p]-1):
#                    g5_F33[i]=x_p[i+1]-(x_p[i]+x_p[i+d.n_dc[p]])
#                g5_F33[i+1]=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
#            else:
#                g5_F33[0]=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
#                    
#            g5=np.concatenate((g5,g5_F33))
#
#        # print(x)
#        # print(g5)
#        return g5     



    
    def g_TempLog_dist(self, x, d):
        
        resid = self.g_TempLog(x, d)
    
        resid_tensor = torch.tensor(resid, dtype=torch.float32)

        #  Aplica clamp para limitar os valores ao intervalo [0, +inf]
        return torch.clamp(resid_tensor, min=0)
    
    
    def ineq_dist(self, x, y):
        
        resids = torch.tensor(self.ineq_resid(x, y))
        
        # min tank = 2
        # max tank = 7
        
        
        
        return torch.clamp(resids, 2, 7)
        
        
    
    def ineq_resid(self, x, y):
        
        return self.gT(x, self.d, 0, self.tanks, self.timeInc)
    


    def ineq_grad(self, X, Y):
        
        
        #ineq_jac = self.jac_gT(X, self.d)
        ineq_dist = self.ineq_dist(X, Y)
        
        #return ineq_jac * ineq_dist.unsqueeze(1)

        #return ineq_dist.unsqueeze(1)
    
        return ineq_dist

    def process_output(self, X, Y):
        return Y        
    
    
    
    

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
        return 'Problem_Non_Linear-{}-{}-{}-{}'.format(
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

        #print('resultado ineq_resid (resids)', resids)
        #print('***')
        #print(resids.unsqueeze(1))
        #print('***')
        #print('antes de mandar para total_loss: ', torch.clamp(resids, 0))
        #print('***')
        
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

        grad_x1 = (1.5 * x1)
        grad_x2 = (0.5 * x2)
        
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
        #grad = self.ineq_dist(x1, x2)

        # Inicialização do tensor para gradientes
        Y = torch.zeros(X.shape[0], X.shape[1], device=self.device)
        
        Y[:, 0] = grad * 1.5 * x1# Gradiente para x1
        Y[:, 1] = grad * 0.5 * x2# Gradiente para x2

        # Retornar gradientes ajustados
        return Y

    def ineq_partial_grad(self, X, Y):
        # Assumindo que as duas variáveis são "parciais"
        grad_x1 = 1.5 * X[:, 0]  # Derivada em relação a x1
        grad_x2 = 0.5 * X[:, 1]  # Derivada em relação a x2
        
        grad = torch.stack([grad_x1, grad_x2], dim=1)
        
        # A parte efetiva do gradiente pode ser calculada diretamente
        grad_effective = 2 * torch.clamp(Y - 0.5, 0)  # Ajuste para garantir valores não negativos
        
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
            x2 = torch.sqrt(torch.clamp((2 / 3) * term, min=0))  # Clamp para evitar valores inválidos
            
            Y[i, self.other_vars] = x2  
        
        return Y
