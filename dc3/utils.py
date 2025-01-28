import torch
import torch.nn as nn
#from torch.autograd import Function
torch.set_default_dtype(torch.float64)

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

def h_red3_acordeao(x,htank,timeInc,d,n_points): #h no inicio + fim de cada DC + divisão em n_points tempos +  24h
    h=np.transpose(htank)
    n_arranques=int(len(x)/2)
    time_seg=np.zeros(n_arranques*(2+n_points) + 1)
    
    idx=0
    for i in range(0,n_arranques): #ordenar tempos e verificar questão de arredondamentos de segundos
        #Correção do tempo de inicio e fim (7h-0h-7h)
        if(d.flag_t_inicio!=0):
            if(x[i] < 0): x[i]=0 
            if(x[i] > 24): x[i]=24

            #START TIME
            start=(x[i]+d.flag_t_inicio)*(60*60)
            if(start>=(24*(60*60))):
                start-=(24*(60*60))
        
            #END TIME
            end_at=(x[i]+x[i+n_arranques]+d.flag_t_inicio)*(60*60) # tempo final com a atualização da hora
            end_dv=(x[i]+x[i+n_arranques])*(60*60) #tempo final pelas variáveis de decisão
            if(end_dv>(24*60*60)):
                end=d.flag_t_inicio
            else:
                if(end_at>=24*(60*60)):
                    end=end_at-(24*(60*60))
                else:
                    end=end_at
            total_time_hor=d.flag_t_inicio*(60*60)
        else:
            start=x[i]*(60*60)
            end=(x[i]+x[i+n_arranques])*(60*60)
            total_time_hor=24*(60*60)

        #guardar tempo de inicio de op.
        time_seg[idx]=math.floor(start + 0.5)
        idx+=1

        if(n_points!=0):
            #tempo de intermédio de cada DC 
            deltaT=(x[i+n_arranques]*(60*60))/(n_points+1)
            for k in range(1,n_points+1):
                interv=start+k*deltaT
                if(interv>=24*60*60):
                    interv-=(24*60*60)
                time_seg[idx]=math.floor(interv + 0.5)
                idx+=1
        
        #tempo de fim
        time_seg[idx]=math.floor(end + 0.5)
        idx+=1

    h_min=np.ones(len(time_seg))*999
    time_seg[len(time_seg)-1]=total_time_hor

    # guardar valores no inicio do duty cycle
    for i in range(0,len(time_seg)-1,2+n_points): 
        idx = (np.where(timeInc['StartTime']==time_seg[i]))[0]
        if (idx.size != 0):
            h_min[i]=h[idx[0]]
        
        # guardar valores nos pontos intermédios do duty-cycle
        if(n_points!=0):
            for k in range(i+1,i+1+n_points):
                idx2 = (np.where(timeInc['StartTime']==time_seg[k]))[0]
                if (idx2.size != 0):
                    h_min[k]=h[idx2[0]]

    # guardar valores no final do duty cycle
    for i in range(1+n_points,len(time_seg)-1,2+n_points): 
        idx = (np.where(timeInc['EndTime']==time_seg[i]))[0]
        if (idx.size != 0):
                h_min[i]=h[idx[0]+1]

    #guardar valores nas 24h
    idx = (np.where(timeInc['EndTime']==time_seg[len(time_seg)-1]))[0]
    if (idx.size != 0):
        h_min[len(time_seg)-1]=h[idx[0]+1]       
    
    #CASO HAJAM NIVEIS POR DEFINIR
    idx_zero=(np.where(h_min==999))[0] #h_min=999 --> Não existe incremento com o tempo que queriamos
    if (idx_zero.size != 0):
        for i in range(0,len(idx_zero)):
            idx_end=set((np.where(time_seg[idx_zero[i]]<=timeInc['EndTime']))[0])
            idx_start=set((np.where(time_seg[idx_zero[i]]>=timeInc['StartTime']))[0])
            aux=list(idx_end.intersection(idx_start))
            if(len(aux)!=0):
                h_min[idx_zero[i]]=h[aux[0]+1] 
            else:
                if(time_seg[idx_zero[i]] >= total_time_hor and time_seg[idx_zero[i]] <= total_time_hor + 1*60*60): #supostamente nunca acontece
                    h_min[idx_zero[i]]=h[len(h)-1]
                    print('ERROR: WATER LEVEL EXTRAPOLATED ->'+str((time_seg[idx_zero[i]]/3600))) 
                # elif(time_seg[idx_zero[i]] <= total_time_hor and time_seg[idx_zero[i]] <= total_time_hor +)
                     
                else:                    
                    print('ERROR: WATER LEVEL NOT FOUND ->'+str((time_seg[idx_zero[i]]/3600))) 
    return h_min

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
        
        
        self._pumps = torch.tensor(pumps, dtype=torch.get_default_dtype())
        self._tanks = torch.tensor(tanks, dtype=torch.get_default_dtype())
        self._pipes = torch.tensor(pipes, dtype=torch.get_default_dtype())
        self._valves = torch.tensor(valves, dtype=torch.get_default_dtype())
        self._timeInc = torch.tensor(timeInc, dtype=torch.get_default_dtype())
        self._controls_epanet = controls_epanet
        self._d = d
        

        self._tar_beg = d.tar_beg
        self._tar_end = d.tar_end
        self._tariff_value = d.tariff_value
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
    def tar_beg(self):
        return self._tar_beg
    
    @property
    def tar_end(self):
        return self._tar_end
    
    @property
    def tariff_value(self):
        return self._tariff_value
    
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
        # print('OF --> x')
        #print(x)

        #if(flag==1):
        #    with open(r"Data Files\x.csv",'ab') as x_C:
        #        np.savetxt(x_C,x*np.ones((1,len(x))),delimiter=";") 

        #flag_sol=0
        #if(len(log.solutions)!=0):
        #    x_round=round_x(x,d)
        #    # procurar solução
        #    try:
        #        id=log.solutions.index(x_round)
        #    except ValueError:
        #        id=-1
#
        #    if(id!=-1):
#       #     flag_sol=1
        #        CostT=log.cost_solutions[id]
        #        log.n_cost+=1

        #if(flag_sol==0):
            #d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
            #SEM CONTAR COM EFICIÊNCIA
            # CostT=0
            # for p in range (0,d.n_pumps):
            #     c=0
            #     for i in range (0,len(timeInc['StartTime'])):
            #         tariffpriceInc=(timeInc['duration'][i]/3600)*timeInc['Tariff'][i]
            #         cost=(tariffpriceInc*pumps["pump"+str(p)+"_p"][i])
            #         # if(pumps["pump"+str(p)+"_sp"][i]!=0):
            #         #     eff=linear_interpolation(d.eff_flow_points,d.eff_points,pumps["pump"+str(p)+"_q"][i])
            #         #     up=(abs(pumps["pump"+str(p)+"_h"][i]) * (pumps["pump"+str(p)+"_q"][i]) * 9.81)/1000 # Q-> m3/s ; H->m ; P-> kW
            #         #     cost1=tariffpriceInc*up/eff
            #         CostT += cost
            #         c+= cost
            #     print(c)    
            # print(CostT)

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
        #log(x,CostT,[],d)

        # if(flag==1):
            # with open(r'Data Files\CostT.csv','ab') as c:
            #     np.savetxt(c,np.ones((1,1))*CostT,delimiter=";")

        # print(CostT)
        return CostT




            
    def gT(self, x,d,id,log): #Lower and Higher Water Level 

        if(len(log.x_round)!=0):
            roundx=round_x(x,d)
            # procurar solução
            try:
                idx=log.x_round.index(roundx)
            except ValueError:
                idx=-1

            if(idx!=-1):
                flag_sol=1
                tanks=log.tanks[idx]
                timeInc=log.timeInc[idx]
                log.n_tank+=1

        #if(flag_sol==0):
        #    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
        g1=[]
        for i in range(0,len(d.dc_pos)-1):
            ini=d.dc_pos[i]
            fin=d.dc_pos[i+1]
            g1_aux=h_red3_acordeao(x[ini:fin],tanks['tank'+str(id)+'_h'],timeInc,d,d.n_points_tank[id])
            g1=np.concatenate((g1,g1_aux[0:len(g1_aux)-1])) # h no inicio e fim de cada arranque 

        g1=np.concatenate((g1, [g1_aux[-1]])) #Acrescentar as 24h
            # print(id)
            # print(x)
            # print(g1)

        #elif(d.ftype==1):
        #    g1=h_tmin(d,tanks['tank'+str(id)+'_h'],timeInc)
        
        # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as x_g:
        #     np.savetxt(x_g,x*np.ones((1,len(x))),delimiter=";") 

        # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as c:
        #     np.savetxt(c,np.ones((1,1))*g1,delimiter=";") 
        
        # print('Water --> x_T'+id)
        # print(x)
        # print(g1)
        return g1        


    
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
    


    def g_TempLog(self, x,d): #tstart(n+1) > tstop(n)  (várias bombas)
        # print('Temporal Logic Const. --> x(start-stop)')
        g5=[]
        for p in range(0,d.n_pumps): #d.n_pumps
            g5_F33=np.zeros(d.n_dc[p])
            x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
            
            if(d.n_dc[p]!=1):
                for i in range(0,d.n_dc[p]-1):
                    g5_F33[i]=x_p[i+1]-(x_p[i]+x_p[i+d.n_dc[p]])
                g5_F33[i+1]=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
            else:
                g5_F33[0]=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
                    
            g5=np.concatenate((g5,g5_F33))

        # print(x)
        # print(g5)
        return g5     
    
    
    def g_TempLog_dist(self, x, d):
        
        resid = self.g_TempLog(x, d)
    
        resid_tensor = torch.tensor(resid, dtype=torch.float32)

        #  Aplica clamp para limitar os valores ao intervalo [0, +inf]
        return torch.clamp(resid_tensor, min=0)
    
    
    
    def g_TempLog_Grad(self, x, d):
        
        pass


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
