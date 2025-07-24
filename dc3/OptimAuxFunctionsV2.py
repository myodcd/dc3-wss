import math
import numpy as np
import EPANET_API as EPA_API
from scipy.optimize import approx_fprime
import matplotlib as plt
plt.rcParams.update(plt.rcParamsDefault)
from numba import jit
import torch
from torch.autograd import grad
import utils as utils


SHOW_PRINT = False
    
    
##############################################
############## AUXILIAR FUNCTIONS ############
##############################################
jit(nopython=True)
def eps_definition_F2(x,d): #definição do eps para a 2a formulação (continuous)
    eps=d.epsF_i # % de perturbação
    eps_aux=[0 for i in range(len(x))]

    for i in range(0,len(x)): # progressivas com pert = (max(x[i],1)*eps)
        if(x[i] + (max(x[i],1)*eps) <=1):
            eps_aux[i]=(max(x[i],1)*eps)   

        else: # regressivas com pert = -(max(x[i],1)*eps)
            eps_aux[i]=-(max(x[i],1)*eps)

    return eps_aux

jit(nopython=True)
def eps_definition_F3(x,d): #definição do eps para a 3a formulação (DC) + VSPs
    # PERTURBAÇÃO DEPENDE DO VALOR DA VARIAVEL
    epsF_i=d.epsF_i # % de perturbação para inicio
    epsF_d=d.epsF_d # % de perturbação para duração
    eps = np.zeros(len(x))

    for p in range(0,len(d.pumps_to_opt)):
        x_p=x[d.dc_pos[p]:d.dc_pos[p+1]] # operação por bomba
        n_dc=d.n_dc[p]
        eps_aux=np.zeros(len(x_p))
        ### definição de perturbação para inicio de DC ###
        for i in range(0,n_dc): 
            inicio, dur = x_p[i], x_p[i + n_dc]
            next = x_p[i+1] if i < n_dc - 1 else 24
            flagR_i = 0

            if(inicio + (max(inicio,1)*epsF_i) + dur < next): # progressivas standard com eps = max(inicio,1)*epsF_i
                eps_aux[i]=max(inicio,1)*epsF_i
            else: 
                dif=next-inicio-dur
                #if(dif >= d.dif_DC - (1/(60*60))): #6e-4): # progressivas com a diferença entre DCs
                if(dif - (1/(60*60))>= d.dif_DC):
                    eps_aux[i]=dif - (1/(60*60)) # Para não juntar
                else: # regressivas
                    pre = x_p[i-1] + x_p[i-1+n_dc] if i > 0 else 0  # definição da variavel pre = fim do dc anterior

                    if(inicio - (max(inicio,1)*epsF_i) >= pre):                
                        eps_aux[i]=-max(inicio,1)*epsF_i # regressiva standard com eps=-max(inicio,1)*epsF_i 
                    else:
                        flagR_i=1 # Não se pode aplicar a regressiva standard sem sobrepor DCs
                        if SHOW_PRINT:
                            print('starting time: standard progressive and regressive not applied')
                            print('prev: '+str(pre)+'; inicio: '+str(inicio)+'; fim: '+str(inicio+dur)+'; next '+str(next))

            if(flagR_i==1): # Não se pode aplicar a regressiva standard sem sobrepor DCs    
                dif=(inicio-pre) 
                if(dif- (1/(60*60))>= d.dif_DC): # regressiva com eps igual a diferente entre dc's
                #if(dif>= d.dif_DC - (1/(60*60))): # regressiva com eps igual a diferente entre dc's
                    eps_aux[i]=-(dif-(1/(60*60)))
                else:
                    eps_aux[i]=max(inicio,1)*epsF_i # sobrepor para a frente -> supostamente isto não acontece     
                    if SHOW_PRINT:
                        print('ERROR: DC overlapping for starting time') 

        ### definição de perturbação para duração de DC ###
        for j in range(n_dc,len(x_p)): 
            inicio, dur = x_p[j - n_dc], x_p[j]
            next = x_p[j + 1 - n_dc] if j < len(x_p) - 1 else 24
            flagR_d = 0
            if(dur + (max(dur,1)*epsF_d) + inicio < next): # progressiva standard com eps=max(dur,1)*epsF_d
                eps_aux[j]=max(dur,1)*epsF_d
            else:
                dif=next - (inicio+dur)
                if(dif- (1/(60*60))>= d.dif_DC): # progressivas com a diferença entre DCs
                #if(dif>= d.dif_DC - (1/(60*60))): # progressivas com a diferença entre DCs
                    eps_aux[j]=dif - (1/(60*60)) # Para não juntar
                else: # regressivas
                    if(dur - max(dur,1)*epsF_d >= 0):                
                        eps_aux[j]=-max(dur,1)*epsF_d # regressiva standard com eps=-max(dur,1)*epsF_d
                    else:
                        flagR_d=1 # Não se pode aplicar a regressiva standard 
                        if SHOW_PRINT:
                            print('duration: standard progressive and regressive not applied')                        
                            print('inicio: '+str(inicio)+'; fim: '+str(inicio+dur)+'; next: '+str(next))

            if(flagR_d==1): # dif. regressiva para duração 
                # eps_aux[j] = -dur if dur >= (d.dif_DC - 1/(60*60)) else max(inicio, 1) * epsF_d
                if(dur >= d.dif_DC): # regressivas com eps = -dur
                    eps_aux[j]= -dur 
                else:
                    eps_aux[j]=max(inicio,1)*epsF_d # sobrepor para a frente -> supostamente isto não acontece     
                    if SHOW_PRINT:
                        print('ERROR: DC overlapping -> duration is to low to regressive')
        
        eps[d.dc_pos[p]:d.dc_pos[p+1]] = eps_aux
    
    #retificar perturbações maiores que 5 minutos
    idx1=np.where(eps > 5/60)
    if(len(idx1[0])!=0): eps[idx1[0]]=5/60
    idx2=np.where(eps < - 5/60)
    if(len(idx2[0])!=0): eps[idx2[0]]=-5/60

    return eps  


jit(nopython=True)
def TempLog(d): #bounds da restrição da logica temporal entre DCs
    g=[]
    for i in range(0,len(d.n_dc)):
        aux=np.concatenate((np.ones(d.n_dc[i]-1)*(d.dif_DC),[0]))
        g=np.concatenate((g,aux))
    return g

jit(nopython=True)
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
                    if SHOW_PRINT:
                        print('ERROR: WATER LEVEL EXTRAPOLATED ->'+str((time_seg[idx_zero[i]]/3600))) 
                # elif(time_seg[idx_zero[i]] <= total_time_hor and time_seg[idx_zero[i]] <= total_time_hor +)
                     
                else:                    
                    if SHOW_PRINT:
                        print('ERROR: WATER LEVEL NOT FOUND ->'+str((time_seg[idx_zero[i]]/3600))) 
    return h_min


jit(nopython=True)
def h_red3(x,htank,timeInc,d): #h no inicio e fim de cada arranque - F3 + 24h
    h=np.transpose(htank)
    n_arranques=int(len(x)/2)
    time_seg=np.zeros(n_arranques*2 + 1)
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

        time_seg[idx]=math.floor(start + 0.5)
        idx+=1
        
        time_seg[idx]=math.floor(end + 0.5)
        idx+=1

    h_min=np.ones(len(time_seg))*999
    time_seg[len(time_seg)-1]=total_time_hor
    
    # guardar valores no inicio do duty cycle
    for i in range(0,len(time_seg)-1,2): 
        idx = (np.where(timeInc['StartTime']==time_seg[i]))[0]
        if (idx.size != 0):
            h_min[i]=h[idx[0]]
    
    # guardar valores no final do duty cycle
    for i in range(1,len(time_seg)-1,2):
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
                    if SHOW_PRINT:
                        print('ERROR: WATER LEVEL EXTRAPOLATED ->'+str((time_seg[idx_zero[i]]/3600))) 
                else:
                   if SHOW_PRINT: 
                        print('ERROR: WATER LEVEL NOT FOUND ->'+str((time_seg[idx_zero[i]]/3600))) 
    return h_min

jit(nopython=True)
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

jit(nopython=True)
def h_red1(x,htank,timeInc,d): #h no fim de cada deltaT - F1 + 24h
    h=np.transpose(htank)
    time_seg=np.zeros(int(len(x)*2)+1)
    t_inic=np.concatenate(([0],np.cumsum(d.t_hor_s_1F)))
    idx=0
    for i in range(0,len(x)):
        start=t_inic[i]
        end=t_inic[i]+(x[i]*d.t_hor_s_1F[i])
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
    
    time_seg[len(time_seg)-1]=24*(60*60)
    
    h_min=np.ones(len(time_seg))*999
    for i in range(0,len(time_seg)-1,2): #guardar valores no inicio do dt
        idx = (np.where(timeInc['StartTime']==time_seg[i]))[0]
        if (idx.size != 0):
            h_min[i]=h[idx[0]]
    
    for i in range(1,len(time_seg)-1,2): #guardar valores no final do dt
        idx = (np.where(timeInc['EndTime']==time_seg[i]))[0]
        if (idx.size != 0):
            if(idx[0]!=len(h)-1):
                h_min[i]=h[idx[0]+1]
            else:
                h_min[i]=h[idx[0]]

    idx = (np.where(timeInc['EndTime']==time_seg[len(time_seg)-1]))[0] #guardar valores nas 24h
    if (idx.size != 0):
        h_min[len(time_seg)-1]=h[idx[0]]  
    
    idx_zero=(np.where(h_min==999))[0] #h_min=999 --> Não existe incremento que termine com tempo multiplo de 10 minutos
    if (idx_zero.size != 0):
        for i in range(0,len(idx_zero)):
            idx_end=(np.where(time_seg[idx_zero[i]]<=timeInc['EndTime']))[0]
            idx_start=(np.where(time_seg[idx_zero[i]]>=timeInc['StartTime']))[0]
            if(idx_end.size!=0):
                if(idx_end[0]==idx_start[len(idx_start)-1]):
                    h_min[idx_zero[i]]=h[idx_end[0]]  
            elif(idx_start.size!=0 and idx_end.size==0): #último incremento (24h)
                h_min[idx_zero[i]]=h[len(h)-1]
    return h_min

jit(nopython=True)
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
                    if SHOW_PRINT:
                        print('ERROR: WATER LEVEL NOT FOUND ->'+str((time_tmin_seg[idx_zero[i]]/3600)))                     
    return h_tmin

jit(nopython=True)
def linear_interpolation(x_values, y_values, x_prime):
    for i in range(len(x_values) - 1):
        if x_values[i] <= x_prime <= x_values[i + 1]:
            # Aplicar interpolação linear
            y_prime = y_values[i] + (y_values[i + 1] - y_values[i]) * (x_prime - x_values[i]) / (x_values[i + 1] - x_values[i])
            return (y_prime/100)
        
    raise ValueError("x_prime está fora do intervalo dos pontos fornecidos.")

### GRÁFICOS ###
jit(nopython=True)
def pre_graphic(x,y):
    n_days=math.floor(x[-1]/(60*60*24)+0.5)
    ynew=[0 for i in range (0,len(y)*2)]
    xnew=[0 for i in range (0,len(y)*2)]
    ind=0
    for i in range(0,len(y)):
        ynew[ind]=y[i]
        xnew[ind]=x[i]
        ind+=1
        if(i<len(y)-1):
            ynew[ind]=y[i]
            xnew[ind]=x[i+1]
            ind+=1
        else:
            ynew[ind]=y[i]
            xnew[ind]=60*60*24*n_days
            ind+=1
    return xnew,ynew

jit(nopython=True)
def plot_cons_pattern():
    file1 = open("PE_Espinheira.pat", "r")
    content1 = file1.read()
    aux1=content1.split("\n")
    consumos1=(np.array(aux1[2:])).astype(float)
    time1=np.arange(0, 24, 5/60)
    file1.close()
    
    # file2 = open("Entroncamento.pat", "r")
    # content2 = file2.read()
    # aux2=content2.split("\n")
    # consumos2=(np.array(aux2[2:])).astype(float)
    # time2=np.arange(0, 24, 5/60)
    # file2.close()

#    with plt.style.context(['science']):
#        # Gráfico status vs nivel dos tanques
#        fig1, ax = plt.subplots(1)     
#        fig1.set_size_inches((10,7))
#        plt.rcParams['xtick.labelsize']=20
#        ax.plot(time1,consumos1,'lightcoral', label='CP Espinheira',linewidth=0.9)        
#        # ax.plot(time2,consumos2,'steelblue', label='CP Entroncamento', linestyle = (0, (5, 5)),linewidth=0.9)
#        ax.legend(loc=2,fontsize=15)
#        ax.set_xlabel('Time (h)', fontsize=20)
#        ax.set_ylabel('Water demand($\mathrm{m}^3/\mathrm{h}$)', fontsize=20)
#        ax.set(ylim=(0, max(consumos1)+15))
#        ax.set(xlim=(0, 24))
#        plt.show()
#        fig1.savefig('esp_cons.pdf',format='pdf')

#####################################
########### CONSTRAINTS #############
#####################################
jit(nopython=True)
def Cost(x, d, log, flag):
    # Converta x em tensor caso ainda não esteja

    if flag == 1:
        with open(r"Data Files\x.csv", 'ab') as x_C:
            torch.save((x * torch.ones((1, len(x)))), x_C)

    flag_sol = 0
    if len(log.solutions) != 0:
        x_round = round_x(x, d)
        try:
            id = log.solutions.index(x_round)
        except ValueError:
            id = -1

        if id != -1:
            flag_sol = 1
            CostT = log.cost_solutions[id]
            log.n_cost += 1

    if flag_sol == 0:
        d, pumps, tanks, pipes, valves, timeInc, controls_epanet = EPA_API.EpanetSimulation(x, d, 0)

        cost_pump = torch.zeros(d.n_pumps)
        for p in range(d.n_pumps):
            #cp = 0.0
            cp = torch.tensor(0.0)
                
            pumps['pump'+str(p)+'_eff']=pumps['pump'+str(p)+'_eff'].astype(float) #eficiência
            pumps['pump'+str(p)+'_pot']=pumps['pump'+str(p)+'_pot'].astype(float) #potência
            
            for i in range(len(timeInc['StartTime'])):
                tariffpriceInc=(timeInc['duration'][i]/3600)*pumps['pump'+str(p)+'_tar'][i] 
                eff=linear_interpolation(d.eff_flow_points[p], d.eff_points[p],pumps["pump"+str(p)+"_q"][i])    

                sp_val = pumps[f"pump{p}_sp"][i]
                q_val = pumps[f"pump{p}_q"][i]
                h_val = pumps[f"pump{p}_h"][i]
                p_val = pumps[f"pump{p}_p"][i]
                
                if(sp_val!=0 and sp_val!=1 and pumps["pump"+str(p)+"_s"][i]==1):
                    n2=1-(1-eff)*((1/pumps["pump"+str(p)+"_sp"][i])**0.1)
                    pumps['pump'+str(p)+'_eff'][i]=n2
                    if(d.units_flow==1): # litros por segundo 
                        up=(abs(pumps["pump"+str(p)+"_h"][i]) * pumps["pump"+str(p)+"_q"][i] * 9.81)/1000                    
                    else: # metros cubicos por hora
                        up=(abs(pumps["pump"+str(p)+"_h"][i]) * (pumps["pump"+str(p)+"_q"][i]/3600) * 9.81) # Q-> m3/s ; H->m ; P-> kW
                    cost1=tariffpriceInc*up/n2
                    pumps['pump'+str(p)+'_pot'][i]=(up/n2)
                else:
                    cost1=(tariffpriceInc*pumps["pump"+str(p)+"_p"][i])
                    pumps['pump'+str(p)+'_pot'][i]=pumps["pump"+str(p)+"_p"][i] 
                    pumps['pump'+str(p)+'_eff'][i]=eff                          




                cp += cost1
            #cost_pump.append(cp)
            cost_pump[p] = cp
        CostT = torch.sum(cost_pump)
        
        if(flag_sol==0):
            log(x,timeInc,tanks,pumps,CostT,d)


        
        ###### INICIO - CUSTO TARIFAS #####

        #tar_beg = [2, 4, 1, 2, 3, 12]
        #tar_end = [2, 6, 7, 9, 12, 24]
        #tariff_value = [0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094]
#
        #x_arr = np.array(x)
        #start_times = x_arr[:5]
        #durations = x_arr[5:]
#
        #custo_tarifario_extra = 0.0
#
        #for start, dur in zip(start_times, durations):
        #    task_start = float(start)
        #    task_end = float(start + dur)
        #    for beg, end, val in zip(tar_beg, tar_end, tariff_value):
        #        overlap = max(0.0, min(task_end, end) - max(task_start, beg))
        #        custo_tarifario_extra += overlap * val

        # Converta para tensor e some ao custo hidráulico

        ###### fim - CUSTO TARIFAS #####

        #CostT += torch.tensor(custo_tarifario_extra, dtype=CostT.dtype)

    return CostT



jit(nopython=True)
def gT_id(x,d,id_t,id_p,log): #Water Level - com id de tanque e das bombas
    flag_sol=0
    if(len(log.x_round)!=0):
        roundx=round_x(x,d)
        try:
            idx=log.x_round.index(roundx)
        except ValueError:
            idx=-1

        if(idx!=-1):
           flag_sol=1
           tanks=log.tanks[idx]
           timeInc=log.timeInc[idx]
           log.n_tank+=1

    if(flag_sol==0):
        d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
        log(x,tanks,timeInc,d)
    
    if(len(id_p)==1):
        x_p=x[d.dc_pos[id_p[0]]:d.dc_pos[id_p[0]+1]]
        g1=h_red3_acordeao(x_p,tanks['tank'+str(id_t)+'_h'],timeInc,d,d.n_points_tank[id_t])
    else:
        g1=[]
        for p in range(len(id_p)):
            x_p=x[d.dc_pos[id_p[p]]:d.dc_pos[id_p[p]+1]]
            g_aux=h_red3_acordeao(x_p,tanks['tank'+str(id_t)+'_h'],timeInc,d,d.n_points_tank[id_t])
            g1=np.concatenate((g1,g_aux[0:len(g_aux)-1]))
        g1=np.concatenate((g1,[g_aux[-1]])) #24h
    return g1
    
jit(nopython=True)    
def gT(x,d,id,log): #Lower and Higher Water Level 
    # print('g'+id)
    
    if(d.ftype==2):
        d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
        # g1_p1=AF.h_red2(x[0:7],tanks['tank0_h'],timeInc) # h no inicio e fim de cada arranque + 24h  - Pump 1A
        # g1_p2=AF.h_red2(x[7:14],tanks['tank0_h'],timeInc)  # h no inicio e fim de cada arranque + 24h  - Pump 2B
        # g1_p3=AF.h_red2(x[14:21],tanks['tank0_h'],timeInc)  # h no inicio e fim de cada arranque + 24h  - Pump 3B
        # g1=np.concatenate((g1_p1[0:len(g1_p1)-1],g1_p2[0:len(g1_p2)-1],g1_p3))
        g1=[]
        for i in range(0,len(x)-int(len(x)/len(d.pumps_to_opt)),int(len(x)/len(d.pumps_to_opt))):
            ini=i
            fin=i+int(len(x)/len(d.pumps_to_opt))
            g1_aux=h_red2(x[ini:fin],tanks['tank'+str(id)+'_h'],timeInc)
            g1=np.concatenate((g1,g1_aux[0:len(g1_aux)-1])) # h no inicio e fim de cada arranque

        ini=(len(d.pumps_to_opt)-1)*int(len(x)/len(d.pumps_to_opt))
        fin=len(x)
        g1_aux= h_red2(x[ini:fin],tanks['tank'+str(id)+'_h'],timeInc) # h no inicio e fim de cada arranque + 24h       
        g1=np.concatenate((g1,g1_aux)) 
            
    elif(d.ftype==3):
            # flag_sol=0
            # log.solutions=[]
            # if(len(log.solutions)!=0):
            #     roundx=round_x(x,d)
            #     # procurar solução
            #     try:
            #         idx=log.solutions.index(roundx)
            #     except ValueError:
            #         idx=-1
    
            #     if(idx!=-1):
            #         flag_sol=1
            #         tanks=log.tanks[idx]
            #         timeInc=log.timeInc[idx]
            #         log.n_tank+=1
            # else:
            #     roundx=round_x(x,d)
            
            # if(flag_sol==0):
                # C=Cost(x,d,log,0)        
                # idx=log.solutions.index(roundx)
                # tanks=log.tanks[idx]
                # timeInc=log.timeInc[idx]
            d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
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

    elif(d.ftype==1):
        g1=h_tmin(d,tanks['tank'+str(id)+'_h'],timeInc)
    
    # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as x_g:
    #     np.savetxt(x_g,x*np.ones((1,len(x))),delimiter=";") 

    # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as c:
    #     np.savetxt(c,np.ones((1,1))*g1,delimiter=";") 
    
    # print('Water --> x_T'+id)
    # print(x)
    # print(g1)
    return g1


jit(nopython=True)
def gT_DC_min(x,d,id,min): #Lower and Higher Water Level -  x em x minutos + inicio e fim de DC
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
    # de x em x minutos + inicio e fim de DC
    g11=h_tmin(d,tanks['tank'+str(id)+'_h'],timeInc,min)
    g22=gT(x,d,id)
    g1=np.concatenate((g11,g22[0:len(g22)-1]))
   
    # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as x_g:
    #     np.savetxt(x_g,x*np.ones((1,len(x))),delimiter=";") 

    # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as c:
    #     np.savetxt(c,np.ones((1,1))*g1,delimiter=";") 
    
    # print('Water --> x_T'+id)
    # print(x)
    # print(g1)
    return g1

jit(nopython=True)
def gT_min(x,d,id,min,log): #Lower and Higher Water Level -  x em x minutos
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
    if(d.ftype==2):
        g1=h_tmin(d,['tank'+str(id)+'_h'],timeInc,min)        

    elif(d.ftype==3):
        flag_sol=0
        if(len(log.x_round)!=0):
            roundx=round_x(x,d)
            try:
                idx=log.x_round.index(roundx)
            except ValueError:
                idx=-1

            if(idx!=-1):
                flag_sol=1
                tanks=log.tanks[idx]
                timeInc=log.timeInc[idx]
                log.n_tank+=1

        if(flag_sol==0):
            d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
            log(x,tanks,timeInc,d)
            
        # de x em x minutos
        g1=h_tmin(d,tanks['tank'+str(id)+'_h'],timeInc,min)     

    elif(d.ftype==1):
        g1=h_tmin(d,tanks['tank'+str(id)+'_h'],timeInc,min)
    
    # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as x_g:
    #     np.savetxt(x_g,x*np.ones((1,len(x))),delimiter=";") 

    # with open(r'Data Files\x_g1_T'+str(id)+'.csv','ab') as c:
    #     np.savetxt(c,np.ones((1,1))*g1,delimiter=";") 
    
    # print('Water --> x_T'+id)
    # print(x)
    # print(g1)
    return g1

jit(nopython=True)
def gDomain(x): # Variables domain
    #print('g3')
    #print('x')
    #print(x)
    return x

jit(nopython=True)
def g_TempLog_correction(x,d): #correção de x0
    x_new=[]
    for p in range(0,d.n_pumps): #d.n_pumps
        x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
        
        for i in range(0,d.n_dc[p]-1):
            dif=x_p[i+1]-(x_p[i]+x_p[i+d.n_dc[p]])
            if(dif<=d.dif_DC):
                x_p[i+d.n_dc[p]]=x_p[i+d.n_dc[p]]-d.dif_DC
        
        dif=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
        if(dif<=d.dif_DC):
            x_p[int(2*d.n_dc[p]-1)]=x_p[int(2*d.n_dc[p]-1)]-d.dif_DC
        x_new=np.concatenate((x_new,x_p))

    return x_new

jit(nopython=True)
def g_TempLog(x,d): #tstart(n+1) > tstop(n)  (várias bombas)
    # print('Temporal Logic Const. --> x(start-stop)')
    g5=[]
    for p in range(0,len(d.pumps_to_opt)): #d.n_pumps
        g5_F33=np.zeros(d.n_dc[p])
        x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
        
        if(d.n_dc[p]!=1):
            for i in range(0,d.n_dc[p]-1):
                g5_F33[i]=x_p[i+1]-(x_p[i]+x_p[i+d.n_dc[p]])
            g5_F33[i+1]=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
        else:
            g5_F33[0]=24-(x_p[d.n_dc[p]-1] + x_p[int(2*d.n_dc[p]-1)]) # garantir que a ultima duração não é superior a T
                  
        g5=np.concatenate((g5,g5_F33))

    return g5

jit(nopython=True)
def gT_cont(x,d,id,log): # restrição de continuidade do nivel dos tanques
    flag_sol=0
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
           log.n_tank+=1

    if(flag_sol==0):
        d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
        log(x,tanks,timeInc,d)

    g_end=tanks['tank'+str(id)+'_h'][-1] 
    g=g_end#d.h0[id]-g_end

    # print(x)
    # print(g)
    return g

jit(nopython=True)           
def gT_cont_perc(x,d,id,perc,log): # restrição de continuidade do nivel dos tanques -> percentagem a cima do hmin
    flag_sol=0
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
           log.n_tank+=1

    if(flag_sol==0):
        d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
        log(x,tanks,timeInc,d)

    g_end=tanks['tank'+str(id)+'_h'][-1] 
    g_lim=d.hmin[id]+((d.hmax[id]-d.hmin[id])*perc)
    g=g_lim-g_end
    return g

jit(nopython=True)
def jac_TempLog(x,d):
    # V2 1
    # eps_aux=AF.eps_definition_F3(x,d) 
    # jac=approx_fprime(x, g5_F3, eps_aux)
    
    n_var_pump=np.multiply(d.n_dc,2) #numero de variaveis por bomba
    for p in range(0,len(d.pumps_to_opt)):
        matriz1 = np.zeros((d.n_dc[p], d.n_dc[p]), dtype=int)  
        matriz2 = np.zeros((d.n_dc[p], d.n_dc[p]), dtype=int)  
        for i in range(0,d.n_dc[p]):
            matriz1[i][i] = -1.  
            matriz2[i][i] = -1.  
            if(i!=d.n_dc[p]-1):
                matriz1[i][i+1] = 1.  
        jac_aux=np.concatenate((matriz1,matriz2), axis=1)
        
        if(len(d.pumps_to_opt)!=1):
            if(p==0):
                matriz_d=np.zeros((d.n_dc[p], sum(n_var_pump[p+1:len(n_var_pump)])), dtype=int)
                jac=np.concatenate((jac_aux,matriz_d), axis=1)
            
            elif(p==len(d.pumps_to_opt)-1):
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

    #if(len(x)>d.dc_pos[len(d.pumps_to_opt)]): #VSP
    #    jac_vsp=np.zeros((len(jac), sum(d.n_dc)), dtype=int)  
    #    jac=np.concatenate((jac,jac_vsp),axis=1)
    
    # mod=np.linalg.norm(jac)
    return jac


jit(nopython=True)
def jac_gT(x,d,id,log):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)     

    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)  

    #x = x.detach().numpy() if isinstance(x, torch.Tensor) else x
    
    #eps_aux = eps_aux.detach().numpy() if isinstance(eps_aux, torch.Tensor) else eps_aux
    

    jac=approx_fprime(x, gT, eps_aux,*(d,id,log))
    
    
    return jac


jit(nopython=True)
def jac_gT_id(x,d,id_t, id_p,log):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)     

    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)  

    jac=approx_fprime(x, gT_id, eps_aux,*(d,id_t,id_p,log))
    return (jac)

jit(nopython=True)
def jac_gT_min(x,d,id,min,log):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)   

    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)  
      
    jac=approx_fprime(x, gT_min, eps_aux,*(d,id,min,log))
    
    return (jac)

jit(nopython=True)
def jac_gT_DC_min(x,d,id,min):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)   

    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)  
      
    jac=approx_fprime(x, gT_DC_min, eps_aux,*(d,id,min))
    return (jac)

jit(nopython=True)
def jac_WTL(x,eps_aux,id_tank,d): #NOT FINISHED
    g=gT(x,d,id_tank)
    x_pump=[]
    for p in range(0,d.n_pumps):
        x_pump.append(x[d.dc_pos[p]:d.dc_pos[p+1]])

    eps_pump=[]
    for p in range(0,d.n_pumps):
        eps_pump.append(eps_aux[d.dc_pos[p]:d.dc_pos[p+1]])
        
    if(d.n_points_tank[id_tank]==0): # 
        n_points=(d.n_points_tank[id_tank]+2)*(sum(d.n_dc)) + 1
        jac=np.zeros((n_points,len(x)))
        idx=0
        for i in range(0,len(x_pump)): # para cada bomba
            jac_p=np.zeros((n_points,len(x_pump[i])))
            for j in range(0,d.n_dc[i]): # para cada DC 
                #STARTING TIME
                if(eps_pump[i][j]>0): # FORWARD
                    x_pump_pert=np.copy(x_pump)
                    x_i=x_pump_pert[i][j]
                    x_st=x_pump_pert[i][j+d.n_dc[i]]
                    x_pump_pert[i][j]+=eps_pump[i][j]

                    #PERTURBADO
                    # d_pert,pumps_pert,tanks_pert,pipes_pert,valves_pert,timeInc_pert,controls_epanet_pert=EPA_API.EpanetSimulation(np.concatenate(x_pump_pert),d,0)
                    # g_x_pert=h_red3_acordeao(np.concatenate(x_pump_pert),tanks_pert['tank'+str(id_tank)+'_h'],timeInc_pert,d_pert,d_pert.n_points_tank[id_tank])
                    g_x_pert=gT(np.concatenate(x_pump_pert),d,id_tank)

                    #ORIGINAL
                    new_st=x_pump_pert[i][j]
                    new_dur=x_pump_pert[i][j+d.n_dc[i]]-eps_pump[i][j]
                    
                    x_pump_orig=np.copy(x_pump)                    
                    x_pump_orig[i][j+d.n_dc[i]]=eps_pump[i][j] #alterar duração do já existente
                    x_new = np.insert(x_pump_orig[i],j+1,new_st) #adição do st1
                    x_new = np.insert(x_new,j+d.n_dc[i]+2,new_dur) #adição do dur1

                    d.n_dc[i]=d.n_dc[i]+1               
                    x_new = np.insert(x_new,j+2,(x_i+x_st+eps_pump[i][j])) #adição do st2
                    x_new = np.insert(x_new,j+d.n_dc[i]+3,0) #adição do dur2

                    d.n_dc[i]=d.n_dc[i]+1
                    d.dc_pos=np.concatenate(([0],np.cumsum(np.multiply(d.n_dc,2))))

                    x_pump_orig2=[]
                    for k in range(0,len(d.n_dc)):
                        if(k!=j):
                            x_pump_orig2.append(x_pump_orig[k])
                        else:
                            x_pump_orig2.append(x_new)

                    d_orig,pumps_orig,tanks_orig,pipes_orig,valves_orig,timeInc_orig,controls_epanet_orig=EPA_API.EpanetSimulation(np.concatenate(x_pump_orig2),d,0)
                    g_x_orig=h_red3_acordeao(x_new,tanks_orig['tank'+str(id_tank)+'_h'],timeInc_orig,d_orig,d_orig.n_points_tank[id_tank])
                    
                    #repor valores
                    d.n_dc[i]=d.n_dc[i]-2
                    d.dc_pos=np.concatenate(([0],np.cumsum(np.multiply(d.n_dc,2))))

                    # calculo "normal" das diferenças finitas
                    for k in range(0,len(g)):
                        dif_j=(g_x_pert[k]-g[k])/eps_pump[i][j] #(x_pert-x_orig)/(d_pert)
                        jac_p[k][j]=dif_j
                    
                    # correção dif
                    jac_p[j*2][j]=(g_x_pert[j*2]-g_x_orig[j*2+1])/eps_pump[i][j]
                    jac_p[j*2+1][j]=(g_x_pert[j*2+1]-g_x_orig[j*2+4])/eps_pump[i][j]

                else: # BACKWARD
                    a=1     
                
                #DURATION TIME                
        
        jac.append(jac_p)                           
    else: # avaliar pontos entre DCs
        jac=np.zeros((len(x)+1,len(x)))

    # for i in range(0,len(x)):
    #     if(eps_aux[i]>0):
    #         xdif=np.copy(x)
    #         xdif[i]=x[i]+eps_aux[i]        
    #         g_x=h_red3(xdif,tanks['tank'+str(id_tank)+'_h'],timeInc,d) #g_x mas nos tempos de xdif

        #     d1,pumps1,tanks1,pipes1,valves1,timeInc1,controls_epanet1=EPA_API.EpanetSimulation(xdif,d,1)
        #     g_xdif=h_red3(xdif,tanks1['tank'+str(id_tank)+'_h'],timeInc1,d) #g_xdif nos tempos de xdif

        #     for j in range(0,len(g_xdif)):
        #         dif_i=(g_xdif[j]-g_x[j])/eps_aux[i] #(x_pert-x_orig)/(d_pert)
        #         jac[j][i]=dif_i
        # else:
        #     xdif=np.copy(x)
        #     xdif[i]=x[i]+eps_aux[i]        
        #     g_x=h_red3(x,tanks['tank'+str(id_tank)+'_h'],timeInc,d) #g_x mas nos tempos de x

            # d1,pumps1,tanks1,pipes1,valves1,timeInc1,controls_epanet1=EPA_API.EpanetSimulation(xdif,d,1)
            # g_xdif=h_red3(x,tanks1['tank'+str(id_tank)+'_h'],timeInc1,d) #g_xdif nos tempos de x

            # for j in range(0,len(g_xdif)):
            #     dif_i=(g_xdif[j]-g_x[j])/eps_aux[i] #(x_pert-x_orig)/(d_pert)
            #     jac[j][i]=dif_i
            
    return jac

jit(nopython=True)
def jac_WTL_vinitial(x, eps_aux,id_tank,d): # versão inicial
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,1)
    jac=np.zeros((len(x)+1,len(x)))

    for i in range(0,len(x)):
        xdif=np.copy(x)
        xdif[i]=x[i]+eps_aux[i]        
        g_x=h_red3(xdif,tanks['tank'+str(id_tank)+'_h'],timeInc) #g_x mas nos tempos de xdif

        d1,pumps1,tanks1,pipes1,valves1,timeInc1,controls_epanet1=EPA_API.EpanetSimulation(xdif,d,1)
        g_xdif=h_red3(xdif,tanks1['tank'+str(id_tank)+'_h'],timeInc1) #g_xdif nos tempos de xdif

        for j in range(0,len(g_xdif)):
            dif_i=(g_xdif[j]-g_x[j])/eps_aux[i] #(x_pert-x_orig)/(d_pert)
            jac[j][i]=dif_i
    return jac

jit(nopython=True)
def jac_gT_cont(x,d,id,log):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)   

    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)  

    jac=approx_fprime(x,gT_cont, eps_aux,*(d,id,log))
    # mod=np.linalg.norm(jac)
    return (jac)

jit(nopython=True)
def jac_gT_cont_perc(x,d,id,perc,log):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)   

    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)  

    jac=approx_fprime(x,gT_cont_perc, eps_aux,*(d,id,perc,log))
    # mod=np.linalg.norm(jac)
    return (jac)

jit(nopython=True)
def grad_Cost(x,d,log,flag):
    if(d.ftype==3): #duty-cycles formulation
        eps_aux=eps_definition_F3(x,d)   
    elif(d.ftype==2): #continuous formulation
        eps_aux=eps_definition_F2(x,d)    
    grad=approx_fprime(x, Cost, eps_aux,*(d,log,0))
    return (grad)

######################################
######### OPTIMIZATION LOGS ##########
######################################

jit(nopython=True)
def round_x(x,d): #arredondar aos segundos    
    x_round=[]
    for p in range(len(d.pumps_to_opt)):
        x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
        x_aux=[math.floor(x_p[i]*3600 + 0.5) for i in range(0,d.n_dc[p])]
        for i in range(d.n_dc[p],2*d.n_dc[p]):
            x_aux.append(math.floor(x_aux[i-d.n_dc[p]] + (x_p[i]*3600) + 0.5))
        x_round=x_round+x_aux

    # x_round=[math.floor(x[i]*3600 + 0.5) for i in range(len(x))]    
    if(len(x)>2*np.sum(d.n_dc)): #VSP
        x_vel=x[d.dc_pos[len(d.pumps_to_opt)]:len(x)]
        n_var_dc=np.sum(np.multiply(d.n_dc,4)) # nº variáveis com M_ini e M_start
        if(len(x)==n_var_dc):
            vel_pos=np.concatenate(([0],2*np.cumsum(d.n_dc))) # velocidade no inicio e fim
        else:
            vel_pos=np.concatenate(([0],np.cumsum(d.n_dc))) 
        for p in range(0,len(d.pumps_to_opt)):
                x_v=x_vel[vel_pos[p]:vel_pos[p+1]]
                x_aux=[round(x_v[i], 6) for i in range(len(x_v))]
                x_round=x_round+x_aux
    return x_round


class OptimizationLog:
    def __init__(self):
        self.solutions = []
        self.x=[]
        self.cost_solutions=[]
        self.n_cost=0
        self.tanks=[]
        self.pumps=[]
        self.timeInc=[]
        self.n_tank=0
        #self.previous_objective = float('inf')
     
    def __call__(self,x,timeInc,tanks,pumps,CostT,d):
        
        x = x.detach().numpy() if isinstance(x, torch.Tensor) else x    
        
        x_r=round_x(x,d)          
        self.x.append(x.copy())
        self.solutions.append(x_r.copy())
        self.cost_solutions.append(CostT)
        self.timeInc.append(timeInc.copy())
        self.tanks.append(tanks.copy())     
        self.pumps.append(pumps.copy())                


class TanksOptimizationLog:
    def __init__(self):
        self.x_round = []
        self.x=[]
        self.tanks=[]
        self.timeInc=[]
        self.n_tank=0
        #self.previous_objective = float('inf')
     
    def __call__(self,x,tanks,timeInc,d):
        round=round_x(x,d)
        self.x_round.append(round.copy())
        self.timeInc.append(timeInc)
        self.tanks.append(tanks.copy())
        self.x.append(x.copy())
        
        
        
def level_plot(x, d):
    
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)

    return tanks, timeInc, pumps     
