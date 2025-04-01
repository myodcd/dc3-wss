
import torch
import EPANET_API as EPA_API
import numpy as np
import math

from data_system import data_system


def linear_interpolation(x_values, y_values, x_prime):
    for i in range(len(x_values) - 1):
        if x_values[i] <= x_prime <= x_values[i + 1]:
            # Aplicar interpolação linear
            y_prime = y_values[i] + (y_values[i + 1] - y_values[i]) * (x_prime - x_values[i]) / (x_values[i + 1] - x_values[i])
            return (y_prime/100)
        
    raise ValueError("x_prime está fora do intervalo dos pontos fornecidos.")

def obj_fn(y): # ,d, pumps, tanks, pipes, valves, timeInc):
            # COM EFICIÊNCIA

    d = data_system([5], [0])   
    
    y = torch.tensor(y)
    
    d, pumps, tanks, pipes, valves, timeInc, controls_epanet = EPA_API.EpanetSimulation(
    y, d, 0
    )

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

def gT(y):
    
    d = data_system([5], [0])    
    #d = self.d
    # Lower and Higher Water Level

    g1_total = []

    y = torch.tensor(y)

    d, pumps, tanks, pipes, valves, timeInc, controls_epanet = EPA_API.EpanetSimulation(
    y, d, 0
    )
    g1 = []
    for i in range(len(d.dc_pos) - 1):
        ini = d.dc_pos[i]
        fin = d.dc_pos[i + 1]

        # por tratar-se de um único tank, o id é 0
        id = 0
        
        g1_aux = h_red3_acordeao(y[ini:fin], tanks['tank' + str(id) + '_h'], timeInc, d, 0)
        
        g1.append(g1_aux)  # Adiciona os valores, removendo o último elemento
    
    g1_total.append(g1[0])  
    
    print(g1_total)
        
    return g1_total



if __name__ == '__main__':

    #d = EPA_API.data_system([5], [0])
    y =   [  1.0000,  7.0000, 10.0000, 15.0000, 20.0000,  2.2417,  3.0000,  3.2678,
          1.2079,  3.9687 ]
    gT(y)
    print('- - -')
    print(obj_fn(y))
    
    