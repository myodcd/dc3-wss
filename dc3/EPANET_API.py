import epamodule as em
import math
import numpy as np
import pandas as pd
#import OptimAuxFunctions as AF
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit

jit(nopython=True)
def WSS_Components(d,n_links,n_nodes): 
    """
    Define the WSS structure to future collect the hydraulic simulation data.

    Parameters:
        d (class): WSS infos  
        n_links (int): No of WSS links
        n_nodes (int): No of WSS nodes

    Returns:
        d (class): WSS infos (updated)  

        pumps (dict): struture to save the pumps infos (i E {0,...,d.n_pumps-1})
                    - 'pumps_idx' (int): each pump idx on EPANET
                    - 'pumps_id' (str): each pump id (name) on EPANET
                    - 'pump'+str(I)+'_s' (int): to save the pump i status in each hydraulic time step (timeInc_s) 
                    - 'pump'+str(i)+'_q' (float): to save the pump i water flow in each timeInc_s
                    - 'pump'+str(i)+'_p' (float): to save the pump i hydraulic power (if sp==1: (Wp/eff) else Wp) in each timeInc_s
                    - 'pump'+str(i)+'_sp' (float): to save the pump i speed in each timeInc_s
                    - 'pump'+str(i)+'_h' (float): to save the pump i head in each timeInc_s

        tanks (dict): struture to save the tanks infos (i E {0,...,d.n_tanks-1}) 
                    - 'tanks_idx' (int): each tank idx on EPANET
                    - 'tanks_id' (str): each tank id (name) on EPANET
                    - 'pump'+str(i)+'_h' (float): to save the tank i water level in each timeInc_s            	    - 

        pipes (dict): struture to save the pipes infos (i E {0,...,d.n_pipes-1})
                    - 'pipes_idx' (int): each pipe idx on EPANET
                    - 'pipe'+str(i)+'_q' (float): to save the pipe i water flow in each timeInc_s

        valves (dict): struture to save the valves infos (i E {0,...,d.n_valves-1})
                    - 'valves_idx' (int): each valve idx on EPANET
                    - 'valve'+str(i)+'_s' (int): to save the valve i status in each timeInc_s
                    - 'valve'+str(i)+'_q' (float): to save the valve i water flow in each timeInc_s

        CP (dict):  struture to save infos about the consumption points and respective patterns 
                    - 'CPs_idx' (int): each consumption point idx on EPANET
                    - 'CPs_id' (int): each consumption point id (name) on EPANETs 
                    - 'patterns_idx' (int): each pattern idx on EPANET
                    - 'pattens_id' (int): each pattern id (name) on EPANET
    """

    pumps_idx = []
    pumps_id= []
    tanks_idx = []
    tanks_id = []
    valves_idx = []
    pipes_idx = []
    CPs_id=[]
    CPs_idx=[]
    patterns_id=[]
    patterns_idx=[]

    for i in range(1, n_links + 1):
        type = em.ENgetlinktype(i)
        if type == em.EN_PUMP:
            pumps_idx.append(i)
            pumps_id.append(em.ENgetlinkid(i))
        if type == em.EN_PIPE:
            pipes_idx.append(i)
        if type == em.EN_FCV:
            valves_idx.append(i)

    for i in range(1, n_nodes + 1):
        type = em.ENgetnodetype(i)
        if type == em.EN_TANK: #tanques
            tanks_idx.append(i)
            tanks_id.append(em.ENgetnodeid(i))
        if type == em.EN_JUNCTION: # pontos de consumo
            node_name=em.ENgetnodeid(i)
            if(str(node_name[0:1])=="b'P'"):
                CPs_idx.append(i)
                CPs_id.append(em.ENgetnodeid(i))
    
    #Patterns and Consumption Point Characterization 
    n_pattern=em.ENgetcount(em.EN_PATCOUNT)
    for i in range(1,n_pattern+1):
        patterns_id.append(em.ENgetpatternid(i))
        patterns_idx.append(i)

    d.tariff_idx=em.ENgetpatternindex(d.tariffpatern)   
    CP={'CPs_idx':CPs_idx, 'CPs_id':[x.decode('utf-8') for x in CPs_id], 'patterns_idx':patterns_idx, 'patterns_id':[x.decode('utf-8') for x in patterns_id]}

    #Pumps Characterization
    d.n_pumps=len(pumps_idx)
    d.n_max_inc=int((d.T_s/d.timeInc_s)*2*d.n_pumps) #numero maximo de incrementos
    pumps={'pumps_idx':pumps_idx, 'pumps_id':[x.decode('utf-8') for x in pumps_id]}
    for i in range(0,d.n_pumps):
        pumps["pump"+str(i)+"_s"]=[0 for j in range(d.n_max_inc)]
        pumps["pump"+str(i)+"_q"]=[0 for j in range(d.n_max_inc)]
        pumps["pump"+str(i)+"_p"]=[0 for j in range(d.n_max_inc)]
        pumps["pump"+str(i)+"_sp"]=[0 for j in range(d.n_max_inc)]
        pumps["pump"+str(i)+"_h"]=[0 for j in range(d.n_max_inc)]

    #Valves Characterization
    d.n_valves=len(valves_idx)
    valves={'valves_idx':valves_idx}
    for i in range(0,d.n_valves):
        valves["valve"+str(i)+"_q"]=[0 for j in range(d.n_max_inc)]        
        valves["valve"+str(i)+"_s"]=[0 for j in range(d.n_max_inc)]

    #Pipes Characterization
    d.n_pipes=len(pipes_idx)
    pipes={'pipes_idx':pipes_idx}
    for i in range(0,d.n_pipes):
        pipes["pipes_idx"][i]=pipes_idx[i]
        pipes["pipe"+str(i)+"_q"]=[0 for j in range(d.n_max_inc)]

    #Tanks Characterization
    d.n_tanks=len(tanks_idx)
    tanks={'tanks_idx': tanks_idx, 'tanks_id':[x.decode('utf-8') for x in tanks_id]}
    for i in range(0,d.n_tanks):
        tanks["tank"+str(i)+"_h"]=[0 for j in range(d.n_max_inc)]
  
    return d, pumps, tanks, pipes, valves, CP

jit(nopython=True)
def EpanetAPIData(d,pumps,tanks,pipes,valves): #Recolha de dados de simulação 
    """
    Colect the hydraulic simulation data.

    Parameters:
        d (class): WSS infos  
        pumps (dict): struture used to save the pumps infos 
        tanks (dict): struture used to save the tanks infos 
        pipes (dict): struture used to save the pipes infos 
        valves (dict): struture used to save the valves infos 

    Returns: d,pumps,tanks,pipes,valves,timeInc
        d (class): WSS infos (updated)  
        pumps (dict): updated pumps structure
        tanks (dict): updated tanks structure
        pipes (dict): updated pipes structure
        valves (dict): updated valves structure
        timeInc (dict): struture to save additional simulation infos
                    - 'n_inc' (int): number of simulation time steps
                    - 'StartTime' (int): initial time of each simulation time step (seconds)
                    - 'duration' (int): duration of each simulation time step (seconds)
                    - 'EndTime' (int): ending time of each simulation time step (seconds)
                    - 'Tariff' (int): energy cost applied in each simulation time step (€/kWh)
    """
    i_dur=[]
    t_next=1
    n_inc=0
    t_i=em.ENgettimeparam(em.EN_PATTERNSTART) 
    t_inicio=[]  

    if(d.flag_pat!=0):
        h0_definition(d,tanks)
        tariff_definition(d)

    while t_next>0:
        em.ENrunH()
        for i in range(d.n_pumps):  
            idx=pumps['pumps_idx'][i] 
            pumps['pump'+str(i)+'_sp'][n_inc]=em.ENgetlinkvalue(idx,em.EN_SETTING)      
            pumps['pump'+str(i)+'_q'][n_inc]=em.ENgetlinkvalue(idx,em.EN_FLOW)
            pumps['pump'+str(i)+'_p'][n_inc]=em.ENgetlinkvalue(idx,em.EN_ENERGY)        
            pumps['pump'+str(i)+'_s'][n_inc]=em.ENgetlinkvalue(idx,em.EN_STATUS)                 
            pumps['pump'+str(i)+'_h'][n_inc]=em.ENgetlinkvalue(idx,em.EN_HEADLOSS)

        for i in range(d.n_valves):
            idx=valves['valves_idx'][i]
            valves['valve'+str(i)+'_q'][n_inc]=em.ENgetlinkvalue(idx,em.EN_FLOW)
            valves['valve'+str(i)+'_s'][n_inc]=em.ENgetlinkvalue(idx, em.EN_STATUS)
      
        for i in range(d.n_pipes):
            idx=pipes['pipes_idx'][i]
            pipes["pipe"+str(i)+"_q"][n_inc]=em.ENgetlinkvalue(idx,em.EN_FLOW)

        for i in range(d.n_tanks): 
            idx=tanks["tanks_idx"][i]            
            #tanks["tank"+str(i)+"_elev"]=em.ENgetnodevalue(idx,em.EN_ELEVATION)
            tanks["tank"+str(i)+"_h"][n_inc]=(em.ENgetnodevalue(idx,em.EN_HEAD)-em.ENgetnodevalue(idx,em.EN_ELEVATION))

        n_inc+=1
        t_next=em.ENnextH()
        i_dur.append(t_next) 
        
        if(t_i==24*60*60 and d.flag_t_inicio!=0): # used for Van Zyl network-->simulation started at 7am)
            t_i=0
            t_inicio.append(t_i)
            t_i+=t_next
        else:
            t_inicio.append(t_i)
            t_i+=t_next

    #Limpeza de valores extra das estruturas
    i_dur=np.delete(i_dur,len(i_dur)-1) 
    data_dump_p=[i for i in range(len(i_dur),d.n_max_inc)]
    data_dump_t=[i for i in range(len(i_dur)+1,d.n_max_inc)]
    t_inicio=np.delete(t_inicio,len(t_inicio)-1) 
    
    for i in range(d.n_pumps):
        pumps['pump'+str(i)+'_s']=np.delete(pumps['pump'+str(i)+'_s'],data_dump_p)
        pumps['pump'+str(i)+'_q']=np.delete(pumps['pump'+str(i)+'_q'],data_dump_p)
        pumps['pump'+str(i)+'_p']=np.delete(pumps['pump'+str(i)+'_p'],data_dump_p)
        pumps['pump'+str(i)+'_sp']=np.delete(pumps['pump'+str(i)+'_sp'],data_dump_p)
        pumps['pump'+str(i)+'_h']=np.delete(pumps['pump'+str(i)+'_h'],data_dump_p)

    for i in range(d.n_valves):
        valves['valve'+str(i)+'_s']=np.delete(valves['valve'+str(i)+'_s'],data_dump_p)
        valves['valve'+str(i)+'_q']=np.delete(valves['valve'+str(i)+'_q'],data_dump_p)

    for i in range(d.n_pipes):
        pipes["pipe"+str(i)+"_q"]=np.delete(pipes["pipe"+str(i)+"_q"],data_dump_p)

    for i in range(d.n_tanks):
        tanks["tank"+str(i)+"_h"]=np.delete(tanks["tank"+str(i)+"_h"],data_dump_t)

    #Caracterização timeInc
    timeInc={'n_inc':len(i_dur) ,'StartTime':t_inicio, 'duration':i_dur, 'EndTime':t_inicio+i_dur, 'Tariff':np.zeros(len(i_dur))}
    for i in range(0,len(timeInc['Tariff'])):
        period=math.ceil((timeInc['EndTime'][i]/60)/d.n_tariffs)
        timeInc['Tariff'][i]=em.ENgetpatternvalue(d.tariff_idx,period) 

    return d,pumps,tanks,pipes,valves,timeInc

jit(nopython=True)
def EpanetSimulation(x,d,sim_step): #Simulação hidraulica utilizando o epamodule
    em.ENopen(d.EpanetFile, nomerpt=d.nomerpt, nomebin=d.nomebin) # abrir epanet
    if(sim_step!=0):
        em.ENsettimeparam(em.EN_HYDSTEP,sim_step) # definir hidraulic step a sim_step seg por causa das dif.finitas
    d.timeInc_s=em.ENgettimeparam(em.EN_HYDSTEP) # duração de cada incremento de simulação
    d.T_s=em.ENgettimeparam(0) # duração do total time horizon    
    d.n_tariffs=(em.ENgettimeparam(em.EN_PATTERNSTEP)/60) # duração das tarifas em minutos
    em.ENopenH() # abrir a simulação
    em.ENsetstatusreport(2)
    em.ENinitH(10) 
       
    #Get pump, pipes e tanks indexes
    n_links=em.ENgetcount(em.EN_LINKCOUNT)
    n_nodes=em.ENgetcount(em.EN_NODECOUNT)
    
    d,pumps,tanks,pipes,valves,CP=WSS_Components(d,n_links,n_nodes) #Definição das estruturas WSS

    if(d.flag_pat!=0):
        #ler patterns
        patterns=pattern_definition(d)        
        #definir patterns
        set_patterns(d,patterns,CP)
    
    # comenbtário abaixo pois apresentava erro por não utilizar VSP
    #if(len(x)>2*np.sum(d.n_dc)):
    #    speed=speed_definition(x,d,pumps) # dicionário para speed
    #else:
    #    speed=[]
    
    speed = []
    
    controls_epanet=conversor(x,pumps,d,speed) 

    #Obtenção de dados do EPANET
    d,pumps,tanks,pipes,valves,timeInc=EpanetAPIData(d,pumps,tanks,pipes,valves)
    
    if(d.flag_pat==1):
        d.EpanetFile=d.EpanetFile_new + d.path_day + '_pred.inp'
        em.ENsaveinpfile(d.EpanetFile)
    elif(d.flag_pat==2):     
        d.EpanetFile=d.EpanetFile_new + d.path_day + '_real.inp'
        em.ENsaveinpfile(d.EpanetFile)        
    
    em.ENcloseH()
    em.ENclose()      
    return d,pumps,tanks,pipes,valves,timeInc,controls_epanet

jit(nopython=True)
def speed_definition(x,d,pumps): #definição de dicionário para lidar com VSPs
    st=[]
    dc_end=[]
    pumpidx=[]
    s=x[d.dc_pos[d.n_pumps]:len(x)]
    speed=[round(s[i],4) for i in range(len(s))] #arredondar a 4 casas decimais
    speed_pos=np.concatenate(([0],np.cumsum(d.n_dc))) 
    sp=[]
    for p in range(0, d.n_pumps):
        dur_aux=x[d.dc_pos[p]:d.dc_pos[p+1]][d.n_dc[p]:2*d.n_dc[p]]
        st_aux=x[d.dc_pos[p]:d.dc_pos[p+1]][0:d.n_dc[p]]
        if(d.flag_t_inicio!=0):
            init_time=np.ones(len(dur_aux))*d.flag_t_inicio
            
            inicio = [x + z for x,z in zip(np.multiply(st_aux,(60*60)), np.multiply(init_time,(60*60)))] # inicio_sec, init_time
            aux2=np.where(np.array(inicio)>24*60*60)
            if(len(aux2[0])!=0): 
                for i in range(0,len(aux2[0])):
                    inicio[aux2[0][i]]=inicio[aux2[0][i]]-24*60*60
            inicio_sec=[math.floor(inicio[i] + 0.5) for i in range(len(inicio))]

            final_sec = [x + y  for x,y in zip(inicio_sec, np.multiply(dur_aux,(60*60)))] # inicio_sec(com o init_time e arredondado!), dur_sec
            aux2=np.where(np.array(final_sec)>24*60*60)
            if(len(aux2[0])!=0): 
                for i in range(0,len(aux2[0])):
                    final_sec[aux2[0][i]]=final_sec[aux2[0][i]]-24*60*60

            dc_end.append(final_sec) 
            st.append(inicio_sec) 
        else:
            st_sec=[math.floor((st_aux[i]*60*60) + 0.5) for i in range(len(st_aux))]
            final_sec = [math.floor(x + y + 0.5) for x, y in zip(np.multiply(dur_aux,(60*60)),st_sec)]           
            dc_end.append(final_sec) 
            st.append(st_sec) 

        pumpidx.append(np.ones(d.n_dc[p])*pumps['pumps_idx'][p]) 
        #f_aux=np.copy(speed[speed_pos[p]:speed_pos[p+1]])
        #f_arredondado=[int(vel * 100) / 100 for vel in f_aux]
        sp.append(speed[speed_pos[p]:speed_pos[p+1]]) 
    
    #ordenar
    st=np.concatenate(st)
    sp=np.concatenate(sp)
    dc_end=np.concatenate(dc_end)
    pumpidx=(np.concatenate(pumpidx)).astype(int)    
    sortst = np.argsort(st) #ordenação de tempos de inicio
    sortend = np.argsort(dc_end)

    #criar dicionario
    speed={'st_sec':st[sortst],  'endt_sec':dc_end[sortend], 'speed':sp[sortst], 'pumps_idx_st':pumpidx[sortst],'pumps_idx_end':pumpidx[sortend]}
    
    return speed

jit(nopython=True)
def conversor_v1_OLD(x,pumps,d,speed): #conversor variaveis de decisão --> pump status
    rule_id=d.ncontrols_idx+1 #caso existem controlos a não mexer (valvulas)
    n_controls=em.ENgetcount(em.EN_CONTROLCOUNT)
    controls={'t':[], 'controls':[], 'pump_idx':[]}

    if(d.ftype==1): #Formulação binária  (SEM VPSs)     
        End_time=np.cumsum(d.t_hor_s_1F)
        Start_time=np.concatenate(([0],End_time[0:len(End_time)-1]),axis=0)
        t_off=np.zeros(len(d.t_hor_s_1F))
        
        idx_pump=0
        for i in range(0,len(x),int(len(x)/d.n_pumps)): # 0, 48, 96
            idx=0
            for k in range(i,i+int(len(x)/d.n_pumps)):
                em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],float(x[k]),0,Start_time[idx])
                controls['t'].append(em.ENgetcontrol(rule_id)[4])
                controls['controls'].append(float(x[k]))
                rule_id+=1
                em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,End_time[idx])
                controls['t'].append(em.ENgetcontrol(rule_id)[4])
                controls['controls'].append(0.0)
                rule_id+=1  
                idx+=1    
            idx_pump+=1    
         
    elif(d.ftype==2): #Formulação Real-Continua (SEM VPSs)     
        End_time=np.cumsum(d.t_hor_s_2F)
        Start_time=np.concatenate(([0],End_time[0:len(End_time)-1]),axis=0)
        t_off=np.zeros(len(x))
        idx_pump=0
        for i in range(0,len(x),int(len(x)/d.n_pumps)): # 0, 9, 18
            idx=0
            for k in range(i,i+int(len(x)/d.n_pumps)):                
                if(x[k]!=0):
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],1.0,0,Start_time[idx])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(1.0)
                    rule_id+=1
                    t_off[k]=(x[k]*d.t_hor_s_2F[idx])+Start_time[idx]
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,t_off[k])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(0.0)
                    rule_id+=1
                else:
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,Start_time[idx])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(0.0)
                    rule_id+=1
                    t_off[k]=End_time[idx]
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,t_off[k])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(0.0)
                    rule_id+=1
                idx+=1
            idx_pump+=1
  
    elif(d.ftype==3): #Formulação Duty-Cycles
        if(len(x)>np.sum(d.n_dc)*2):
            for i in range(d.n_pumps):
                em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][i],0.0,0,0)   
                rule_id+=1
                
            time=np.concatenate((speed['st_sec'],speed['endt_sec']))
            sp=np.concatenate((speed['speed'],np.zeros(len(speed['speed']))))
            idx=np.concatenate((speed['pumps_idx_st'],speed['pumps_idx_end']))
            sort = np.argsort(time)
            time=time[sort]
            sp=sp[sort]
            idx=idx[sort]

            #Definição de regras para vel
            for i in range(len(time)):  
                em.ENsetcontrol(rule_id,em.EN_TIMER,idx[i],sp[i],0,time[i])  
                controls['t'].append(em.ENgetcontrol(rule_id)[4])
                controls['controls'].append(sp[i]) 
                controls['pump_idx'].append(idx[i]) 
                rule_id+=1            
        else:
            for p in range(0,d.n_pumps):
                time_epanet=[] #np.zeros(n_var_pump[p]+1)
                s_epanet=[] #np.zeros(n_var_pump[p]+1)
                x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
                
                time_epanet.append(0) #para caso o default da bomba seja ON 
                s_epanet.append(0)
                
                for i in range(0,d.n_dc[p]):
                    #inicio DC
                    start=x_p[i]*(3600)
                    time_epanet.append(math.floor(start + 0.5))
                    s_epanet.append(1)

                    #fim DC
                    # end=(x_p[i]+x_p[i+d.n_dc[p]])*3600
                    end=time_epanet[-1]+(x_p[i+d.n_dc[p]])*3600
                    time_epanet.append(math.floor(end + 0.5))
                    s_epanet.append(0)

                time_epanet=np.array(time_epanet)
                s_epanet=np.array(s_epanet)
                sort = np.argsort(time_epanet) 
                
                todel=[]
                out=np.where(sort[1:] < sort[:-1])[0]
                if (len(out)!=0):
                    for k in range(0,len(out)):
                        if(s_epanet[out[k]]==0 and s_epanet[out[k]+1]==1): #sobreposição
                            todel.append(out[k])
                            todel.append(out[k]+1)
                    s_epanet=np.delete(s_epanet,todel)
                    time_epanet=np.delete(time_epanet,todel)

                for i in range(0,len(s_epanet)-1): #correção de sobreposições
                    if(s_epanet[i]==s_epanet[i+1]):
                        if(s_epanet[i]==0):
                            time_epanet[i]=time_epanet[i+1]     
            
                for i in range(0,len(time_epanet)): #definição das regras
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][p],float(s_epanet[i]),0,time_epanet[i])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(int(s_epanet[i]))
                    controls['pump_idx'].append(pumps['pumps_idx'][p]) 
                    rule_id+=1  
            
    #Eliminar os restantes controlos
    if(rule_id<n_controls):
        for id in range(rule_id,n_controls+1):
            em.ENsetcontrol(id,em.EN_TIMER,pumps['pumps_idx'][0],float(0),0,(24*60*60))

    return controls

jit(nopython=True)
def conversor(x,pumps,d,speed): #conversor variaveis de decisão --> pump status
    rule_id=d.ncontrols_idx+1 #caso existem controlos a não mexer (valvulas)
    n_controls=em.ENgetcount(em.EN_CONTROLCOUNT)
    controls={'t':[], 'controls':[]}
    if(d.ftype==1): #Formulação binária      
        End_time=np.cumsum(d.t_hor_s_1F)
        Start_time=np.concatenate(([0],End_time[0:len(End_time)-1]),axis=0)
        t_off=np.zeros(len(d.t_hor_s_1F))
        
        idx_pump=0
        for i in range(0,len(x),int(len(x)/d.n_pumps)): 
            idx=0
            for k in range(i,i+int(len(x)/d.n_pumps)):
                em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],float(x[k]),0,Start_time[idx])
                controls['t'].append(em.ENgetcontrol(rule_id)[4])
                controls['controls'].append(float(x[k]))
                rule_id+=1
                em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,End_time[idx])
                controls['t'].append(em.ENgetcontrol(rule_id)[4])
                controls['controls'].append(0.0)
                rule_id+=1  
                idx+=1    
            idx_pump+=1    
         
    elif(d.ftype==2): #Formulação Real-Continua
        End_time=np.cumsum(d.t_hor_s_2F)
        Start_time=np.concatenate(([0],End_time[0:len(End_time)-1]),axis=0)
        t_off=np.zeros(len(x))
        
        idx_pump=0
        for i in range(0,len(x),int(len(x)/d.n_pumps)): # 0, 9, 18
            idx=0
            for k in range(i,i+int(len(x)/d.n_pumps)):                
                if(x[k]!=0):
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],1.0,0,Start_time[idx])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(1.0)
                    rule_id+=1
                    t_off[k]=(x[k]*d.t_hor_s_2F[idx])+Start_time[idx]
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,t_off[k])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(0.0)
                    rule_id+=1
                else:
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,Start_time[idx])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(0.0)
                    rule_id+=1
                    t_off[k]=End_time[idx]
                    em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][idx_pump],0.0,0,t_off[k])
                    controls['t'].append(em.ENgetcontrol(rule_id)[4])
                    controls['controls'].append(0.0)
                    rule_id+=1
                idx+=1
            idx_pump+=1
  
    elif(d.ftype==3): #Formulação Duty-Cycles
        n_var_pump=np.multiply(2,d.n_dc) #numero de variaveis por bomba
        for p in range(0,d.n_pumps):
            idx=0
            time_epanet=np.zeros(n_var_pump[p]+1)
            s_epanet=np.zeros(n_var_pump[p]+1)
            x_p=x[d.dc_pos[p]:d.dc_pos[p+1]]
            x_p=x_p.clone()
            # x_p=x[int(p*n_var_pump):int(n_var_pump*(p+1))]
            for i in range(0,int(len(time_epanet)/2)):
                if(x_p[i] < 0): x_p[i]=0 
                if(x_p[i] > 24): x_p[i]=24
                
                if(i==0): # para o caso: bombas tenham definido status initial a 1
                    time_epanet[idx]=0  
                    s_epanet[idx]=0
                    idx+=1
                start=x_p[i]*(3600)
                if(start%1>=0.5):
                    time_epanet[idx]=math.ceil(start)
                else:
                    time_epanet[idx]=round(float(start))
                s_epanet[idx]=1
                idx+=1
                
                #end=(x_p[i]+x_p[i+int(len(x_p)/2)])*3600
                end=((time_epanet[idx-1]/3600)+x_p[i+int(len(x_p)/2)])*3600
                if(end%1>=0.5):
                    time_epanet[idx]=math.ceil(end)
                else:
                    time_epanet[idx]=round(float(end))
                s_epanet[idx]=0
                idx+=1

            # t=g5_F3(x)
            # if(any(elemento > 0 for elemento in t)):
            #     print('aqui') 
             
            sort = np.argsort(time_epanet) #ordenação de tempos de inicio/paragem
            # # time_epanet=time_epanet[sort]
            # # s_epanet=s_epanet[sort]
            
            todel=[]
            out=np.where(sort[1:] < sort[:-1])[0]
            if (len(out)!=0):
                for k in range(0,len(out)):
                    if(s_epanet[out[k]]==0 and s_epanet[out[k]+1]==1): #sobreposição
                        todel.append(out[k])
                        todel.append(out[k]+1)
                s_epanet=np.delete(s_epanet,todel)
                time_epanet=np.delete(time_epanet,todel)

            for i in range(0,len(s_epanet)-1): #correção de sobreposições
                if(s_epanet[i]==s_epanet[i+1]):
                    if(s_epanet[i]==0):
                        time_epanet[i]=time_epanet[i+1]     
        
            for i in range(0,len(time_epanet)):
                em.ENsetcontrol(rule_id,em.EN_TIMER,pumps['pumps_idx'][p],float(s_epanet[i]),0,time_epanet[i])
                controls['t'].append(em.ENgetcontrol(rule_id)[4])
                controls['controls'].append(int(s_epanet[i]))
                rule_id+=1  
            
            # print('it')
            
        #Eliminar os restantes controlos
        if(rule_id<n_controls):
            for id in range(rule_id,n_controls+1):
                em.ENsetcontrol(id,em.EN_TIMER,pumps['pumps_idx'][p],float(0),0,(24*60*60))

    return controls

jit(nopython=True)
def pattern_definition(d):
    #output: dicionario com o idx dos patterns; valor dos patterns
    patterns={}
    for i in range(0,len(d.sheet_name)):
        df_origem = pd.read_excel(d.path_pat, sheet_name=d.sheet_name[i])
        time=df_origem.iloc[:, 0].values
        real=df_origem.iloc[:, 1].values
        pred=df_origem.iloc[:, 2].values
        patterns['time_pred_'+d.sheet_name[i]]=time[~np.isnan(pred)]  
        patterns['pred_'+d.sheet_name[i]]=pred[~np.isnan(pred)]

        patterns['time_real_'+d.sheet_name[i]]=time    

        patterns['real_'+d.sheet_name[i]]=real
        patterns['real_'+d.sheet_name[i]][np.isnan(real)]=0

    # data -> pd.to_datetime(time)      
    return patterns

jit(nopython=True)
def set_patterns(d,patterns,CP):
    target_date = pd.to_datetime(d.path_day).date()
    for i in range(0,len(d.sheet_name)):
        if(d.flag_pat==1): #pred
            tt=pd.to_datetime(patterns['time_pred_'+d.sheet_name[i]])
            time=tt[tt.date == target_date]
            pat=patterns['pred_'+d.sheet_name[i]][tt.date == target_date]
        else: # real
            tt=pd.to_datetime(patterns['time_real_'+d.sheet_name[i]])
            time=tt[tt.date == target_date]
            pat=patterns['real_'+d.sheet_name[i]][tt.date == target_date]
        
        pat_name='Pattern_PE_'+d.sheet_name[i]
        node_name='PE_'+d.sheet_name[i]

        if(d.flag_pat==2):
            pat,time=RealDataPre_Processing(pat, time, d.sheet_name[i])

        if(len(pat)!=12*24):
             print('\n ERROR:DATA IS MISSING')

        # definir pattern
        idx_pat=CP['patterns_id'].index(pat_name)
        em.ENsetpattern(CP['patterns_idx'][idx_pat], pat)
        # associar pattern a nó 
        #idx_node=CP['CPs_id'].index(node_name)
        #em.ENsetnodevalue(CP['CPs_idx'][idx_node], em.EN_SOURCEPAT, idx_pat)

jit(nopython=True)
def h0_definition(d,tanks):  
    for i in range(0,d.n_tanks):
        tank_name='R_Res.'+d.tanks_id[i]
        idx_tank=tanks['tanks_id'].index(tank_name)
        em.ENsetnodevalue(tanks['tanks_idx'][idx_tank], em.EN_TANKLEVEL, d.h0[i])

jit(nopython=True)
def tariff_definition(d):
    date_obj = datetime.strptime(d.path_day, "%Y-%m-%d")
    weekday = date_obj.strftime("%A")
    pat_tar=[]

    if(weekday=='Saturday'):
        for i in range(0,len(d.tariff_sat_time)):
            aux=np.multiply(np.ones(int(d.tariff_sat_time[i]/(5/60))),d.tariff_sat[i])
            pat_tar=np.concatenate((pat_tar,aux))
    elif(weekday=='Sunday'):
        for i in range(0,len(d.tariff_sun_time)):
            aux=np.multiply(np.ones(int(d.tariff_sun_time[i]/(5/60))),d.tariff_sun[i])
            pat_tar=np.concatenate((pat_tar,aux))
    else: # week days
        for i in range(0,len(d.tariff_week_time)):
            aux=np.multiply(np.ones(int(d.tariff_week_time[i]/(5/60))),d.tariff_week[i])
            pat_tar=np.concatenate((pat_tar,aux))

    em.ENsetpattern(d.tariff_idx, pat_tar)

jit(nopython=True)
def RealDataPre_Processing(pat, time, pat_name):
    start_of_day = time[0].normalize()
    time_series = pd.DataFrame({
                'Time': (time - start_of_day).total_seconds(),  
                'Pattern': pat}) 

    idx=np.where(pat<0.5)
    if(len(idx[0])!=0):
        time_series['Pattern'][idx[0]]=0

    pattern_time=[i for i in range(0, (60*60*24), 5*60)]

    interpolator = interp1d(time_series['Time'], time_series['Pattern'], kind='linear', fill_value="extrapolate")
    interpolated_pattern = interpolator(pattern_time)

    # fig1, ax = plt.subplots(1)     
    # ax.plot(np.array(pattern_time)/3600,interpolated_pattern,label='processed data ('+pat_name+')')            
    # ax.plot(np.array(time_series['Time'])/3600,time_series['Pattern'],label='raw data ('+pat_name+')')
    # ax.legend(loc=1,fontsize=9)
    # fig1.savefig(pat_name+'_'+str(time[0].day)+'.pdf',format='pdf')
    # plt.show()

    return interpolated_pattern, pattern_time


