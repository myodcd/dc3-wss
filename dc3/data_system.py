import numpy as np

class data_system:
    def __init__(self,n_dc,n_points_t):
        ### Dados simulação WSS ###
        self.n_pumps=1
        self.n_pipes=0
        self.n_tanks=1
        self.n_max_inc=0
        self.timeInc_s=0
        self.n_tariffs=0
        self.T_s=0        
        self.tariffpatern='Tariff' #nome do pattern no inp com o tarifário
        self.tariff_idx=0
        self.hmax=[2] 
        self.hmin=[8]
        self.h0=[4]   
        
        self.num_dc=4
          
        self.tar_beg=[2,4,1,2,3,12] # duração tarifas
        self.tar_end=[2,6,7,9,12,24] #tempo final tarifários
        self.tariff_value=[0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094] # valor tarifas
        self.ncontrols_idx=0 #nº de regras a ignorar, caso existem controlos a não mexer (valvulas) 
        self.flag_t_inicio=0 #flag para assinalar simulações que não começam as 0 horas (p.e. Van Zyl)        
        self.n_points_tank=n_points_t #nº de pontos a avaliar dentro de cada DC para cada tanque
        self.flag_pat=0

        self.EpanetFile='Bomba-deposito_v1.inp'  
        self.flag_t_inicio=0 #flag para assinalar simulações que não começam as 0 horas (p.e. Van Zyl) 
        self.nomerpt='report.rpt'
        self.nomebin='output.bin'

        ### Dados otimização ###    
        self.ftype=3
        #Formulação 1 - Binária
        self.t_hor_1F=[0.5 for i in range(0,int(24/0.5))]
        self.t_hor_s_1F=self.t_hor_1F*np.array([60*60])
        
        #Formulação 2 - Real Continuous
        self.t_hor_2F=np.array([2,4,1,2,3,12]) 
        self.t_hor_s_2F=self.t_hor_2F*np.array([60*60])

        #Formulação 3 - Duty-Cycles       
        self.epsF_i=0.01 #0.01
        self.epsF_d=0.01 #0.018
        self.dif_DC=8e-4 #diferença entre DC's (2 seg)
        self.n_dc=n_dc  #numeros de DC's por bomba 
        self.dc_pos=np.concatenate(([0],np.cumsum(np.multiply(self.n_dc,2)))) #posições das variaveis por bomba