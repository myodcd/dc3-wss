#import EPANET_API as EPA_API
import EPANET_API as EPA_API
import OptimAuxFunctions_v2 as AF
import numpy as np
from functools import partial
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import BFGS
import warnings

warnings.filterwarnings("ignore")

def Cost(x,chart,d):
    # print('OF --> x')
    # print(x)

    with open(r'x.csv','ab') as x_C:
        np.savetxt(x_C,x*np.ones((1,len(x))),delimiter=";") 
     
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
    
    cost_it=[]
    cost_acum=[]
    CostT=0
    for i in range (0,len(timeInc['StartTime'])):
        tariffpriceInc=(timeInc['duration'][i]/3600)*timeInc['Tariff'][i]
        Cost=tariffpriceInc*pumps['pump0_p'][i]
        cost_it.append(Cost)
        CostT += Cost
        cost_acum.append(CostT)

    # print(CostT)
    with open(r'Data Files\CostT.csv','ab') as c:
        np.savetxt(c,np.ones((1,1))*CostT,delimiter=";") 
    
    #if(chart==1):
    #    plot_chart(timeInc,pumps,tanks,valves,d)
        
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
        
def optimization(x0, d, alg,ftol,gtol): #,flag_opt_day, opt_day):  
    log_tank=AF.TanksOptimizationLog()
    log_cost=AF.CostOptimizationLog()
    jac_T=partial(AF.jac_gT,d=d,id=0,log=log_tank)        
    jac_DC=partial(AF.jac_TempLog,d=d)

    C1 = NonlinearConstraint(lambda x: AF.gT(x,d,0,log_tank), 2.0, 8.0, jac=jac_T, keep_feasible=False) #Water Level

    C_DC = NonlinearConstraint(lambda x: AF.g_TempLog(x,d), d.dif_DC, np.inf, hess=BFGS(), jac=jac_DC, keep_feasible=True) #AF.TempLog(d)


    bounds=Bounds([0 for i in range(0,len(x0))],[24 for i in range(0,len(x0))], keep_feasible=True);

    
    res=minimize(AF.Cost, x0, args=(d,log_cost,1), method='SLSQP', constraints=[C_DC,C1], bounds=bounds, jac=AF.grad_Cost, tol=gtol, options={'iprint':3,'maxiter':120, 'disp': True,'ftol':ftol})
    
    return res,log_cost,log_tank    

def main():

    nDutyCycles= 6; nInc = 1; nIncOpt = 2*nDutyCycles
    timeHorizon = 24*3600; maxInc = timeHorizon/nInc
    h0 = 4.0
    
    #d=data_system([5],[5])
    d=data_system([5],[0])
    horario_funcionamento_bombas=[1, 8, 12, 18, 21]
    duracao_funcionamento_bombas=[3, 3, 3, 2.5, 2.5]
    
    x= horario_funcionamento_bombas + duracao_funcionamento_bombas
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
    res=optimization(x,d,'SLSQP',0.05,0)
    
    #Cost(res.x, 1, d)
    print('RES ', res)
if __name__ == '__main__':
    main()