import EPANET_API as EPA_API
import numpy as np
import os
import sys
import pickle
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import Problem_DC_WSS
from data_system import data_system

                
def main():

    nDutyCycles= 6
    #nInc = 1; nIncOpt = 2*nDutyCycles
    #timeHorizon = 24*3600; maxInc = timeHorizon/nInc
    #h0 = 4.0
    
    d=data_system([5],[5])
    
    horario_funcionamento_bombas=[1, 8, 12, 18, 21]
    duracao_funcionamento_bombas=[3, 3, 3, 2.5, 2.5]
    
    x= horario_funcionamento_bombas + duracao_funcionamento_bombas
    
    

    
    d,pumps,tanks,pipes,valves,timeInc,controls_epanet=EPA_API.EpanetSimulation(x,d,0)
    
    
    
    x = [[1,8,12,14,19,3,3,3,2.5,2.5], 
         [2,5,9,13,22,3,3,3,2.5,2.5], 
         [3,8,10,14,21,3,3,3,2.5,0.5], 
         [1,8,12,18,21,3,2.5,3,2.5,2.5], 
         [1,8,12,16,19,3,3,3,2.5,2.5], 
         [4,8,12,14,21,3,1,3,2.5,2.5], 
         [1,8,12,15,19,3,1,3,0.5,0.5], 
         [3,6,12,18,22,1,3,3,2.5,2.5], 
         [1,8,12,17,20,1,3,3,2.5,1.5], 
         [1,8,10,17,20,1,1,1,2.5,2.5]]
    
    
    
    problem = Problem_DC_WSS(d,pumps,tanks,pipes,valves,timeInc,controls_epanet, x)
    
    if os.name == 'nt':
        file_path = ".\\dc_wss_dataset_dc_{}".format(nDutyCycles)
    else:
        file_path = "./dc_wss_dataset_dc_{}".format(nDutyCycles)
        
    with open(file_path, 'wb') as f:
        pickle.dump(problem, f)
    
    
    
    
if __name__ == '__main__':
    main()