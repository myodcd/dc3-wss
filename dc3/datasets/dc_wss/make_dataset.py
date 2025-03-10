#import EPANET_API as EPA_API
import numpy as np
import os
import sys
import pickle
import torch
import random
torch.set_default_dtype(torch.float64)
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import Problem_DC_WSS
from data_system import data_system
from tqdm import tqdm

def generate_time():
    while True:
        numbers = random.sample(range(24), 5)  # Escolher 5 números aleatórios de 0 a 23
        numbers.sort()  # Ordenar os números em ordem crescente
        
        if all(numbers[i] >= numbers[i - 1] + 2 for i in range(1, 5)):
            return numbers

def generate_duration(i_list):
    durations = []
    
    for idx, i in enumerate(i_list):
        if i == 23:
            duration = random.choice([0.1, 1])
        else:
            next_i = i_list[idx + 1] if idx + 1 < len(i_list) else 23
            max_value = min(5, next_i - i)
            duration = random.uniform(0.1, max_value)  # Gera um valor aleatório entre 0.5 e o valor máximo
            
        rounded_duration = round(duration * 2) / 2  # Arredondar para o múltiplo de 0.5 mais próximo
        durations.append(rounded_duration)
    
    return durations

    

def main():

    nDutyCycles = 5
    # nInc = 1; nIncOpt = 2*nDutyCycles
    # timeHorizon = 24*3600; maxInc = timeHorizon/nInc
    # h0 = 4.0

    d = data_system([5], [0])

#    horario_funcionamento_bombas = [1, 8, 12, 18, 21]
#    duracao_funcionamento_bombas = [3, 3, 3, 2.5, 2.5]

#    x = horario_funcionamento_bombas + duracao_funcionamento_bombas

#    d, pumps, tanks, pipes, valves, timeInc, controls_epanet = EPA_API.EpanetSimulation(
#        x, d, 0
#    )

#    x = [
#        [1, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [1, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [1, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [1, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [1, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [1, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [9, 10, 12, 18, 21, 1, 1, 3, 2.5, 2.5 ],
#        [1, 10, 12, 18, 21, 1, 1, 3, 2.5, 2.5 ],
#        [2, 8, 12, 18, 21, 3, 3, 3, 2.5, 2.5 ],
#        [1, 10, 12, 18, 21, 1, 1, 3, 2.5, 2.5 ]
#    ]




    X = []


    for i in tqdm(range(100)):
        
        time = generate_time()

        duration = generate_duration(time)

        concatenated = time + duration
        
        X.append(concatenated)


    problem = Problem_DC_WSS(
        #d, pumps, tanks, pipes, valves, timeInc, controls_epanet, x
        
        d, X
    )

    if os.name == "nt":
        file_path = ".\\dc_wss_dataset_dc_{}_ex_{}".format(nDutyCycles, len(X))
    else:
        file_path = "./dc_wss_dataset_dc_{}_ex_{}".format(nDutyCycles, len(X))

    with open(file_path, "wb") as f:
        pickle.dump(problem, f)


if __name__ == "__main__":
    main()
