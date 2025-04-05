# import EPANET_API as EPA_API
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
        # Gera 5 números decimais em [0, 23.9] com incremento de 0.1
        numbers = sorted([x / 10 for x in random.sample(range(240), 5)])
        # Verifica se a diferença mínima de 2 está garantida entre os números
        if all(numbers[i] >= numbers[i - 1] + 2 for i in range(1, 5)):
            return numbers


def generate_duration(i_list):
    durations = []

    for idx, i in enumerate(i_list):

        if i == 23:
            duration = random.choice([0.1, 1])
        else:
            next_i = i_list[idx + 1] if idx + 1 < len(i_list) else 23
            max_value = min(
                5, max(0.1, next_i - i)
            )  # Garante que o valor máximo esteja entre 0.1 e 5
            duration = random.uniform(
                0.1, max_value
            )  # Gera um valor aleatório dentro do intervalo ajustado

        durations.append(duration)

    return durations


def main():

    qty_x = [6, 8, 10, 20, 30, 50, 100, 200, 500, 1000]    

    nDutyCycles = 5

    d = data_system([5], [0])

    for qty in tqdm(qty_x):

        X = []

        for i in range(qty):

            time = generate_time()

            duration = generate_duration(time)

            concatenated = time + duration

            X.append(concatenated)

        # print(X)
        problem = Problem_DC_WSS(
            # d, pumps, tanks, pipes, valves, timeInc, controls_epanet, x
            d,
            X,
        )

        if os.name == "nt":
            file_path = ".\\dc_wss_dataset_dc{}_ex{}".format(nDutyCycles, len(X))
        else:
            file_path = "./dc_wss_dataset_dc{}_ex{}".format(nDutyCycles, len(X))

        with open(file_path, "wb") as f:
            pickle.dump(problem, f)


if __name__ == "__main__":
    main()
