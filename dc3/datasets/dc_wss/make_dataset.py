# import EPANET_API as EPA_API
import numpy as np
import os
import sys
import pickle
import torch
import random

torch.set_default_dtype(torch.float32)
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import Problem_DC_WSS
from data_system import data_system
from tqdm import tqdm

import random


def generate_time(nDutyCycles: int) -> list[float]:
    """
    Gera `nDutyCycles` tempos distintos em [0, 23.9] com passo 0.1,
    garantindo espaçamento mínimo de 2 horas entre eles.
    Retorna a lista ordenada de floats.
    """
    while True:
        # amostra nDutyCycles valores inteiros em [0, 239]
        candidatos = random.sample(range(240), nDutyCycles)
        # converte para décimos (0.1h) e ordena
        times = sorted(x / 10 for x in candidatos)
        # verifica espaçamento mínimo de 2h entre vizinhos
        ok = all(times[i] >= times[i - 1] + 2 for i in range(1, nDutyCycles))
        if ok:
            return times


def generate_duration(time_list: list[float]) -> list[float]:
    """
    Para cada tempo em time_list, gera uma duração aleatória:
      - se time == 23, escolhe entre 0.1 e 1
      - senão, escolhe uniformemente entre 0.1 e min(5, próximo_tempo - tempo_atual)
    Retorna lista de mesmas dimensões de time_list.
    """
    durations = []
    for idx, t in enumerate(time_list):
        if t >= 23:
            d = random.choice([0.001, 1.0])
        else:
            next_t = time_list[idx + 1] if idx + 1 < len(time_list) else 23.0
            max_d = min(5.0, max(000.5, next_t - t))
            d = random.uniform(000.5, max_d)
        durations.append(d)
    return durations


def generate_dataset(nDutyCycles: int):
    

    qty_x = [10, 20, 30, 50, 100, 200, 500, 1000]
    nDutyCycles = nDutyCycles  # agora usaremos isso para tamanho de generate_time

    d = data_system([nDutyCycles], [0])

    for qty in tqdm(qty_x):
        X = []
        for _ in range(qty):
            times = generate_time(nDutyCycles)
            durations = generate_duration(times)
            X.append(times + durations)

        problem = Problem_DC_WSS(d, X)
        fname = f"dc_wss_dataset_dc{nDutyCycles}_ex{len(X)}_tt-dd"
        path = (".\\" if os.name == "nt" else "./") + fname
        with open(path, "wb") as f:
            pickle.dump(problem, f)



if __name__ == "__main__":
    # nDutyCycles = 2
    # generate_dataset(nDutyCycles)
    for i in range(3, 6):
        generate_dataset(i)