import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_nonlinear(data, y_new):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Linhas para criar o grid de contornos
    x1 = np.linspace(-1, 4, 300)
    x2 = np.linspace(-1, 4, 300)

    # x1_ e x2_ serão os pontos resultantes do treinamento do modelo (y_new)
    x1_ = y_new[:, 0]  # Assumindo que y_new seja uma matriz com duas colunas
    x2_ = y_new[:, 1]

    # Criando uma grade de valores de x1 e x2
    x1_vec, x2_vec = np.meshgrid(x1, x2)

    # Calculando a função objetivo (passando x1_vec e x2_vec empacotados em X)
    X_vec = np.column_stack([x1_vec.ravel(), x2_vec.ravel()])
    z = data.obj_fn(X_vec)
    z = z.reshape(x1_vec.shape)  # Reshape para manter as dimensões 2D

    # Calculando os resíduos das equações (passando X_vec)
    eq_resid_vals = data.eq_resid(X_vec, 0)
    eq_resid_vals = eq_resid_vals.reshape(x1_vec.shape)  # Reshape para 2D

    # Calculando os resíduos de desigualdade (passando X_vec)
    ineq_resid_vals = data.ineq_resid(X_vec)
    
    # Verificando a forma de ineq_resid_vals
    print("Shape of ineq_resid_vals:", ineq_resid_vals.shape)  # Verificando a forma
    
    # Caso ineq_resid_vals seja 1D (uma única desigualdade)
    if ineq_resid_vals.ndim == 1:
        ineq_resid_vals1 = ineq_resid_vals.reshape(x1_vec.shape)  # Primeira desigualdade
        ineq_resid_vals2 = ineq_resid_vals1  # Caso não haja segunda desigualdade, pode ser a mesma
    else:
        # Caso ineq_resid_vals seja 2D (duas desigualdades)
        ineq_resid_vals1 = ineq_resid_vals[:, 0].reshape(x1_vec.shape)  # Primeira desigualdade
        ineq_resid_vals2 = ineq_resid_vals[:, 1].reshape(x1_vec.shape)  # Segunda desigualdade

    # Plotando o contorno da função objetivo
    cp = ax.contour(
        x1_vec, x2_vec, z, levels=np.linspace(-5, 200, 300), cmap="viridis"
    )
    ax.clabel(cp, fmt="%2.2f", inline=True)

    # Contornos para os resíduos de equações (vermelho) e desigualdades (azul)
    cg1 = ax.contour(x1_vec, x2_vec, eq_resid_vals, levels=[0], colors="red", linewidths=2)
    cg2_ineq1 = ax.contour(x1_vec, x2_vec, ineq_resid_vals1, levels=[0], colors="blue", linewidths=2)
    cg2_ineq2 = ax.contour(x1_vec, x2_vec, ineq_resid_vals2, levels=[0], colors="green", linewidths=2)

    # Plotando os pontos de y_new sobre os contornos
    scatter = ax.scatter(x1_, x2_, c=data.obj_fn(y_new), cmap='viridis', edgecolors='black', label='Pontos')

    # Adicionando barra de cores
    plt.colorbar(scatter, label="Objetivo")

    # Anotando os pontos com índices
    for i, (xi, xj) in enumerate(zip(x1_, x2_)):
        ax.annotate(f'{i}', (xi, xj), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='red')

    # Ponto inicial (exemplo: primeiro ponto de treinamento)
    x_init = data.trainX[0, 0]  # Pega o valor de x1 no ponto inicial
    x_init_2 = data.trainX[0, 1]  # Pega o valor de x2 no ponto inicial

    # Traçando a linha do ponto inicial até o ponto final (y_new)
    ax.plot([x_init, x1_[0]], [x_init_2, x2_[0]], color='black', linestyle='-', linewidth=2, label='Caminho do Treinamento')

    # Adicionando título e labels
    plt.title("Contours of the objective function with model output points")    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    # Adicionando legenda
    plt.legend()

    # Exibindo o gráfico
    plt.show()