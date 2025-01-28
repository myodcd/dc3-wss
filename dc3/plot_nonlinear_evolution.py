import matplotlib.pyplot as plt
import numpy as np
import os

def plot_nonlinear_evolution(data, y1_new_history, y2_new_history, filename):
    
    fig, ax = plt.subplots(figsize=(6, 6))

    # Empacotando as coordenadas em uma única matriz
    y_new_history = np.column_stack([y1_new_history, y2_new_history])
    
    print('Y NEW ', y_new_history)
    print('Y NEW SHAPE', y_new_history.shape)

    # Definindo a grade de pontos para o gráfico de contorno
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)

    x1_vec, x2_vec = np.meshgrid(x1, x2)
    X_vec = np.column_stack([x1_vec.ravel(), x2_vec.ravel()])
    z = data.obj_fn(X_vec)  # Função objetivo
    z = z.reshape(x1_vec.shape)

    # Calculando as restrições de equações e desigualdades
    eq_resid_vals = data.eq_resid(X_vec, 0)
    eq_resid_vals = eq_resid_vals.reshape(x1_vec.shape)

    ineq_resid_vals = data.ineq_resid(X_vec)
    ineq_resid_vals = ineq_resid_vals.reshape(x1_vec.shape)

    # Gerando os contornos da função objetivo, equações e desigualdades
    cp = ax.contour(x1_vec, x2_vec, z, levels=np.linspace(-5, 200, 300), cmap="viridis")
    ax.clabel(cp, fmt="%2.2f", inline=True)

    cg1 = ax.contour(x1_vec, x2_vec, eq_resid_vals, levels=[0], colors="red", linewidths=2)
    cg2 = ax.contour(x1_vec, x2_vec, ineq_resid_vals, levels=[0], colors="blue", linewidths=2)

    # Trajetória geral
    trajectory_x1 = []
    trajectory_x2 = []

    # Plotando as trajetórias para todos os pontos em y1_new_history e y2_new_history
    for i in range(len(y1_new_history)):
        y1_new = y1_new_history[i]
        y2_new = y2_new_history[i]

        trajectory_x1.append(y1_new)  # Primeiro valor de y_new (y1_new)
        trajectory_x2.append(y2_new)  # Segundo valor de y_new (y2_new)

    # Plotando a trajetória completa
    ax.plot(trajectory_x1, trajectory_x2, linestyle='-', linewidth=2, color='orange')

    # Calculando os valores da função objetivo para os pontos inicial e final
    obj_initial = data.obj_fn(np.array([[y1_new_history[0], y2_new_history[0]]]))  # Ajuste aqui
    obj_final = data.obj_fn(np.array([[y1_new_history[-1], y2_new_history[-1]]]))  # Ajuste aqui

    # Extrair os valores escalares da função objetivo (assumindo que obj_fn retorna um array)
    obj_initial_value = obj_initial.item() if isinstance(obj_initial, np.ndarray) else obj_initial
    obj_final_value = obj_final.item() if isinstance(obj_final, np.ndarray) else obj_final

    # Plotando os pontos inicial e final com marcadores diferentes
    ax.scatter(trajectory_x1[0], trajectory_x2[0], color='green', zorder=5, 
               label=f'Ponto Inicial: ({y1_new_history[0]:.2f}, {y2_new_history[0]:.2f}), Obj = {obj_initial_value:.2f}', s=100, edgecolors='black')  # Ponto inicial
    ax.scatter(trajectory_x1[-1], trajectory_x2[-1], color='red', zorder=5, 
               label=f'Ponto Final: ({y1_new_history[-1]:.2f}, {y2_new_history[-1]:.2f}), Obj = {obj_final_value:.2f}', s=100, edgecolors='black')  # Ponto final

    # Barra de cores para os pontos intermediários (não é necessário para legendas)
    scatter = ax.scatter(trajectory_x1[1:], trajectory_x2[1:], 
                         c=data.obj_fn(np.array([trajectory_x1[1:], trajectory_x2[1:]]).T), 
                         cmap='viridis', edgecolors='black')

    # Barra de cores (caso haja)
    if scatter is not None:
        plt.colorbar(scatter, label="Objetivo")

    # Títulos e labels
    plt.title("Contours da função objetivo com pontos de saída do modelo")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    
    # Exibindo a legenda
    plt.legend()

    # Salvando o gráfico
    plt.savefig(os.path.join(filename))

    # Exibindo o gráfico
    plt.show()