import matplotlib.pyplot as plt
import numpy as np
import os

def plot_nonlinear_2ineq_evolution(data, y1_new_history, y2_new_history, filename):
    
    fig, ax = plt.subplots(figsize=(6, 6))

    # Trajetória empacotada
    y_new_history = np.column_stack([y1_new_history, y2_new_history])

    # Grade para contornos
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)
    x1_vec, x2_vec = np.meshgrid(x1, x2)
    X_vec = np.column_stack([x1_vec.ravel(), x2_vec.ravel()])

    # Avaliação da função objetivo
    z = data.obj_fn(X_vec).reshape(x1_vec.shape)

    # Resíduos das restrições
    eq_resid_vals = data.eq_resid(X_vec, 0).reshape(x1_vec.shape)
    ineq_g1_vals = data.ineq_g1(X_vec).reshape(x1_vec.shape)
    ineq_g2_vals = data.ineq_g2(X_vec).reshape(x1_vec.shape)

    # Contorno da função objetivo
    cp = ax.contour(x1_vec, x2_vec, z, levels=np.linspace(-5, 200, 300), cmap="viridis")
    ax.clabel(cp, fmt="%2.2f", inline=True)

    # Contornos das restrições
    ax.contour(x1_vec, x2_vec, eq_resid_vals, levels=[0], colors="red", linewidths=2, linestyles="--", label="Restrição de igualdade")
    ax.contour(x1_vec, x2_vec, ineq_g1_vals, levels=[0], colors="blue", linewidths=2, label="Desigualdade g1")
    ax.contour(x1_vec, x2_vec, ineq_g2_vals, levels=[0], colors="purple", linewidths=2, label="Desigualdade g2")

    # Plot da trajetória
    ax.plot(y1_new_history, y2_new_history, linestyle='-', linewidth=2, color='orange')

    # Pontos inicial e final
    obj_initial = data.obj_fn(np.array([[y1_new_history[0], y2_new_history[0]]]))
    obj_final = data.obj_fn(np.array([[y1_new_history[-1], y2_new_history[-1]]]))
    ax.scatter(y1_new_history[0], y2_new_history[0], color='green', zorder=5, 
               label=f'Ponto Inicial: ({y1_new_history[0]:.2f}, {y2_new_history[0]:.2f}), Obj = {obj_initial.item():.2f}',
               s=100, edgecolors='black')
    ax.scatter(y1_new_history[-1], y2_new_history[-1], color='red', zorder=5, 
               label=f'Ponto Final: ({y1_new_history[-1]:.2f}, {y2_new_history[-1]:.2f}), Obj = {obj_final.item():.2f}',
               s=100, edgecolors='black')

    # Pontos intermediários com cor por valor da função objetivo
    scatter = ax.scatter(y1_new_history[1:], y2_new_history[1:], 
                         c=data.obj_fn(np.array([y1_new_history[1:], y2_new_history[1:]]).T),
                         cmap='viridis', edgecolors='black')

    plt.colorbar(scatter, label="Valor da função objetivo")
    plt.title("Contornos da função objetivo e restrições com trajetória de otimização")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(filename))
    plt.show()
