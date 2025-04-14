def plot_dc_wss(data, y1_history, y2_history, filename):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    fig, ax = plt.subplots(figsize=(6, 6))

    y_history = np.column_stack([y1_history, y2_history])
    x1 = np.linspace(-2, 4, 400)
    x2 = np.linspace(-2, 4, 400)
    x1_vec, x2_vec = np.meshgrid(x1, x2)
    X_vec = np.column_stack([x1_vec.ravel(), x2_vec.ravel()])

    # Função objetivo
    z = data.obj_fn(X_vec).detach().numpy().reshape(x1_vec.shape)

    # Restrições de desigualdade individualizadas
    gt_up = data.ineq_gt_up(X_vec, y_history).detach().numpy()
    gt_down = data.ineq_gt_down(X_vec, y_history).detach().numpy()
    g_temp = data.ineq_temp_log(y_history).detach().numpy()

    # Contornos
    cp = ax.contour(x1_vec, x2_vec, z, levels=np.linspace(z.min(), z.max(), 50), cmap="viridis")
    ax.clabel(cp, fmt="%.1f")

    # Contornos das desigualdades (nível = 0 indica a fronteira da viabilidade)
    for i in range(gt_up.shape[1]):
        ax.contour(x1_vec, x2_vec, gt_up[:, i].reshape(x1_vec.shape), levels=[0], colors='blue', linewidths=1.5, linestyles='--')

    for i in range(gt_down.shape[1]):
        ax.contour(x1_vec, x2_vec, gt_down[:, i].reshape(x1_vec.shape), levels=[0], colors='cyan', linewidths=1.5, linestyles='--')

    for i in range(g_temp.shape[1]):
        ax.contour(x1_vec, x2_vec, g_temp[:, i].reshape(x1_vec.shape), levels=[0], colors='purple', linewidths=1.5, linestyles='-.')

    # Trajetória da otimização
    ax.plot(y1_history, y2_history, color="orange", linewidth=2, label="Trajetória")

    # Pontos inicial e final
    y0 = np.array([[y1_history[0], y2_history[0]]])
    yf = np.array([[y1_history[-1], y2_history[-1]]])

    obj0 = data.obj_fn(y0).item()
    objf = data.obj_fn(yf).item()

    ax.scatter(*y0[0], color='green', s=100, edgecolors='black', label=f"Inicial (Obj={obj0:.2f})")
    ax.scatter(*yf[0], color='red', s=100, edgecolors='black', label=f"Final (Obj={objf:.2f})")

    # Ponto intermediário com colormap
    scatter = ax.scatter(y1_history[1:], y2_history[1:], 
                         c=data.obj_fn(np.column_stack([y1_history[1:], y2_history[1:]])).detach().numpy(), 
                         cmap='viridis', edgecolors='black')

    plt.colorbar(scatter, label="Valor da função objetivo")
    plt.title("Evolução da Otimização com Restrições")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(filename))
    plt.show()
