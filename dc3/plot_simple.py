import matplotlib.pyplot as plt
import numpy as np
import data_system
import OptimAuxFunctionsV2 as op
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import datetime
import os


def plot_simple(results, iteration, args,y_steps=None, n_sample=000, title_comment=None, total_iteration=None, epoch=None):

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tar_duration = [2, 4, 1, 2, 3, 12]
    tariff_value = [0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094]
    hours = np.arange(0, 24, 0.1)

    d = data_system.data_system([args["dc"]], [0])
    # roda sua simulação e retorna tanks, timeInc, pumps
    tanks, timeInc, pumps = op.level_plot(results, d)

    # monta figura
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.xaxis.set_major_locator(MultipleLocator(1))  # ticks de hora em hora
    ax.yaxis.set_major_locator(MultipleLocator(0.5))  # ticks de 0.5 em 0.5 (eixo y)
    ax.axhline(y=2, color="red", linestyle="--", linewidth=1, label="Tank Level 2m")
    ax.axhline(y=8, color="green", linestyle="--", linewidth=1, label="Tank Level 8m")
    ax.plot((timeInc["StartTime"] / 3600), tanks["tank0_h"][:-1])
    formatted_y = np.array2string(
    np.array(results), precision=2, separator=', ', suppress_small=True, formatter={'float_kind':lambda x: "%.2f" % x}
    )
    formatted_y_steps = np.array2string(
    np.array(y_steps), precision=2, separator=', ', suppress_small=True, formatter={'float_kind':lambda x: "%.2f" % x}
    )    
    
    formatted_y_steps = formatted_y_steps[1:-1]
    
    iteration_ = '-' if total_iteration is None else f"{iteration} of {total_iteration}"
    #y_step = '-' if formatted_y_steps is '' else formatted_y_steps
    
    title = (
        f"{title_comment if title_comment else ''}\n"
        f"Sample: {n_sample} \n"
        f"Epoch: {'-' if epoch is None else epoch  } \n"
        f"Iteration: {iteration_} \n"
        f"Y: {formatted_y} \n"
        f"Y Step: {formatted_y_steps} "
        
    )
    
    ax.set_title(title, fontsize=10, pad=10)
    
    
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Water Tank Level (m)")
    ax.plot((timeInc['StartTime']/3600), pumps['pump0_s'], label='pump 0')
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)




    os.makedirs("plots", exist_ok=True)


    iteration_str = f"{iteration:02d}"  # ou use zfill ou format conforme preferir
    
    
    filename = f"plot_Y_sample_nr{n_sample}_iteration_nr{iteration_str}_dc{args['dc']}_{now}.png"
    
    plt.savefig(os.path.join("plots", filename), dpi=300, bbox_inches="tight")
    # mostra figura
    #plt.show()
    plt.close(fig)
