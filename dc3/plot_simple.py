import matplotlib.pyplot as plt
import numpy as np
import data_system
import OptimAuxFunctionsV2 as op
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import os


def plot_simple(results, epoch, args):

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tar_duration = [2, 4, 1, 2, 3, 12]
    tariff_value = [0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094]
    hours = np.arange(0, 24, 0.1)

    d = data_system.data_system([args["dc"]], [0])
    # roda sua simulação e retorna tanks, timeInc, pumps
    tanks, timeInc, pumps = op.level_plot(results, d)

    # monta figura
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.axhline(y=2, color="red", linestyle="--", linewidth=1, label="Nível 2")
    ax.axhline(y=8, color="green", linestyle="--", linewidth=1, label="Nível 8")
    ax.plot((timeInc["StartTime"] / 3600), tanks["tank0_h"][:-1])
    ax.set_title(f"Epoch: {str(epoch)} ")
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("Nível do tanque")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    os.makedirs("plots", exist_ok=True)
    filename = f"plot_simple_epochNr{str(epoch)}_dc{args['dc']}_{args.get('vector_format','')}_{now}.png"
    plt.savefig(os.path.join("plots", filename), dpi=300, bbox_inches="tight")

    # mostra figura
    #plt.show()
    # plt.close(fig)
