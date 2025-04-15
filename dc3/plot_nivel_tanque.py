import matplotlib.pyplot as plt
import torch
import numpy as np
import datetime
import os


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def plot_nivel_tanque(output_data, gT, total_cost, args, save_plot=False):
    # Tariffs per period
    tar_duration = [2, 4, 1, 2, 3, 12]
    tariff_value = [0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094]

    # Continuous tariff vector
    horas = np.arange(0, 24, 0.1)
    tarifas_por_hora = np.zeros_like(horas)

    inicio = 0
    for dur, val in zip(tar_duration, tariff_value):
        fim = inicio + dur
        tarifas_por_hora[(horas >= inicio) & (horas < fim)] = val
        inicio = fim

    # Start times and durations
    horarios = output_data[:5]
    duracoes = output_data[5:]
    fim_bombas = horarios + duracoes

    # Convert gT from tensor to array
    gT = gT.numpy()[0]

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Tariff line
    ax1.plot(horas, tarifas_por_hora, "b-", label="Tariff per hour")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Tariff Cost (€)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # X-axis ticks
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xlim(0, 24)

    # Tank levels
     # Tank levels
    ax2 = ax1.twinx()

    # Linha pontilhada antes do primeiro acionamento
    first_start = horarios[0].item()
    first_nivel_ini = gT[0]  # Primeiro nível no início
    ax2.plot(
        [0, first_start],
        [first_nivel_ini, first_nivel_ini],
        "r--",
        label="Initial Level",
    )

    for i in range(5):
        start = horarios[i].item()
        end = fim_bombas[i].item()
        nivel_ini = gT[2 * i]
        nivel_fim = gT[2 * i + 1]

        # Solid line (pump operation)
        ax2.plot(
            [start, end],
            [nivel_ini, nivel_fim],
            "r-",
            label="Tank Level" if i == 0 else "",
        )

        # Dotted line between pump i and i+1
        if i < 4:
            next_start = horarios[i + 1].item()
            next_nivel_ini = gT[2 * (i + 1)]
            ax2.plot(
                [end, next_start],
                [nivel_fim, next_nivel_ini],
                "r--",
                label="Transition" if i == 0 else "",
            )

        # Vertical band for pump activity
        ax1.axvspan(
            start,
            end,
            color="green",
            alpha=0.2,
            label="Pump Operation" if i == 0 else "",
        )

    # Horizontal limit lines
    ax2.axhline(2, color="gray", linestyle="--", label="Minimum Level")
    ax2.axhline(8, color="black", linestyle="--", label="Maximum Level")
    ax2.set_ylabel("Tank Levels", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Title and legend
    fig.suptitle(f"Tariff, Pump Operation and Tank Levels - plot_tank_level_samples{args['qtySamples']}_epochs_{args['epochs']}_{now}")

    # Legend
    fig.legend(
        loc="upper left", bbox_to_anchor=(0.85, 0.85), bbox_transform=ax1.transAxes
    )

    # Total cost annotation
    ax1.text(
        0.95,
        0.05,
        f"Total Cost: € {total_cost:.2f}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="black",
        weight="bold",
    )

    plt.tight_layout()
    
    
    if save_plot:
        
        plt.savefig(
            os.path.join(
                "plots",
                f"plot_tank_level_samples{args['qtySamples']}_epochs_{args['epochs']}_{now}.png",
            )
        )

    plt.show()