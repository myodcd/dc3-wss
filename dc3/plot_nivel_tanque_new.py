import torch
import data_system
import OptimAuxFunctionsV2 as op
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

def plot_nivel_tanque_new(args,
                          example,
                          total_cost: float,
                          save_plot: bool = False,
                          show: bool = False,
                          title=None,
                          sample=None
                          ):
    """
    Plots pump schedule, tank level, and tariff curve,
    with grid lines every 0.5 units on both axes.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- prepare data ---
    d = data_system.data_system([args['dc']], [0])
    ex = (example.detach().cpu().numpy().reshape(-1)
          if torch.is_tensor(example) else example)
    tanks, timeInc, pumps = op.level_plot(ex, d)
    time_h = timeInc['StartTime'] / 3600
    tank_levels = tanks['tank0_h'][:-1]

    # --- build tariff vector ---
    tar_duration  = [2, 4, 1, 2, 3, 12]
    tariff_value  = [0.0737, 0.06618, 0.0737, 0.10094, 0.18581, 0.10094]
    hours         = np.arange(0, 24, 0.1)
    tariff_by_hour = np.zeros_like(hours)
    start = 0
    for duration, val in zip(tar_duration, tariff_value):
        end = start + duration
        mask = (hours >= start) & (hours < end)
        tariff_by_hour[mask] = val
        start = end

    # --- create figure and first axis ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # tariff curve (dashed green)
    ax1.plot(
        hours,
        tariff_by_hour,
        'g--',
        linewidth=1.5,
        label='Tariff (€ / h)'
    )
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Tariff (€)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, tariff_by_hour.max() * 1.2)

    # X-axis major ticks every 1h
    ax1.set_xticks(np.arange(0, 25, 1))

    # X-axis minor ticks every 0.5h + grids
    ax1.set_xticks(np.arange(0, 24.5, 0.5), minor=True)
    ax1.minorticks_on()
    ax1.grid(which='major', linestyle='-', linewidth=0.8, color='gray', alpha=0.7)
    ax1.grid(which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    # pump status (blue step)
    ax1.step(
        time_h,
        pumps['pump0_s'],
        where='post',
        color='b',
        label='Pump Status'
    )
    # ensure pump status fits in the same y-range
    ax1.set_ylim(-0.1, 1.1)

    # --- secondary axis: tank level ---
    ax2 = ax1.twinx()
    for i in range(len(time_h) - 1):
        t0, t1 = time_h[i], time_h[i + 1]
        h0, h1 = tank_levels[i], tank_levels[i + 1]
        style = '-' if h1 >= h0 else ':'
        ax2.plot(
            [t0, t1],
            [h0, h1],
            f'r{style}',
            label='Tank Level' if i == 0 else ''
        )

    # horizontal limit lines
    ax2.axhline(2, color='gray', linestyle='--', linewidth=1, label='Min Level')
    ax2.axhline(8, color='gray', linestyle='--', linewidth=1, label='Max Level')

    ax2.set_ylabel('Tank Level', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Y-axis minor ticks every 0.5 + grid on secondary axis
    ax2.set_yticks(np.arange(0, 10.5, 0.5), minor=True)
    ax2.minorticks_on()
    ax2.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    # title, legend, and cost annotation
    title = title or f"DC={args['dc']} • Samples={args['qtySamples']} • Epochs={args['epochs']}"
    fig.suptitle(title)

    fig.legend(
        loc='upper left',
        bbox_to_anchor=(0.85, 0.85),
        bbox_transform=ax1.transAxes
    )
    ax1.text(
        0.95,
        0.05,
        f'Total Cost: € {total_cost:.2f}',
        transform=ax1.transAxes,
        ha='right',
        va='bottom',
        fontsize=12,
        weight='bold'
    )

    plt.tight_layout()

    if save_plot:
        os.makedirs('plots', exist_ok=True)
        filename = f"plot_cost_nr_{str(sample)}_dc{args['dc']}_{now}.png"
        plt.savefig(
            os.path.join('plots', filename),
            dpi=300,
            bbox_inches='tight'
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
