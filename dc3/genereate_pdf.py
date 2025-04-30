import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

image_paths = [
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr18_dc3_td-td_2025-04-29_10-46-17.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr17_dc3_td-td_2025-04-29_10-44-31.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr16_dc3_td-td_2025-04-29_10-42-29.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr15_dc3_td-td_2025-04-29_10-40-42.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr14_dc3_td-td_2025-04-29_10-39-17.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr13_dc3_td-td_2025-04-29_10-37-40.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr12_dc3_td-td_2025-04-29_10-35-45.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr11_dc3_td-td_2025-04-29_10-34-01.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr10_dc3_td-td_2025-04-29_10-32-28.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr9_dc3_td-td_2025-04-29_10-30-45.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr8_dc3_td-td_2025-04-29_10-28-35.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr7_dc3_td-td_2025-04-29_10-26-25.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr6_dc3_td-td_2025-04-29_10-24-41.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr5_dc3_td-td_2025-04-29_10-23-09.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr4_dc3_td-td_2025-04-29_10-21-18.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr3_dc3_td-td_2025-04-29_10-19-52.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr2_dc3_td-td_2025-04-29_10-18-27.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr1_dc3_td-td_2025-04-29_10-16-43.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr0_dc3_td-td_2025-04-29_10-15-04.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr29_dc3_td-td_2025-04-29_11-07-57.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr28_dc3_td-td_2025-04-29_11-06-08.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr27_dc3_td-td_2025-04-29_11-03-40.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr26_dc3_td-td_2025-04-29_11-01-18.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr25_dc3_td-td_2025-04-29_10-59-00.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr24_dc3_td-td_2025-04-29_10-57-03.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr23_dc3_td-td_2025-04-29_10-55-05.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr22_dc3_td-td_2025-04-29_10-53-10.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr21_dc3_td-td_2025-04-29_10-51-24.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr20_dc3_td-td_2025-04-29_10-49-28.png",
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr19_dc3_td-td_2025-04-29_10-47-46.png"
]


image_paths = sorted(image_paths)

# Criar o PDF
with PdfPages(
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\todos_os_graficos.pdf"
) as pdf:
    plots_por_linha = 4
    num_plots = len(image_paths)
    linhas = (num_plots + plots_por_linha - 1) // plots_por_linha  # arredonda para cima

    fig, axs = plt.subplots(
        linhas, plots_por_linha, figsize=(plots_por_linha * 5, linhas * 4)
    )
    axs = axs.flatten()  # transforma para lista para iterar fácil

    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(f"Gráfico {i}")

    # Desliga eixos vazios se sobrarem
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    pdf.savefig(fig)
