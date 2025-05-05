import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

image_paths = [





r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_nr13_epochNr00_dc5_tt-dd_sw100_2025-05-05_18-50-37.png", r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_nr13_epochNr20_dc5_tt-dd_sw100_2025-05-05_18-52-02.png", r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_nr13_epochNr40_dc5_tt-dd_sw100_2025-05-05_18-53-32.png", r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_nr13_epochNr60_dc5_tt-dd_sw100_2025-05-05_18-55-18.png",
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_nr13_epochNr80_dc5_tt-dd_sw100_2025-05-05_18-56-38.png"


]

image_paths = sorted(image_paths)

# Criar o PDF
with PdfPages(
    r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plots_consolidate.pdf"
) as pdf:
    plots_por_linha = 2
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
