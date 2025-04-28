import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# Lista de caminhos das imagens
image_paths = [
    
    
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr00_dc3_td-td_2025-04-28_22-42-13.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr13_dc3_td-td_2025-04-28_22-58-56.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr12_dc3_td-td_2025-04-28_22-57-38.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr11_dc3_td-td_2025-04-28_22-55-58.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr10_dc3_td-td_2025-04-28_22-54-55.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr09_dc3_td-td_2025-04-28_22-53-13.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr08_dc3_td-td_2025-04-28_22-52-09.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr07_dc3_td-td_2025-04-28_22-50-57.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr06_dc3_td-td_2025-04-28_22-49-41.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr05_dc3_td-td_2025-04-28_22-48-34.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr04_dc3_td-td_2025-04-28_22-47-21.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr03_dc3_td-td_2025-04-28_22-46-11.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr02_dc3_td-td_2025-04-28_22-44-53.png", 
r"C:\Users\mtcd\Documents\Codes\dc3-wss\dc3\plots\plot_simple_epochNr01_dc3_td-td_2025-04-28_22-43-47.png"
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
