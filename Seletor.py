import tkinter as tk
from tkinter import filedialog

def selecionar_imagem():
    """Abre o diálogo e retorna o caminho da imagem selecionada."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    caminho_imagem = filedialog.askopenfilename(
        title="Selecione a imagem do carro",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    return caminho_imagem