import tkinter as tk
from tkinter import filedialog
import glob
import os

def selecionar_pasta():
    """Abre o diálogo e retorna o caminho da pasta e a lista de imagens nela."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    caminho_pasta = filedialog.askdirectory(
        title="Selecione a pasta com as imagens dos veículos"
    )
    
    if not caminho_pasta:
        return None, []
        
    # Busca por extensoes de imagem comuns na pasta selecionada
    tipos_busca = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    arquivos = []
    for tipo in tipos_busca:
        arquivos.extend(glob.glob(os.path.join(caminho_pasta, tipo)))
        
    return caminho_pasta, arquivos