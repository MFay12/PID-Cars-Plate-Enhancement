import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import tkinter as tk
from tkinter import filedialog, messagebox

# ==========================================
# 1. SELETOR DE ARQUIVOS 
# ==========================================
# Esconde a janela principal do Tkinter, deixando só o pop-up de arquivo
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True) #Janela na frente

# Abre a janela do sistema operacional para o usuário escolher a foto
caminho_imagem = filedialog.askopenfilename(
    title="Selecione a imagem para análise",
    filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
)

# Se o usuário cancelar e fechar a janela, o programa encerra
if not caminho_imagem:
    print("Nenhuma imagem selecionada. Encerrando o sistema...")
    exit()

# Carrega a imagem já em escala de cinza
img_original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

if img_original is None:
    print("Erro: Não foi possível ler o arquivo de imagem.")
    exit()

print(f"Imagem carregada com sucesso: {caminho_imagem}")

# ==========================================
# 2. MOTOR DE PROCESSAMENTO (Escuridão, Topologia e Geometria)
# ==========================================

# Correção de Luz Inteligente (CLAHE)
# Realce local de contraste que evita o superdimensionamento de fontes de luz pontuais.
#Faz equalização de histograma por setores (8x8)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
img_equalizada = clahe.apply(img_original)
 
# Desfoque (Suavização Espacial)
# Reduz o ruído de alta frequência para evitar que o filtro direcional confunda granulação com bordas.
img_suavizada = cv2.GaussianBlur(img_equalizada, (5, 5), 0)

# Filtros Espaciais Direcionais (Sobel)
# Calcula as derivadas em X e Y para mapear transições bruscas de contraste (as bordas dos caracteres).
sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y)) 

# Binarização Global (Otsu)
# Isola a geometria de interesse descobrindo matematicamente o limiar ideal de corte entre fundo e borda.
limiar, img_otsu = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morfologia Matemática (Refino Topológico)
# Elemento estruturante de 3x3 pixels
# Passa um "pincel" 3x3
kernel = np.ones((3, 3), np.uint8)

# Abertura: Suprime ruídos isolados (poeira digital) gerados pelo alto ISO do sensor.
img_limpa = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)

# Fechamento: Restaura a integridade de componentes fragmentados pela ausência de luz.
img_final = cv2.morphologyEx(img_limpa, cv2.MORPH_CLOSE, kernel)

# Extração de Componentes Conexos (Busca de Contornos)
# A flag RETR_EXTERNAL ignora bordas internas para poupar processamento.
contornos, _ = cv2.findContours(img_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Gera a imagem base colorida para desenhar as caixas de diagnóstico
img_debug_geometria = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

# Filtragem Geométrica por Bounding Box
for contorno in contornos:
    x, y, largura, altura = cv2.boundingRect(contorno)
    
    area = largura * altura
    proporcao = largura / float(altura)
    
    # Validação Estrutural: 
    # Área mínima descarta lixo visual.
    # Proporção 0.2 a 0.9 garante que a forma é um retângulo vertical (padrão tipográfico de uma placa).
    if area > 150 and 0.2 < proporcao < 0.9:
        cv2.rectangle(img_debug_geometria, (x, y), (x + largura, y + altura), (0, 255, 0), 2)

# ==========================================
# 3. INTERFACE VISUAL 
# ==========================================
# Cria UMA ÚNICA janela e deixa um espaço no rodapé para os botões
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25) # Abre espaço pro botão embaixo
fig.suptitle("Análise e Processamento de Imagens", fontsize=16, fontweight='bold')

# Desenha a tela inicial (A Foto Original)
img_display = ax1.imshow(img_original, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Imagem Original')
ax1.axis('off')

# Desenha o histograma inicial
ax2.hist(img_original.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
ax2.set_title('Histograma Original (Acúmulo na Escuridão)')
ax2.set_xlabel('Tons de Cinza')
ax2.set_ylabel('Quantidade de Pixels')
ax2.grid(axis='y', alpha=0.3)

# --- CRIAÇÃO DA BARRINHA DE SELETOR ---
ax_seletor = plt.axes([0.3, 0.05, 0.4, 0.12]) 
seletor = RadioButtons(ax_seletor, ('Análise Inicial (Problema)', 'Processamento Escuridão'))

# --- A FUNÇÃO QUE TROCA AS IMAGENS AO CLICAR ---
def atualizar_tela(label):
    ax2.clear() 
    ax2.set_xlabel('Tons de Cinza')
    ax2.set_ylabel('Quantidade de Pixels')
    ax2.grid(axis='y', alpha=0.3)

    if label == 'Análise Inicial (Problema)':
        img_display.set_data(img_original)
        ax1.set_title('Imagem Original')
        ax2.hist(img_original.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
        ax2.set_title('Histograma Original (Acúmulo na Escuridão)')
    else:
        img_display.set_data(img_final)
        ax1.set_title(f'Geometria Isolada (Otsu Limiar: {int(limiar)})')
        ax2.hist(img_equalizada.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax2.set_title('Novo Histograma (Distribuição Corrigida)')

    fig.canvas.draw_idle() # Atualiza a tela sem piscar a janela inteira

# Conecta o clique no seletor com a função
seletor.on_clicked(atualizar_tela)


plt.show()