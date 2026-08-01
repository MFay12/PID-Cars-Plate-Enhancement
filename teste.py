import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sys

def selecionar_imagem():
    """Abre o diálogo e retorna o caminho da imagem selecionada."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    caminho_imagem = filedialog.askopenfilename(
        title="Selecione a imagem do veiculo para teste",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    return caminho_imagem

def buscar_veiculo_pdi(img_colorida):
    """Detecta a região de interesse (ROI) usando a Opção A: Downsampling."""
    
    # ==========================================
    # OPCAO A: Padronizacao de Resolucao (Downsampling)
    # ==========================================
    LARGURA_PADRAO = 800
    altura_original, largura_original = img_colorida.shape[:2]
    
    # Calcula a nova altura mantendo a proporcao original
    proporcao_redimensionamento = LARGURA_PADRAO / float(largura_original)
    altura_nova = int(altura_original * proporcao_redimensionamento)
    
    # Redimensiona a imagem
    img_padronizada = cv2.resize(img_colorida, (LARGURA_PADRAO, altura_nova), interpolation=cv2.INTER_AREA)
    
    # ==========================================
    # PDI Classico aplicado na imagem padronizada
    # ==========================================
    
    img_gray = cv2.cvtColor(img_padronizada, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    bordas = cv2.Canny(img_blur, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bordas_fechadas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(bordas_fechadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_debug_veiculo = img_padronizada.copy()
    img_veiculo_recortada = None
    maior_area = 0
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        
        # Filtro de area para matriz de 800px
        if area > 5000: 
            x, y, w, h = cv2.boundingRect(contorno)
            proporcao = float(w) / h
            
            # Filtro 2: Proporção flexibilizada (0.8 a 3.5) para cobrir carros em diagonal/cortados
            if 0.8 < proporcao < 3.5:
                if area > maior_area:
                    maior_area = area
                    
                    # Desenha a Bounding Box
                    cv2.rectangle(img_debug_veiculo, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Recorta a matriz para a proxima fase
                    img_veiculo_recortada = img_padronizada[y:y+h, x:x+w]
                    
    return img_debug_veiculo, img_veiculo_recortada, bordas_fechadas

def main():
    print("==========================================")
    print(" TESTE DE PIPELINE - OPCAO A (DOWNSAMPLING)")
    print("==========================================")
    
    caminho_imagem = selecionar_imagem()
    
    if not caminho_imagem:
        print("Nenhuma imagem selecionada. Encerrando...")
        sys.exit()

    print(f"Carregando imagem: {caminho_imagem}")
    img_colorida = cv2.imread(caminho_imagem)
    
    if img_colorida is None:
        print("[ERRO] Nao foi possivel ler a imagem. Verifique o arquivo.")
        sys.exit()

    # Processamento
    img_debug, img_recorte, bordas_fechadas = buscar_veiculo_pdi(img_colorida)

    # Exibicao
    cv2.imshow("Passo 1: Morfologia (Bordas Unidas)", bordas_fechadas)
    cv2.imshow("Passo 2: Veiculo Encontrado", img_debug)
    
    if img_recorte is not None:
        cv2.imshow("Passo 3: Recorte a ser enviado para Placa.py", img_recorte)
        print("\n[SUCESSO] Veiculo detectado com base nas proporcoes geometricas!")
    else:
        print("\n[FALHA] Nenhum veiculo encontrado com os parametros de area/proporcao atuais.")

    print("\nPressione a tecla '0' na janela da imagem para encerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()