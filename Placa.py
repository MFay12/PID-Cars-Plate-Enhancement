import cv2
import numpy as np

def buscar_placa(img_original, img_colorida):
    """Processa a imagem para encontrar a região de interesse (a placa)."""

    # 1. Antirruído: Median Blur remove a chuva ANTES de realçar o contraste
    img_sem_chuva = cv2.medianBlur(img_original, 3) 

    # 2. Realce de Contraste
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img_equalizada = clahe.apply(img_sem_chuva)

    img_suavizada = cv2.GaussianBlur(img_equalizada, (5, 5), 0)

    # 3. Gradientes (Sobel)
    sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    img_sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 4. Binarização e Morfologia
    limiar, img_otsu = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    img_limpa = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
    img_final = cv2.morphologyEx(img_limpa, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(img_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_debug_geometria = img_colorida.copy()
    img_placa_recortada = None

    # Dimensões e centro da tela
    altura_img, largura_img = img_original.shape
    area_total = altura_img * largura_img
    centro_x_imagem = largura_img / 2 # Ponto central no eixo X
    
    # Limites rígidos de área (0.1% a 4%)
    area_minima = area_total * 0.001 
    area_maxima = area_total * 0.04   

    melhor_candidato = None
    maior_score = 0 

    for contorno in contornos:
        x, y, largura, altura = cv2.boundingRect(contorno)
        area = largura * altura
        proporcao = largura / float(altura)
        
        # Geometria base (Proporção da placa entre 2.2 e 4.0)
        if area_minima < area < area_maxima and 2.2 < proporcao < 2.6:
            cv2.rectangle(img_debug_geometria, (x, y), (x + largura, y + altura), (0, 0, 255), 1)
            
            # --- CÁLCULO DE SCORE AVANÇADO ---
            
            # 1. Quanto mais perto de 3.1 a proporção for, menor o erro.
            erro_proporcao = abs(proporcao - 3.1)
            
            # 2. Quanto mais pra baixo na foto (Y alto), maior o peso_y.
            peso_y = y / altura_img 
            
            # 3. Penalidade Lateral: Distância do centro do contorno para o centro da foto.
            centro_x_box = x + (largura / 2)
            erro_x = abs(centro_x_box - centro_x_imagem) / largura_img 
            
            # SCORE FINAL: Recompensa peso_y, mas divide pelos erros (proporção e lateralidade)
            # Soma 0.1 no divisor para evitar divisão por zero
            score = peso_y / (erro_proporcao + erro_x + 0.1)
            
            if score > maior_score:
                maior_score = score
                melhor_candidato = (x, y, largura, altura)

    # Recorte final
    if melhor_candidato:
        x, y, largura, altura = melhor_candidato
        cv2.rectangle(img_debug_geometria, (x, y), (x + largura, y + altura), (0, 255, 0), 3)
        img_placa_recortada = img_original[y:y+altura, x:x+largura]
        
    return img_debug_geometria, img_placa_recortada