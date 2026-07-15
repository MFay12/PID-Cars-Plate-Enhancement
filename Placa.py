import cv2
import numpy as np

def buscar_placa(img_original, img_colorida):
    """Processa a imagem para encontrar a região de interesse (a placa)."""
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img_equalizada = clahe.apply(img_original)
     
    img_suavizada = cv2.GaussianBlur(img_equalizada, (5, 5), 0)

    # Cálculo dos Gradientes e Normalização (evita overflow nas matrizes)
    sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    img_sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    limiar, img_otsu = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    img_limpa = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
    img_final = cv2.morphologyEx(img_limpa, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(img_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_debug_geometria = img_colorida.copy()
    img_placa_recortada = None

    altura_img, largura_img = img_original.shape

    melhor_candidato = None
    maior_score = 0 

    for contorno in contornos:
        x, y, largura, altura = cv2.boundingRect(contorno)
        area = largura * altura
        proporcao = largura / float(altura)
        
        if 4500 < area < 80000 and 2.5 < proporcao < 4.0:
            cv2.rectangle(img_debug_geometria, (x, y), (x + largura, y + altura), (0, 0, 255), 1)
            
            if y > (altura_img * 0.3):
                erro_proporcao = abs(proporcao - 3.1)
                score = y / (erro_proporcao + 0.05)
                
                if score > maior_score:
                    maior_score = score
                    melhor_candidato = (x, y, largura, altura)

    if melhor_candidato:
        x, y, largura, altura = melhor_candidato
        cv2.rectangle(img_debug_geometria, (x, y), (x + largura, y + altura), (0, 255, 0), 3)
        img_placa_recortada = img_original[y:y+altura, x:x+largura]
        
    return img_debug_geometria, img_placa_recortada