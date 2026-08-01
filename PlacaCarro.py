import cv2
import numpy as np

def buscar_placa_carro(img_original, img_colorida):
    img_sem_chuva = cv2.medianBlur(img_original, 3) 
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img_equalizada = clahe.apply(img_sem_chuva)
    img_suavizada = cv2.GaussianBlur(img_equalizada, (5, 5), 0)

    sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    img_sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    limiar, img_otsu = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    img_limpa = cv2.morphologyEx(img_otsu, cv2.MORPH_OPEN, kernel)
    img_final = cv2.morphologyEx(img_limpa, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(img_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_debug = img_colorida.copy()

    altura_img, largura_img = img_original.shape
    area_total = altura_img * largura_img
    centro_x_imagem = largura_img / 2 
    
    area_minima = area_total * 0.001 
    area_maxima = area_total * 0.04   

    lista_candidatos = []

    for contorno in contornos:
        x, y, largura, altura = cv2.boundingRect(contorno)
        area = largura * altura
        proporcao = largura / float(altura)
        
        if area_minima < area < area_maxima and 2.2 < proporcao < 4.0:
            roi_bordas = img_otsu[y:y+altura, x:x+largura]
            pixels_brancos = cv2.countNonZero(roi_bordas)
            densidade = pixels_brancos / float(area)
            
            if densidade > 0.15: 
                erro_proporcao = abs(proporcao - 3.1)
                peso_y = y / altura_img 
                centro_x_box = x + (largura / 2)
                erro_x = abs(centro_x_box - centro_x_imagem) / largura_img 
                
                score = (peso_y * densidade) / (erro_proporcao + erro_x + 0.1)
                lista_candidatos.append((score, x, y, largura, altura))

    lista_candidatos.sort(key=lambda item: item[0], reverse=True)
    top_5 = lista_candidatos[:5]

    # Desenha todos os candidatos como "pendentes" (laranja)
    for indice, candidato in enumerate(top_5):
        score, x, y, largura, altura = candidato
        cv2.rectangle(img_debug, (x, y), (x + largura, y + altura), (0, 165, 255), 1)
        cv2.putText(img_debug, f"Teste #{indice + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
    return img_debug, top_5