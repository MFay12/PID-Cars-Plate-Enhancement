import cv2

def validar_caracteres(img_placa_recortada):
    """Analisa os contornos da placa recortada para contar os caracteres válidos."""
    if img_placa_recortada is None:
        return None, 0

    _, placa_bin = cv2.threshold(img_placa_recortada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contornos_placa, _ = cv2.findContours(placa_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caracteres_encontrados = 0
    for c in contornos_placa:
        xc, yc, wc, hc = cv2.boundingRect(c)
        proporcao_c = wc / float(hc)
        
        if 0.2 < proporcao_c < 0.9 and (wc * hc) > 40:
            caracteres_encontrados += 1
            
    return placa_bin, caracteres_encontrados