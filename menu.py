import cv2
import sys
from Seletor import selecionar_imagem
from Placa import buscar_placa
from Caractere import validar_caracteres

def main():
    # ==========================================
    # 1. SELETOR DE ARQUIVOS 
    # ==========================================
    caminho_imagem = selecionar_imagem()
    
    if not caminho_imagem:
        print("Nenhuma imagem selecionada. Encerrando...")
        sys.exit()

    img_colorida = cv2.imread(caminho_imagem)
    img_original = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # 2. BUSCA MACRO (Encontrar a Placa)
    # ==========================================
    img_debug_geometria, img_placa_recortada = buscar_placa(img_original, img_colorida)

    # ==========================================
    # 3. VALIDAR CARACTERES
    # ==========================================
    if img_placa_recortada is not None:
        placa_bin, caracteres_encontrados = validar_caracteres(img_placa_recortada)
        
        if caracteres_encontrados >= 5:
            print(f"\n[SUCESSO] Placa detectada: {caracteres_encontrados} caracteres validados estruturalmente.")
        else:
            print(f"\n[ATENÇÃO] Placa detectada, mas caracteres borrados/ilegíveis (Apenas {caracteres_encontrados} encontrados).")

        cv2.imshow("1. Veiculo com Bounding Box", img_debug_geometria)
        cv2.imshow("2. Matriz Recortada (A Placa)", img_placa_recortada)
        cv2.imshow("3. Binarizacao da Placa (Busca Letras)", placa_bin)

    else:
        print("\n[FALHA] Nenhuma geometria semelhante a uma placa foi encontrada nesta foto.")
        cv2.imshow("1. Tentativa de Busca", img_debug_geometria)

    print("Pressione a tecla '0' na janela da imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()