import cv2
import sys
import os
from Seletor import selecionar_pasta
from Placa import buscar_placa
from Caractere import validar_caracteres

def main():
    print("==========================================")
    print(" PIPELINE DIRETA: BUSCA DE PLACAS")
    print("==========================================")
    
    caminho_pasta, arquivos = selecionar_pasta()
    
    if not caminho_pasta or not arquivos:
        print("Nenhuma pasta ou imagem selecionada. Encerrando...")
        sys.exit()

    pasta_destino = "./placas_recortadas"
    os.makedirs(pasta_destino, exist_ok=True)

    print(f"Foram encontradas {len(arquivos)} imagens na pasta.")
    contador = 1

    for caminho_imagem in arquivos:
        print(f"\nProcessando: {caminho_imagem}")
        img_colorida = cv2.imread(caminho_imagem)
        
        if img_colorida is None:
            continue

        # 1. Prepara a imagem cinza diretamente da foto original
        img_cinza = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)
        
        # 2. Envia a imagem inteira para a busca da placa
        img_debug_placa, img_placa_recortada = buscar_placa(img_cinza, img_colorida)

        if img_placa_recortada is None:
            print("[FALHA] Nenhuma placa encontrada na imagem inteira.")
            cv2.imshow("Debug Placa", img_debug_placa)
            
        else:
            # 3. Validação dos Caracteres
            placa_bin, caracteres = validar_caracteres(img_placa_recortada)
            
            if caracteres >= 5:
                print(f"[SUCESSO] Placa validada estruturalmente ({caracteres} caracteres).")
                nome_arquivo = f"placa_detectada_{contador}.jpg"
                caminho_salvamento = os.path.join(pasta_destino, nome_arquivo)
                
                cv2.imwrite(caminho_salvamento, img_placa_recortada)
                contador += 1
            else:
                print(f"[ATENÇÃO] Falso positivo ou placa ilegível (apenas {caracteres} formatações).")

            # Exibe o fluxo
            cv2.imshow("1. Busca na Foto", img_debug_placa)
            cv2.imshow("2. Recorte", img_placa_recortada)
            cv2.imshow("3. Letras", placa_bin)

        key = cv2.waitKey(0)
        if key == 27: # Tecla ESC
            break
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()