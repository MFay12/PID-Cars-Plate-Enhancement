import cv2
import sys
import os
from Seletor import selecionar_pasta
from PlacaCarro import buscar_placa_carro
from PlacaMoto import buscar_placa_moto
from Caractere import validar_caracteres

def main():
    print("==========================================")
    print(" PIPELINE DIRETA: TOP 5 VALIDATIVO")
    print("==========================================")
    
    # Resolve os Falsos Positivos perguntando o escopo
    print("\nO que voce deseja procurar nesta pasta?")
    print("1 - Apenas Carros")
    print("2 - Apenas Motos")
    print("3 - Misturado (Tenta Carro, se falhar tenta Moto)")
    opcao_busca = input("Digite o numero da opcao (1, 2 ou 3): ")

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

        img_cinza = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY)
        sucesso = False

        # ==========================================
        # BUSCA DE CARROS
        # ==========================================
        if opcao_busca in ['1', '3']:
            img_debug, top_5_carros = buscar_placa_carro(img_cinza, img_colorida)
            
            # Testa o Top 5 um por um
            for indice, candidato in enumerate(top_5_carros):
                score, x, y, largura, altura = candidato
                img_recorte = img_cinza[y:y+altura, x:x+largura]
                placa_bin, caracteres = validar_caracteres(img_recorte)
                
                if caracteres >= 5:
                    sucesso = True
                    # Desenha a caixa verde da Vitoria
                    cv2.rectangle(img_debug, (x, y), (x + largura, y + altura), (0, 255, 0), 3)
                    cv2.putText(img_debug, "VENCEDOR", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    print(f"[SUCESSO] Placa de CARRO no candidato #{indice+1} ({caracteres} caracteres).")
                    cv2.imshow("Debug Geral", img_debug)
                    cv2.imshow("Recorte Vencedor", img_recorte)
                    cv2.imshow("Caracteres", placa_bin)
                    
                    cv2.imwrite(os.path.join(pasta_destino, f"placa_carro_{contador}.jpg"), img_recorte)
                    contador += 1
                    break # Para de testar os outros candidatos desta foto

            if not sucesso and opcao_busca == '1':
                print("[FALHA] Nenhum dos 5 candidatos de carro passou no teste de letras.")
                cv2.imshow("Debug Geral", img_debug)

        # ==========================================
        # BUSCA DE MOTOS
        # ==========================================
        if not sucesso and opcao_busca in ['2', '3']:
            img_debug, top_5_motos = buscar_placa_moto(img_cinza, img_colorida)
            
            for indice, candidato in enumerate(top_5_motos):
                score, x, y, largura, altura = candidato
                img_recorte = img_cinza[y:y+altura, x:x+largura]
                placa_bin, caracteres = validar_caracteres(img_recorte)
                
                if caracteres >= 5:
                    sucesso = True
                    cv2.rectangle(img_debug, (x, y), (x + largura, y + altura), (255, 0, 0), 3)
                    cv2.putText(img_debug, "VENCEDOR", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    print(f"[SUCESSO] Placa de MOTO no candidato #{indice+1} ({caracteres} caracteres).")
                    cv2.imshow("Debug Geral", img_debug)
                    cv2.imshow("Recorte Vencedor", img_recorte)
                    cv2.imshow("Caracteres", placa_bin)
                    
                    cv2.imwrite(os.path.join(pasta_destino, f"placa_moto_{contador}.jpg"), img_recorte)
                    contador += 1
                    break 

            if not sucesso:
                print("[FALHA] Nenhum candidato passou no teste de letras.")
                cv2.imshow("Debug Geral", img_debug)

        key = cv2.waitKey(0)
        if key == 27: 
            break
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()