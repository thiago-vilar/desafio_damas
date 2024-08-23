import cv2
import numpy as np
import os
import sys
from datetime import datetime

def detect_aruco_markers(image_path):
    # Verificar se o arquivo existe
    if not os.path.exists(image_path):
        print(f"Erro: Imagem não encontrada no caminho especificado: {image_path}")
        return

    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Falha ao carregar a imagem. Verifique o formato ou integridade do arquivo.")
        return

    # Configuração inicial para os marcadores ArUco
    aruco_dict_type = cv2.aruco.DICT_6X6_250
    arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters()

    # Detectar marcadores na imagem
    corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if ids is not None:
        # Desenhar os contornos dos marcadores detectados na imagem
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        for i, corner in enumerate(corners):
            # corner[0] são os pontos dos cantos para o i-ésimo marcador detectado
            print(f"Cantos do Marcador ID {ids[i][0]}: {corner[0]}")

        # Gerar nome de arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_image_path = f"aruco_detected_{timestamp}.jpg"

        # Salvar a imagem processada
        cv2.imwrite(new_image_path, image)
        print(f"Imagem salva como: {new_image_path}")

        # Mostrar a imagem com marcadores detectados
        cv2.imshow('Detected ArUco markers', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhum marcador ArUco detectado.")

# Exemplo de uso
if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_aruco_markers(sys.argv[1])
    else:
        print("Por favor, forneça o caminho da imagem como argumento.")
