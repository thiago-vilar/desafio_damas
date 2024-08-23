import cv2
import os
from datetime import datetime

def capture_and_detect_aruco():
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)

    # Tenta capturar um frame
    ret, frame = cap.read()

    # Verifica se capturou o frame corretamente
    if not ret:
        print("Falha ao capturar imagem da câmera.")
        return

    # Carrega o dicionário ArUco e parâmetros de detecção
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()

    # Detecta os marcadores ArUco na imagem
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    # Se algum marcador foi encontrado, desenha na imagem
    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Gera o nome do arquivo com datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"img_{timestamp}.jpg"
    filepath = os.path.join("frames", filename)

    # Cria a pasta se não existir
    if not os.path.exists("frames"):
        os.makedirs("frames")

    # Salva a imagem processada no diretório frames
    cv2.imwrite(filepath, frame)
    print(f"Imagem salva como: {filepath}")

    # Libera a câmera e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect_aruco()
