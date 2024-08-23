import cv2
import numpy as np

def detect_aruco_live():
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)  # Usa a primeira câmera conectada

    # Carrega o dicionário ArUco e parâmetros de detecção
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()

    while True:
        # Captura frame a frame
        ret, frame = cap.read()
        if not ret:
            break  # Se falhou em capturar o frame, saia do loop

        # Detecta os marcadores na imagem
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        # Verifica se algum marcador foi detectado
        if ids is not None and len(ids) > 1:  # Precisamos de pelo menos dois marcadores
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Encontra os cantos mais próximos entre os marcadores
            min_distance = float('inf')
            best_pair = None

            for i in range(len(corners)):
                for j in range(i + 1, len(corners)):
                    for corner1 in corners[i][0]:
                        for corner2 in corners[j][0]:
                            distance = np.linalg.norm(corner1 - corner2)
                            if distance < min_distance:
                                min_distance = distance
                                best_pair = (corner1, corner2)

            if best_pair is not None:
                # Desenha uma linha entre os cantos mais próximos
                pt1 = tuple(best_pair[0].astype(int))
                pt2 = tuple(best_pair[1].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                print(f"Distância entre cantos: {min_distance} pixels")

        # Mostra o frame resultante
        cv2.imshow('Frame', frame)

        # Aperte 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Quando tudo estiver feito, libera a captura
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_live()
