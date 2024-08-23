import cv2
import numpy as np
import os

def detect_aruco_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Imagem não encontrada.")
        return

    process_image_and_detect_aruco(image)

def detect_aruco_from_camera():
    cap = cv2.VideoCapture(0)
    print("Pressione 'q' para sair da captura...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem da câmera.")
            break

        process_image_and_detect_aruco(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image_and_detect_aruco(image):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

    cv2.imshow('Detected ArUco markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    choice = input("Escolha o modo de detecção de ArUco - 'camera' ou 'imagem': ").lower()
    if choice == 'camera':
        detect_aruco_from_camera()
    elif choice == 'imagem':
        image_path = input("Digite o caminho da imagem: ")
        detect_aruco_from_image(image_path)
    else:
        print("Escolha inválida. Por favor, digite 'camera' ou 'imagem'.")

if __name__ == "__main__":
    main()
