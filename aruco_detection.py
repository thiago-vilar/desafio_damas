import cv2
from cv2 import aruco
import numpy as np

def find_aruco_markers(img, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
    
    return corners, ids

def resize_image(img, scale=0.75):
    return cv2.resize(img, (0,0), fx=scale, fy=scale)

def main():
    print("Digite a opção:\n 1 - Imagem\n 2 - Vídeo")
    choice = input()

    if choice == '1':
        path = input("Digite o caminho da imagem: ")
        img = cv2.imread(path)
        if img is not None:
            img = resize_image(img)
            corners, ids = find_aruco_markers(img)
            cv2.imshow("ArUco Markers", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Imagem não encontrada. Verifique o caminho fornecido.")
    
    elif choice == '2':
        cap = cv2.VideoCapture(0) 

        while True:
            ret, frame = cap.read()
            if ret:
                frame = resize_image(frame)
                corners, ids = find_aruco_markers(frame)
                cv2.imshow("ArUco Markers", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Falha ao capturar vídeo.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Opção inválida. Por favor, digite '1' para imagem ou '2' para vídeo.")

if __name__ == "__main__":
    main()
