import cv2
import numpy as np

def detect_checkers_pieces():
    image_path = input("Digite o caminho completo da imagem: ")
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem.")
        return

    # Coordenadas hipotéticas dos cantos, substitua pela detecção automática ou entrada manual
    points_src = np.float32([[100, 100], [700, 100], [700, 700], [100, 700]])
    points_dst = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

    # Aplicando a correção de perspectiva
    matrix = cv2.getPerspectiveTransform(points_src, points_dst)
    image = cv2.warpPerspective(image, matrix, (800, 800))

    # Processamento para detecção de círculos
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=30, minRadius=10, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('Detected Checkers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_checkers_pieces()
