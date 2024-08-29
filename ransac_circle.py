import cv2
import numpy as np

def detect_shapes(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao abrir imagem!")
        return

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=10, maxRadius=100)

    # Desenha os círculos detectados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), 30, (0, 255, 0), 4)
            cv2.circle(output, (x, y), 3, (0, 0, 255), 3)

    #Detecção de elipses
    contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Ajusta elipses
    for contour in contours:
        if len(contour) >= 5:  #Filtro Ellipse
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(output, ellipse, (255, 0, 0), 2)

    # Exibe a imagem
    cv2.imshow('Detected Shapes', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = input("Caminho da imagem: ")
detect_shapes(image_path)
