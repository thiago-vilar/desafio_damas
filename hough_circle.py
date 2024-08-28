import cv2
import numpy as np

def detect_circles():
    # Solicitar ao usuário o caminho da imagem
    image_path = input("Digite o caminho completo da imagem: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro: Não foi possível abrir a imagem.")
        return

    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar um desfoque Gaussiano para reduzir o ruído
    gray = cv2.medianBlur(gray, 5)

    # Detecção de círculos usando HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=50, param2=90, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # coordenadas do centro do círculo
            radius = i[2]  # raio do círculo
            # Desenhar o círculo no centro
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # Desenhar o contorno do círculo
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    # Mostrar a imagem
    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_circles()
