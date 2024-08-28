import cv2
import numpy as np

def detect_colored_objects(image_path, green_thresholds, purple_thresholds, min_distance):
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem.")
        return

    # Redimensionar imagem
    image = cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

    # Convertendo para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Aplicando um blur suave (Kernel) para reduzir ruídos e melhorar a detecção de cor
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    # Definindo os limites para verde e roxo
    lower_green = np.array(green_thresholds[0], dtype="uint8")
    upper_green = np.array(green_thresholds[1], dtype="uint8")
    lower_purple = np.array(purple_thresholds[0], dtype="uint8")
    upper_purple = np.array(purple_thresholds[1], dtype="uint8")

    # Criando máscaras
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Encontrando contornos
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Função para desenhar contornos
    def draw_contours(contours, color):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                point = np.array([cx, cy])

                # Verificar a distância mínima entre os centros detectados
                if all(np.linalg.norm(point - np.array(center)) > min_distance for center in centers):
                    centers.append(point)
                    cv2.circle(image, (cx, cy), 10, color, -1)

    draw_contours(green_contours, (0, 255, 0))
    draw_contours(purple_contours, (128, 0, 128))

    # Exibir imagem
    cv2.imshow("Detected Colors", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


green_thresholds = ([90, 150, 80], [100, 250, 110])  
purple_thresholds = ([130, 50, 50], [160, 255, 255])  
min_distance = 50  

image_path = input("Digite o caminho do arquivo da imagem: ")
detect_colored_objects(image_path, green_thresholds, purple_thresholds, min_distance)
