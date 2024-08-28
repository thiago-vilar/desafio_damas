import cv2
import numpy as np

def detect_and_draw_pieces():
    # Solicitar ao usuário o caminho da imagem
    image_path = input("Digite o caminho completo da imagem: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro: Não foi possível abrir a imagem. Verifique se o caminho está correto.")
        return

    # Converter a imagem para espaço de cores HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir os limites para a cor verde
    lower_green = np.array([77, 121, 38])
    upper_green = np.array([109, 255, 82])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Definir os limites para a cor roxa
    lower_purple = np.array([115, 80, 109])
    upper_purple = np.array([149, 235, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    min_radius = 20  # Tamanho mínimo do raio
    max_radius = 31  # Tamanho máximo do raio
    min_distance = 30  # Distância mínima entre centros de círculos detectados

    # Função para processar contornos e desenhar círculos
    def process_contours(contours, image, color):
        centers = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if min_radius <= radius <= max_radius:
                if all(np.linalg.norm(np.array(center) - np.array(c)) >= min_distance for c in centers):
                    centers.append(center)
                    cv2.circle(image, center, radius, color, 2)

    # Encontrar contornos e desenhar círculos para peças verdes
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    process_contours(contours_green, image, (0, 255, 0))  # Verde

    # Encontrar contornos e desenhar círculos para peças roxas
    contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    process_contours(contours_purple, image, (128, 0, 128))  # Roxo

    # Redimensionar a imagem para 75% do tamanho original
    resized_image = cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

    # Mostrar resultados
    cv2.imshow('Detected Pieces with Circles', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_draw_pieces()
