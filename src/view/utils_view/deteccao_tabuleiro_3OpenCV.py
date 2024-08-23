import cv2
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans

def load_image(path):
    return cv2.imread(path)

def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow("Press space to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.destroyAllWindows()
            cap.release()
            return frame

def detect_intersection_points(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
 
    edges = cv2.Canny(blur, 50, 150)
  
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return []
    

    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y1 - y2) < 10:
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(x1 - x2) < 10:
                vertical_lines.append((x1, y1, x2, y2))
    

    intersection_points = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            point = intersection(h_line, v_line)
            if point:
                intersection_points.append(point)
    
    return intersection_points

def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    a1, b1 = y2 - y1, x1 - x2
    a2, b2 = y4 - y3, x3 - x4
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    
    c1, c2 = a1 * x1 + b1 * y1, a2 * x3 + b2 * y3
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    
    return int(x), int(y)

def filtrar_49_pontos(pontos_contraste):
    print(f"Número de pontos detectados: {len(pontos_contraste)}") 
    if len(pontos_contraste) < 49:
        raise ValueError("Não foi possível encontrar 49 pontos de contraste.")
    

    kmeans = KMeans(n_clusters=49)
    kmeans.fit(pontos_contraste)
    pontos_49 = kmeans.cluster_centers_.astype(int).tolist()

    pontos_49 = sorted(pontos_49, key=lambda x: (x[1], x[0]))
    matriz_pontos = []
    for i in range(7):
        linha = pontos_49[i*7:(i+1)*7]
        matriz_pontos.append(sorted(linha, key=lambda x: x[0]))
    
    pontos_49 = [p for linha in matriz_pontos for p in linha]
    return pontos_49

def draw_border_and_save(image, intersection_points):
    if not intersection_points:
        print("No intersection points found.")
        return
    
    pontos_array = np.array(intersection_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pontos_array)
    side_length = max(w, h)
    square_rect = (x, y, side_length, side_length)
    box = np.array([
        [square_rect[0], square_rect[1]],
        [square_rect[0] + square_rect[2], square_rect[1]],
        [square_rect[0] + square_rect[2], square_rect[1] + square_rect[3]],
        [square_rect[0], square_rect[1] + square_rect[3]]
    ], dtype=np.int32)
    

    cv2.drawContours(image, [box], 0, (255, 0, 0), 3)
    
  
    for i, ponto in enumerate(intersection_points):
        cv2.putText(image, f'{i+1}', (ponto[0] - 10, ponto[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho_saida = os.path.join("src", "view", "frames", f"img_processed_{data_hora}.jpg")
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, image)
    print(f"Image processed saved to: {caminho_saida}")

def main_menu():
    while True:
        print("Main Menu")
        print("1. Recognize table from JPEG image")
        print("2. Recognize table from webcam image")
        print("3. Exit")
        opcao = input("Choose an option: ")
        
        if opcao == '1':
            caminho = input("Enter the path to the JPEG image: ")
            image = load_image(caminho)
            if image is not None:
                intersection_points = detect_intersection_points(image)
                if len(intersection_points) < 49:
                    print(f"Apenas {len(intersection_points)} pontos de contraste detectados. Tente outra imagem.")
                else:
                    pontos_49 = filtrar_49_pontos(intersection_points)
                    draw_border_and_save(image, pontos_49)
            else:
                print("Failed to load image.")
        elif opcao == '2':
            image = capture_webcam_image()
            if image is not None:
                intersection_points = detect_intersection_points(image)
                if len(intersection_points) < 49:
                    print(f"Apenas {len(intersection_points)} pontos de contraste detectados. Tente outra captura.")
                else:
                    pontos_49 = filtrar_49_pontos(intersection_points)
                    draw_border_and_save(image, pontos_49)
            else:
                print("Failed to capture webcam image.")
        elif opcao == '3':
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main_menu()
