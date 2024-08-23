import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import datetime
import matplotlib.colors as mcolors
import os


def get_color_name(rgb):
    min_dist = float('inf')
    closest_color = None
    for name, hex_val in mcolors.CSS4_COLORS.items():
        r2, g2, b2 = mcolors.hex2color(hex_val)
        dist = np.linalg.norm([r2 * 255 - rgb[0], g2 * 255 - rgb[1], b2 * 255 - rgb[2]])
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    return closest_color

def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    test_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    test_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return test_width == width and test_height == height

def capture_image_from_camera():
    resolutions = [
        (640, 480),  # VGA
        (1280, 720),  # HD
        (1920, 1080),  # Full HD
        (2048, 1536),  # 3MP
        (3840, 2160)  # 4K
    ]
    print("Escolha a resolução para a captura:")
    for i, res in enumerate(resolutions, 1):
        print(f"{i}. {res[0]}x{res[1]}")
    
    choice = int(input("Digite o número da resolução (1-5): ")) - 1
    width, height = resolutions[choice]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return None

    if not set_camera_resolution(cap, width, height):
        print("A resolução escolhida não é suportada pela sua câmera.")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Falha ao capturar imagem da câmera.")
        return None

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def load_image_from_file():
    path = input("Digite o caminho absoluto da imagem: ")
    return Image.open(path)

def print_colored_text(rgb, text):
    r, g, b = rgb
    print(f"\033[38;2;{r};{g};{b}m{text}\033[0m")

def analyze_and_save_image(image):
    image_np = np.array(image)
    resolution = image.size

    pixels = image_np.reshape(-1, 3)
    model = KMeans(n_clusters=10)
    labels = model.fit_predict(pixels)
    palette = model.cluster_centers_
    counts = np.bincount(labels)
    dominant_colors = palette[np.argsort(counts)[::-1]]
    percentages = (counts[np.argsort(counts)[::-1]] / sum(counts)) * 100


    draw = ImageDraw.Draw(image)
    font_size = 48  
    font = ImageFont.truetype("arial.ttf", font_size)


    text = f"Resolution: {resolution[0]}x{resolution[1]}"
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))


    y_offset = 80
    box_size = 50 
    for color, percentage in zip(dominant_colors, percentages):
        color_name = get_color_name(color)
        color_box_start = (10, y_offset)
        color_box_end = (10 + box_size, y_offset + box_size)
        draw.rectangle([color_box_start, color_box_end], fill=tuple(color.astype(int)))
        color_text = f"RGB: {int(color[0])}, {int(color[1])}, {int(color[2])} - {percentage:.2f}% {color_name}"
        draw.text((10 + box_size + 10, y_offset), color_text, font=font, fill=(0, 0, 0))
        y_offset += box_size + 20
        

        print_colored_text(color.astype(int), color_text)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image.save(f'src/view/frames/img_{current_time}.jpg')
    print(f"Imagem salva em 'src/view/frames/img_{current_time}.jpg'")

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
    if len(pontos_contraste) > 49:
        pontos_contraste = sorted(pontos_contraste, key=lambda x: x[0] + x[1])[:49]
    elif len(pontos_contraste) < 49:
        raise ValueError("Não foi possível encontrar 49 pontos de contraste.")
    
   
    pontos_contraste = sorted(pontos_contraste, key=lambda x: (x[1], x[0]))
    matriz_pontos = []
    for i in range(7):
        linha = pontos_contraste[i*7:(i+1)*7]
        matriz_pontos.append(sorted(linha, key=lambda x: x[0]))
    
    pontos_contraste = [p for linha in matriz_pontos for p in linha]
    return pontos_contraste

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
    

    data_hora = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
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
