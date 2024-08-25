import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import datetime
import os  # Importando o módulo os para manipulação de diretórios
import matplotlib.colors as mcolors

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

    # Criação do diretório se não existir
    output_dir = 'src/view/frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image.save(os.path.join(output_dir, f'img_{current_time}.jpg'))
    print(f"Imagem salva em '{os.path.join(output_dir, f'img_{current_time}.jpg')}'")

def main():
    choice = input("Digite 'f' para analisar uma imagem de arquivo ou 'w' para usar a webcam: ")
    if choice.lower() == 'w':
        image = capture_image_from_camera()
    elif choice.lower() == 'f':
        image = load_image_from_file()
    else:
        print("Opção inválida.")
        return

    if image:
        analyze_and_save_image(image)

if __name__ == "__main__":
    main()
