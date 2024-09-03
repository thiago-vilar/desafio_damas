import os
import cv2

def capture_image(save_path, width, height):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Falha no acesso à câmera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Falha ao capturar imagem")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_path = os.path.join(save_path, 'captured_image.jpg')
    cv2.imwrite(image_path, frame)
    print(f"Imagem salva em: {image_path}")

if __name__ == "__main__":
    resolucoes = {
        1: (640, 480),
        2: (1280, 720),
        3: (1920, 1080),
        4: (2560, 1440),
        5: (3840, 2160)
    }

    print("Escolha uma opção de resolução para captura:")
    for key, value in resolucoes.items():
        print(f"{key}: {value[0]}x{value[1]}")

    escolha = int(input("Digite o número da resolução desejada: "))
    if escolha in resolucoes:
        width, height = resolucoes[escolha]
    else:
        print("Opção inválida.")
        exit()

    save_directory = 'src\\view\\frame_img'
    capture_image(save_directory, width, height)
