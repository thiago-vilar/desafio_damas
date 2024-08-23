import os
import cv2

def capture_images(save_path, width, height, num_images=10):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise Exception("Falha no acesso à câmera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            raise Exception("Falha ao capturar imagem")

        image_path = os.path.join(save_path, f'captured_image_{i+1}.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Imagem {i+1} salva em: {image_path}")

    cap.release()

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
        save_directory = 'src\\view\\utils_view\\frames_dataset\\resolution_' + str(escolha)
        capture_images(save_directory, width, height)
    else:
        print("Opção inválida.")
