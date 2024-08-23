import os
import cv2
import numpy as np
from ultralytics import YOLO

class UpdateBoard:
    def __init__(self):
        self.input_dir = os.path.join(os.path.dirname(__file__), 'frame0')
        self.output_dir = os.path.join(os.path.dirname(__file__), 'frames')
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'best.pt')
        self.model = self.load_model()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado no caminho: {self.model_path}")
        print("Modelo carregado")
        return YOLO(self.model_path)

    def capture_image(save_path, width=1280, height=720):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Falha no acesso à câmera")

   
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("Falha ao capturar imagem")
        print("Imagem capturada da câmera")
        return frame

    def load_image(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError("Arquivo de imagem não encontrado.")
        print(f"Imagem carregada de {filepath}")
        return image

    def save_image(self, image, stage):
        if image is not None:
            existing_files = [f for f in os.listdir(self.output_dir) if f.startswith(stage) and f.endswith('.jpg')]
            max_index = max([int(f[len(stage)+1:-4]) for f in existing_files], default=0)
            save_path = os.path.join(self.output_dir, f'{stage}_{max_index + 1:03d}.jpg')
            cv2.imwrite(save_path, image)
            print(f"Imagem salva em: {save_path}")
            return save_path
        else:
            print("Nenhuma imagem para salvar.")
            return None

    def detect_pieces(self, image):
        resized_image = cv2.resize(image, (640, 640))
        results = self.model(resized_image)
        detections = []
        scale_x, scale_y = image.shape[1] / 640, image.shape[0] / 640
        for result in results:
            for det in result.boxes.data:
                x1, y1, x2, y2 = map(int, [det[0].item() * scale_x, det[1].item() * scale_y, 
                                           det[2].item() * scale_x, det[3].item() * scale_y])
                conf, cls_id = det[4].item(), int(det[5].item())
                detections.append((x1, y1, x2, y2, conf, cls_id))
        print("Peças detectadas:", detections)
        return detections

    def draw_detections(self, image, detections):
        color_map = {
            'peca_roxa': (255, 0, 255), 'peca_verde': (0, 255, 0),
            'dama_roxa': (128, 0, 128), 'dama_verde': (0, 128, 0)
        }
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            label = self.model.names[cls_id]
            print(f"Rótulo: {label}, Confiança: {conf:.2f}")
            color = color_map.get(label, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        return image

    def draw_lines(self, image):
        height, width = image.shape[:2]
        line_color = (255, 0, 0)
        for i in range(1, 8):
            cv2.line(image, (0, i * height // 8), (width, i * height // 8), line_color, 2)
            cv2.line(image, (i * width // 8, 0), (i * width // 8, height), line_color, 2)
        print("Linhas desenhadas na imagem")
        return image

    def add_labels(self, image):
        height, width = image.shape[:2]
        text_color = (255, 0, 0)
        for i in range(8):
            for j in range(8):
                cell_label = chr(65 + j) + str(8 - i)
                cv2.putText(image, cell_label, (j * width // 8 + 15, i * height // 8 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        print("Rótulos adicionados à imagem")
        return image

    def map_image(self, image, points):
        width, height = 640, 640
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(points, dst)
        mapped = cv2.warpPerspective(image, M, (width, height))
        print("Imagem mapeada e redimensionada para 640x640")
        return mapped

    def process_image(self, image, kinova_choice):
        points = {
            '1': np.array([[2546, 1913], [1021, 1995], [962, 442], [2487, 415]], dtype="float32"),
            '2': np.array([[2703, 1861], [1187, 1871], [1204, 351], [2682, 360]], dtype="float32")
        }
        mapped_image = self.map_image(image, points[kinova_choice])
        self.save_image(mapped_image, 'mapped_image')
        detections = self.detect_pieces(mapped_image)
        detected_image = self.draw_detections(mapped_image, detections)
        self.save_image(detected_image, 'detected_image')
        return detected_image, detections

    def map_board(self):
        board = np.full((8, 8), '', dtype=object)
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    board[row, col] = f'{chr(65 + col)}{8 - row}'
        return board

    def process_image_step_by_step(self, image, kinova_choice):
        points = {
            '1': np.array([[2546, 1913], [1021, 1995], [962, 442], [2487, 415]], dtype="float32"),
            '2': np.array([[2703, 1861], [1187, 1871], [1204, 351], [2682, 360]], dtype="float32")
        }
        mapped_image = self.map_image(image, points[kinova_choice])
        self.save_image(mapped_image, 'mapped_image')

        detections = self.detect_pieces(mapped_image)
        detected_image = self.draw_detections(mapped_image, detections)
        self.save_image(detected_image, 'detected_image')

        image_with_lines = self.draw_lines(detected_image)
        image_with_labels = self.add_labels(image_with_lines)
        self.save_image(image_with_labels, 'processed_image')

        return image_with_labels, detections

if __name__ == "__main__":
    app = UpdateBoard()
    kinova_choice = input("Escolha o Kinova [1 ou 2]: ")
    if kinova_choice not in ['1', '2']:
        print("Escolha de Kinova inválida.")
        exit()
    operation_choice = input("Escolha a operação: [1] Reconhecer peças e tabuleiro a partir de um arquivo [2] Capturar via webcam: ")
    try:
        if operation_choice == '1':
            filepath = input("Digite o caminho completo do arquivo de imagem: ")
            image = app.load_image(filepath)
            processed_image, detections = app.process_image_step_by_step(image, kinova_choice)
        elif operation_choice == '2':
            image = app.capture_image()
            captured_image_path = app.save_image(image, 'captured_image')
            image = app.load_image(captured_image_path)
            processed_image, detections = app.process_image_step_by_step(image, kinova_choice)
        else:
            print("Escolha de operação inválida.")
            exit()
        
        final_image_path = app.save_image(processed_image, 'final_processed_image')
        print("Processamento de imagem concluído com sucesso.")

        board = app.map_board()
        emoji_map = {
            'peca_verde': '🟢',
            'peca_roxa': '🟣',
            'dama_verde': '👑',
            'dama_roxa': '👑'
        }
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = app.model.names[cls_id]
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            col = x_center // (processed_image.shape[1] // 8)
            row = y_center // (processed_image.shape[0] // 8)
            if board[row, col]:
                board[row, col] = emoji_map.get(label, label)

        print("Matriz do tabuleiro:")
        print(board)
    except Exception as e:
        print(f"Erro: {e}")
