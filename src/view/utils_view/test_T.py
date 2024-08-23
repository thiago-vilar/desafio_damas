import datetime
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

    def set_kinova(self, kinova_choice):
        points = {
            '1': np.array([[2766, 2082], [760, 2133], [746, 123], [2720, 142]], dtype="float32"),
            '2': np.array([[2703, 1861], [1187, 1871], [1204, 351], [2682, 360]], dtype="float32")
        }
        return points[kinova_choice]

    def capture_image(self, width=3840, height=2160):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Falha no acesso à câmera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Falha ao capturar imagem")
        return frame

    def load_image(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError("Arquivo de imagem não encontrado.")
        return image

    def save_image(self, image, prefix):
        if image is not None:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'{prefix}_processed_{date_time}.jpg')
            cv2.imwrite(save_path, image)
            return save_path
        else:
            return None

    def detect_pieces(self, image):
        results = self.model(image)
        detections = []
        for result in results:
            for det in result.boxes.data:
                x1, y1, x2, y2 = map(int, [det[0].item(), det[1].item(), det[2].item(), det[3].item()])
                conf, cls_id = det[4].item(), int(det[5].item())
                if conf > 0.50:
                    detections.append((x1, y1, x2, y2, conf, cls_id))
        return detections

    def apply_homography_and_crop(self, image, points):
        width, height = 2020, 2020
        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
        M = cv2.getPerspectiveTransform(points, dst)
        transformed = cv2.warpPerspective(image, M, (width, height))
        return transformed

    def draw_detections(self, image, detections):
        color_map = {'peca_roxa': (255, 0, 255), 'peca_verde': (0, 255, 0),
                     'dama_roxa': (128, 0, 128), 'dama_verde': (0, 128, 0)}
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            color = color_map.get(self.model.names[cls_id], (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{self.model.names[cls_id]} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image

    def draw_lines(self, image):
        height, width = image.shape[:2]
        line_color = (150, 0, 0)
        for i in range(1, 8):
            cv2.line(image, (0, i * height // 8), (width, i * height // 8), line_color, 1)
            cv2.line(image, (i * width // 8, 0), (i * width // 8, height), line_color, 1)
        return image

    def add_labels(self, image):
        text_color = (150, 0, 0)
        height, width = image.shape[:2]
        for i in range(8):
            for j in range(8):
                cell_label = f"{chr(65 + j)}{8 - i}"
                x_pos = j * width // 8 + 20
                y_pos = i * height // 8 + 20
                cv2.putText(image, cell_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        return image

    def map_board(self):
        board = np.full((8, 8), '', dtype=object)
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    board[row, col] = f'{chr(65 + col)}{8 - row}'
        return board

    def update_board_state(self, detections, image):
        board = self.map_board()  
        emoji_map = {
            'peca_verde': '🟢',
            'peca_roxa': '🟣',
            'dama_verde': '👑',
            'dama_roxa': '👑'
        }
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = self.model.names[cls_id]
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            col = x_center // (image.shape[1] // 8)
            row = y_center // (image.shape[0] // 8)
            if 0 <= row < 8 and 0 <= col < 8:  
                board[row][col] = emoji_map.get(label, '❓')  

        print("Matriz do tabuleiro:")
        print(board)
        return board

    def process_image_step_by_step(self, image, kinova_choice):
        points = self.set_kinova(kinova_choice)
        transformed_image = self.apply_homography_and_crop(image, points)
        detections = self.detect_pieces(transformed_image)
        detected_image = self.draw_detections(transformed_image, detections)
        image_with_lines = self.draw_lines(detected_image)
        image_with_labels = self.add_labels(image_with_lines)
        final_image_path = self.save_image(image_with_labels, 'final_processed')
        board_state = self.update_board_state(detections, transformed_image)
        return final_image_path, board_state

if __name__ == "__main__":
    app = UpdateBoard()
    kinova_choice = input("Escolha o Kinova [1 ou 2]: ")
    operation_choice = input("Escolha a operação: [1] Carregar de um arquivo [2] Capturar via webcam: ")
    if operation_choice == '1':
        filepath = input("Digite o caminho completo do arquivo de imagem: ")
        image = app.load_image(filepath)
    elif operation_choice == '2':
        image = app.capture_image()
    final_image_path, board_state = app.process_image_step_by_step(image, kinova_choice)
    print("Processamento de imagem concluído com sucesso, imagem final salva em:", final_image_path)
    print("Estado atual do tabuleiro:\n", board_state)
