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
            '1': np.array([[849,649],[332,679], [312,156], [825,145]], dtype="float32"),
            '2': np.array([ [413,109], [931,110], [916,629],[411,614]], dtype="float32")
        }
        return points[kinova_choice]

    def capture_image(self, width=1280, height=720):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Falha no acesso à câmera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
      
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if not ret:
            raise Exception("Falha ao capturar imagem")
        
        print(f"Imagem capturada da câmera com resolução: {actual_width}x{actual_height}")
        return frame

    def load_image(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError("Arquivo de imagem não encontrado.")
        print(f"Imagem carregada de {filepath}")
        return image


    def save_image(self, image, prefix):
        if image is not None:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'{prefix}_processed_{date_time}.jpg')
            cv2.imwrite(save_path, image)
            print(f"Imagem salva em: {save_path}")
            return save_path
        else:
            print("Nenhuma imagem para salvar.")
            return None

    def detect_pieces(self, image):
        resized_image = cv2.resize(image, (2000, 2000))
        results = self.model(resized_image)
        detections = []
        scale_x, scale_y = image.shape[1] / 2000, image.shape[0] / 2000
        for result in results:
            for det in result.boxes.data:
                x1, y1, x2, y2 = map(int, [det[0].item() * scale_x, det[1].item() * scale_y, 
                                        det[2].item() * scale_x, det[3].item() * scale_y])
                conf, cls_id = det[4].item(), int(det[5].item())
                if conf > 0.50: 
                    detections.append((x1, y1, x2, y2, conf, cls_id))
        return detections

    def draw_detections(self, image, detections):
        color_map = {
            'peca_roxa': (255, 0, 255), 'peca_verde': (0, 255, 0),
            'dama_roxa': (128, 0, 128), 'dama_verde': (0, 128, 0)
        }
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            label = self.model.names[cls_id]
            # print(f"Rótulo: {label}, Confiança: {conf:.2f}")
            color = color_map.get(label, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image

    def draw_lines(self, image):
        height, width = image.shape[:2]
        line_color = (150, 0, 0)  
        for i in range(1, 8):
            cv2.line(image, (0, i * height // 8), (width, i * height // 8), line_color, 1)
            cv2.line(image, (i * width // 8, 0), (i * width // 8, height), line_color, 1)
        # print("Linhas desenhadas na imagem")
        return image

    def add_labels(self, image):
        height, width = image.shape[:2]
        text_color = (150, 0, 0)
        for i in range(8):
            start = (i + 1) % 2
            for j in range(start, 8, 2):
                cell_label = chr(65 + j) + str(8 - i)
                x_pos = j * width // 8 + 10
                y_pos = i * height // 8 + 20
                cv2.putText(image, cell_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        # print("Rótulos adicionados apenas em casas brancas")
        return image
    
    def apply_homography(image, matrix):
        transformed_img = cv2.warpPerspective(image, matrix, (2000, 2000))
        return transformed_img

    def map_image(self, image, points):
        width, height = 2000, 2000
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(points, dst)
        mapped = cv2.warpPerspective(image, M, (width, height))
        # print("Imagem mapeada e redimensionada para 2000x2000")
        return mapped


    def process_image_step_by_step(self, image, kinova_choice):
        points = self.set_kinova(kinova_choice)
        mapped_image = self.map_image(image, points)
        detections = self.detect_pieces(mapped_image)
        detected_image = self.draw_detections(mapped_image, detections)
        image_with_lines = self.draw_lines(detected_image)
        image_with_labels = self.add_labels(image_with_lines)
        final_image_path = self.save_image(image_with_labels, 'processed_image')  
        return final_image_path, self.update_board_state(detections, image_with_labels)

    
    def map_board(self):
        board = np.full((8, 8), '', dtype=object)
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    board[row, col] = ''
        return board

    def update_board_state(self, detections, image):
        board = self.map_board()  
        emoji_map = {
            'peca_verde': 'V',
            'peca_roxa': 'R',
            'dama_verde': 'DV',
            'dama_roxa': 'DR'
        }
        for x1, y1, x2, y2, conf, cls_id in detections:
            label = self.model.names[cls_id]
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            col = x_center // (image.shape[1] // 8)
            row = y_center // (image.shape[0] // 8)
            if 0 <= row < 8 and 0 <= col < 8:  
                board[row][col] = emoji_map.get(label, '❓')  

        return board


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
        elif operation_choice == '2':
            image = app.capture_image()
        else:
            print("Escolha de operação inválida.")
            exit()

        final_image_path, board_state = app.process_image_step_by_step(image, kinova_choice)
        print("Processamento de imagem concluído com sucesso, imagem final salva em:", final_image_path)

    except Exception as e:
        print(f"Erro: {e}")