import cv2
import numpy as np
from cv2 import aruco
import time

class ArucoDetector:
    def __init__(self, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, id_map=None):
        self.dictionary_type = dictionary_type
        self.id_map = id_map

    def find_aruco_markers(self, img, draw=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(self.dictionary_type)
        aruco_params = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if draw and ids is not None:
            aruco.drawDetectedMarkers(img, corners, ids)
        return corners, ids

    def find_closest_point_to_center(self, img, corners):
        image_center = np.array([img.shape[1] // 2, img.shape[0] // 2])
        closest_points = []
        for corner in corners:
            distances = np.linalg.norm(corner[0] - image_center, axis=1)
            min_index = np.argmin(distances)
            closest_point = corner[0][min_index]
            closest_points.append(closest_point)
            cv2.circle(img, tuple(closest_point.astype(int)), 4, (0, 255, 0), -1)
        return np.array(closest_points)

    def associate_points_with_ids(self, closest_points, ids, img):
        required_ids = set(self.id_map.keys())
        if not required_ids.issubset(set(ids.flatten())):
            return None

        ordered_points = [None] * len(self.id_map)
        for i, point in enumerate(closest_points):
            aruco_id = ids[i][0]
            if aruco_id in self.id_map:
                item = self.id_map[aruco_id]
                position_index = item['position'][0] + item['position'][1] * 2
                ordered_points[position_index] = point
                cv2.putText(img, f"{item['label']} ({aruco_id})",
                            (int(point[0]), int(point[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if any(p is None for p in ordered_points):
            print("Erro: Falha ao associar corretamente os pontos aos IDs dos ArUcos.")
            return None
        return np.array(ordered_points)

class PerspectiveTransformer:
    def apply_transform(self, img, src_points):
        dst_points = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src_points.astype('float32'), dst_points)
        warped = cv2.warpPerspective(img, matrix, (480, 480))
        return warped

class ObjectAndLineDetector:
    def __init__(self, green_thresholds, purple_thresholds, min_distance):
        self.green_thresholds = green_thresholds
        self.purple_thresholds = purple_thresholds
        self.min_distance = min_distance

    def detect_colored_objects(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        green_mask = cv2.inRange(hsv, np.array(self.green_thresholds[0], dtype="uint8"),
                                 np.array(self.green_thresholds[1], dtype="uint8"))
        purple_mask = cv2.inRange(hsv, np.array(self.purple_thresholds[0], dtype="uint8"),
                                  np.array(self.purple_thresholds[1], dtype="uint8"))
        green_centers = self._draw_contours(img, green_mask, (0, 255, 0))
        purple_centers = self._draw_contours(img, purple_mask, (128, 0, 128))
        return green_centers, purple_centers

    def _draw_contours(self, img, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                point = np.array([cx, cy])
                if all(np.linalg.norm(point - np.array(center)) > self.min_distance for center in centers):
                    centers.append(point)
                    # Desenho de círculo nas peças detectadas
                    cv2.circle(img, (cx, cy), 20, color, 2)
        return centers

def detect_board_status(warped, green_centers, purple_centers):
    board = np.full((8, 8), '⬜', dtype=object)
    cell_size = 60
    emoji_map = {1: '🟢', 2: '🟣'}
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                cell_x = j * cell_size + cell_size // 2
                cell_y = i * cell_size + cell_size // 2
                for center in green_centers:
                    if abs(center[0] - cell_x) < cell_size // 2 and abs(center[1] - cell_y) < cell_size // 2:
                        board[i, j] = emoji_map[1]
                for center in purple_centers:
                    if abs(center[0] - cell_x) < cell_size // 2 and abs(center[1] - cell_y) < cell_size // 2:
                        board[i, j] = emoji_map[2]
            else:
                board[i, j] = '⬛'
    return board

def calculate_average_board(board_accumulator):
    final_board = np.full((8, 8), '⬜', dtype=object)
    for i in range(8):
        for j in range(8):
            cell_votes = [board[i][j] for board in board_accumulator if board[i][j] in ['🟢', '🟣']]
            if cell_votes:
                final_board[i, j] = max(set(cell_votes), key=cell_votes.count)
            else:
                final_board[i, j] = '⬛' if (i + j) % 2 == 1 else '⬜'
    return final_board

def process_frame(frame, detector, transformer, object_line_detector, previous_board, board_accumulator):
    corners, ids = detector.find_aruco_markers(frame)
    if ids is not None and corners:
        closest_points = detector.find_closest_point_to_center(frame, corners)
        labeled_points = detector.associate_points_with_ids(closest_points, ids, frame)
        if labeled_points is not None:
            warped = transformer.apply_transform(frame, labeled_points)
            green_centers, purple_centers = object_line_detector.detect_colored_objects(warped)
            current_board = detect_board_status(warped, green_centers, purple_centers)
            board_accumulator.append(current_board)

            # Calcula a média dos frames a cada 5 acumulações
            if len(board_accumulator) >= 5:
                average_board = calculate_average_board(board_accumulator)
                board_accumulator.clear()

                if previous_board is not None:
                    move_made = detect_move(previous_board, average_board)
                    if move_made:
                        print("Tabuleiro atualizado:")
                        print(average_board)
                        previous_board = average_board
                else:
                    print("Tabuleiro inicial:")
                    print(average_board)
                    previous_board = average_board

    return frame, previous_board

def detect_move(previous_board, current_board):
    differences = compare_boards(previous_board, current_board)
    move_made = False
    for diff in differences:
        i, j, prev, curr = diff
        if prev == '⬛' and curr in ['🟢', '🟣']:
            print(f"Peça {curr} movida para {chr(65 + j)}{8 - i}")
            move_made = True
        elif curr == '⬛' and prev in ['🟢', '🟣']:
            print(f"Peça {prev} capturada de {chr(65 + j)}{8 - i}")
            move_made = True
    return move_made

def compare_boards(board1, board2):
    differences = []
    for i in range(8):
        for j in range(8):
            if board1[i, j] != board2[i, j]:
                differences.append((i, j, board1[i, j], board2[i, j]))
    return differences

def main():
    aruco_id_map = {
        2: {'label': 'P1', 'position': (0, 0)},
        10: {'label': 'P2', 'position': (1, 0)},
        11: {'label': 'P3', 'position': (1, 1)},
        12: {'label': 'P4', 'position': (0, 1)},
    }

    # Limiares para detecção das peças verdes e roxas
    green_thresholds = ([88, 180, 104], [135, 255, 187])
    purple_thresholds = ([118, 100, 66], [255, 251, 255])
    min_distance = 50

    # Inicializa os detectores e transformadores de perspectiva
    detector = ArucoDetector(id_map=aruco_id_map)
    transformer = PerspectiveTransformer()
    object_line_detector = ObjectAndLineDetector(green_thresholds, purple_thresholds, min_distance)

    # Carrega o vídeo
    cap = cv2.VideoCapture("2024-09-03 14-31-00.mp4")

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    previous_board = None  # Armazena o estado anterior do tabuleiro
    board_accumulator = []  # Acumula os estados do tabuleiro a cada frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Vídeo finalizado ou erro ao ler o frame.")
            break

        # Redimensiona o frame para melhor visualização
        frame_resized = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

        # Processa o frame e detecta movimentos
        frame_resized, previous_board = process_frame(
            frame_resized, detector, transformer, object_line_detector, previous_board, board_accumulator
        )

        # Exibe o vídeo processado com as peças detectadas
        cv2.imshow("Processed Video", frame_resized)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos de vídeo
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
