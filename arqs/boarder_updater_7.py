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
                    cv2.circle(img, (cx, cy), 20, color, -1)
        return centers


def draw_lines_and_labels(warped):
    text_color = (0, 0, 0)
    cell_size = 60
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                cell_label = chr(65 + j) + str(8 - i)
                x_pos = j * cell_size + cell_size // 2 - 10
                y_pos = i * cell_size + cell_size // 2 + 5
                cv2.putText(warped, cell_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return warped


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


def compare_boards(board1, board2):
    """ Compara dois tabuleiros e retorna as diferenças. """
    differences = []
    for i in range(8):
        for j in range(8):
            if board1[i, j] != board2[i, j]:
                differences.append((i, j, board1[i, j], board2[i, j]))
    return differences


def detect_move(previous_board, current_board):
    differences = compare_boards(previous_board, current_board)
    PLAYER_GREEN = 1
    PLAYER_PURPLE = 2
    color_to_player = {'🟢': PLAYER_GREEN, '🟣': PLAYER_PURPLE}
    move_made = False
    for diff in differences:
        i, j, prev, curr = diff
        if prev == '⬛' and curr in ['🟢', '🟣']:
            print(f"Movimento detectado: Jogador {color_to_player[curr]} moveu a peça {curr} para {chr(65 + j)}{8 - i}")
            move_made = True
        elif curr == '⬛' and prev in ['🟢', '🟣']:
            print(f"Peça {prev} do Jogador {color_to_player[prev]} capturada de {chr(65 + j)}{8 - i}")
            move_made = True
        elif prev in ['🟢', '🟣'] and curr in ['🟢', '🟣'] and prev != curr:
            print(f"Peça {prev} do Jogador {color_to_player[prev]} movida e substituída por {curr} do Jogador {color_to_player[curr]} em {chr(65 + j)}{8 - i}")
            move_made = True
    # if not move_made:
    #     # print("Nenhuma jogada detectada neste frame.")


def process_frame(frame, detector, transformer, object_line_detector, previous_board):
    corners, ids = detector.find_aruco_markers(frame)
    if ids is not None and corners:
        closest_points = detector.find_closest_point_to_center(frame, corners)
        labeled_points = detector.associate_points_with_ids(closest_points, ids, frame)
        if labeled_points is not None:
            warped = transformer.apply_transform(frame, labeled_points)
            green_centers, purple_centers = object_line_detector.detect_colored_objects(warped)
            current_board = detect_board_status(warped, green_centers, purple_centers)
            final_image = draw_lines_and_labels(warped)
            # rotated_final = cv2.rotate(final_image, cv2.ROTATE_180)
            cv2.imshow('Processed Image', final_image)
            cv2.waitKey(1)
            if previous_board is not None:
                detect_move(previous_board, current_board)

            previous_board = current_board
            return frame, previous_board
    return frame, previous_board


def calculate_average_board(board_accumulator):
    """ Função para calcular a média dos tabuleiros acumulados """
    final_board = np.full((8, 8), '⬜', dtype=object)

    for i in range(8):
        for j in range(8):
            cell_votes = [board[i][j] for board in board_accumulator if board[i][j] in ['🟢', '🟣']]
            if cell_votes:
                final_board[i, j] = max(set(cell_votes), key=cell_votes.count)  
            else:
                final_board[i, j] = '⬛' if (i + j) % 2 == 1 else '⬜'

    return final_board


def update_accumulated_board(detector, transformer, object_line_detector, frame, board_accumulator):
    """ Função para acumular estados dos tabuleiros a cada frame """
    corners, ids = detector.find_aruco_markers(frame)
    if ids is not None and corners:
        closest_points = detector.find_closest_point_to_center(frame, corners)
        labeled_points = detector.associate_points_with_ids(closest_points, ids, frame)
        if labeled_points is not None:
            warped = transformer.apply_transform(frame, labeled_points)
            green_centers, purple_centers = object_line_detector.detect_colored_objects(warped)
            current_board = detect_board_status(warped, green_centers, purple_centers)
            board_accumulator.append(current_board)


def main():
    aruco_id_map = {
        # Define as posições dos ArUcos
         2: {'label': 'P1', 'position': (0, 0)},  # canto superior-esquerdo (próximo ao A8)
        10: {'label': 'P2', 'position': (1, 0)},  # canto superior-direito (próximo ao H8)
        11: {'label': 'P3', 'position': (1, 1)},  # canto inferior-esquerdo (próximo ao A1)
        12: {'label': 'P4', 'position': (0, 1)},  # canto inferior direito (próximo ao H1)
    }

    green_thresholds = ([88, 182, 104], [135, 255, 200])
    # green_thresholds = ([88, 180, 104], [135, 255, 187])
    purple_thresholds = ([118, 100, 66], [255, 251, 255])  
    min_distance = 50 

    detector = ArucoDetector(id_map=aruco_id_map)
    transformer = PerspectiveTransformer()
    object_line_detector = ObjectAndLineDetector(green_thresholds, purple_thresholds, min_distance)

    cap = cv2.VideoCapture("2024-09-03 14-31-00.mp4")

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    previous_board = None
    last_update_time = time.time()
    board_accumulator = []
    last_board_update = time.time()  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Vídeo finalizado ou erro ao ler o frame.")
            break

        # Saída 1: Exibir o vídeo redimensionado e cortado
        frame_resized = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        height, width = frame_resized.shape[:2]
        video_cropped = frame_resized[:, 100:width - 100]  
        cv2.imshow("video", video_cropped)

        # Saída 2: Detecção em tempo real (tabuleiro e peças)
        if time.time() - last_update_time >= 0:
            frame_resized, previous_board = process_frame(frame_resized, detector, transformer, object_line_detector, previous_board)
            last_update_time = time.time()

        # Saída 3: Mostrar a média do tabuleiro no terminal a cada 3 segundos
        if time.time() - last_board_update >= 1:
            if board_accumulator:
                average_board = calculate_average_board(board_accumulator)
                # rotated_board = np.rot90(average_board, 2)
                print(average_board)
                board_accumulator = []  
            last_board_update = time.time()

        # Acumula o estado do tabuleiro atual
        update_accumulated_board(detector, transformer, object_line_detector, frame, board_accumulator)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
