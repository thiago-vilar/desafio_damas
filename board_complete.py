import cv2
from cv2 import aruco
import numpy as np

def find_aruco_markers(img, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, draw=True):
    """1. Detects ArUco markers in the provided image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
    return corners, ids

def find_closest_point_to_center(img, corners):
    """2. Finds the point closest to the center of the image for each ArUco marker."""
    image_center = np.array([img.shape[1] // 2, img.shape[0] // 2])
    closest_points = []
    for corner in corners:
        if corner[0].size > 0:
            distances = np.linalg.norm(corner[0] - image_center, axis=1)
            min_index = np.argmin(distances)
            closest_point = corner[0][min_index]
            closest_points.append(closest_point)
            cv2.circle(img, tuple(closest_point.astype(int)), 4, (0, 255, 0), -1)
    return np.array(closest_points)

def associate_points_with_ids(closest_points, ids, img):
    """3. Associates closest points with ArUco IDs to ensure board orientation and labels them."""
    labeled_points = {}
    id_map = {474: 'P1', 553: 'P2', 424: 'P3', 224: 'P4'}
    for i, point in enumerate(closest_points):
        label = id_map.get(ids[i][0], f"ID{ids[i][0]}")
        labeled_points[label] = point
        cv2.putText(img, f"{label} ({ids[i][0]})", (int(point[0]), int(point[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return labeled_points

def draw_min_polygon(img, points):
    """4. Draws the minimum enclosing polygon around given points."""
    if len(points) > 2:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        cv2.polylines(img, [np.int32(hull)], True, (0, 255, 0), 2)  
        return hull
    return None

def draw_lines(image, closest_points):
    """5. Draws the grid lines on the board based on closest points."""
    top_left, top_right, bottom_right, bottom_left = closest_points

    for i in range(1, 8):
        start_x = int((1 - i/8) * top_left[0] + i/8 * top_right[0])
        start_y = int((1 - i/8) * top_left[1] + i/8 * top_right[1])
        end_x = int((1 - i/8) * bottom_left[0] + i/8 * bottom_right[0])
        end_y = int((1 - i/8) * bottom_left[1] + i/8 * bottom_right[1])
        cv2.line(image, (start_x, start_y), (end_x, end_y), (150, 0, 0), 1)

        start_x = int((1 - i/8) * top_left[0] + i/8 * bottom_left[0])
        start_y = int((1 - i/8) * top_left[1] + i/8 * bottom_left[1])
        end_x = int((1 - i/8) * top_right[0] + i/8 * bottom_right[0])
        end_y = int((1 - i/8) * top_right[1] + i/8 * bottom_right[1])
        cv2.line(image, (start_x, start_y), (end_x, end_y), (150, 0, 0), 1)

    return image

def add_labels(image, closest_points):
    """6. Adds labels to the board based on closest points."""
    text_color = (150, 0, 0)
    top_left, top_right, bottom_right, bottom_left = closest_points

    for i in range(8):
        start = (i + 1) % 2
        for j in range(start, 8, 2):
            cell_label = chr(65 + j) + str(8 - i)

            alpha = j / 8.0
            beta = i / 8.0

            x_pos = int((1 - alpha) * top_left[0] + alpha * top_right[0])
            y_pos = int((1 - beta) * top_left[1] + beta * bottom_left[1])

            cv2.putText(image, cell_label, (x_pos + 10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return image

def main():
    choice = input("Digite a opção:\n 1 - Imagem\n 2 - Vídeo\n 3 - Webcam Realtime\n")
    if choice == '1':
        path = input("Digite o caminho da imagem: ")
        img = cv2.imread(path)
        if img is not None:
            # 1. Detectar ArUcos
            aruco_corners, ids = find_aruco_markers(img)
            if ids is not None and aruco_corners:
                # 2. Encontrar pontos mais próximos ao centro
                closest_points = find_closest_point_to_center(img, aruco_corners)
                # 3. Associar pontos aos IDs dos ArUcos para garantir orientação e rotular
                labeled_points = associate_points_with_ids(closest_points, ids, img)
                # 4. Delimitar e desenhar o polígono mínimo
                hull = draw_min_polygon(img, closest_points)
                if hull is not None:
                    # 5. Desenhar linhas
                    img = draw_lines(img, closest_points)
                    # 6. Rotular casas brancas
                    img = add_labels(img, closest_points)
            cv2.imshow('Frame', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Imagem não encontrada. Verifique o caminho fornecido.")

if __name__ == "__main__":
    main()
