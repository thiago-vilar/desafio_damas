import cv2
import numpy as np
from cv2 import aruco

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
    reference_map = {474: (0, 0), 553: (1, 0), 424: (1, 1), 224: (0, 1)}

    required_ids = {474, 553, 424, 224}
    if not required_ids.issubset(set(ids.flatten())):
        print("Erro: Nem todos os ArUcos necessários foram encontrados.")
        return None

    ordered_points = [None] * 4
    for i, point in enumerate(closest_points):
        aruco_id = ids[i][0]
        label = id_map.get(aruco_id, f"ID{aruco_id}")
        position = reference_map.get(aruco_id)
        if position is not None:
            ordered_points[position[0] + position[1] * 2] = point
        labeled_points[label] = point
        cv2.putText(img, f"{label} ({aruco_id})", (int(point[0]), int(point[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if any(p is None for p in ordered_points):
        print("Erro: Falha ao associar corretamente os pontos aos IDs dos ArUcos.")
        return None

    return np.array(ordered_points)

def apply_perspective_transform(img, src_points):
    """Applies a perspective transform to the board based on the source points (ArUcos)."""
    dst_points = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, matrix, (400, 400))
    return warped, matrix

def detect_hough_lines(warped):
    """Detects lines in the warped image using the Hough Transform."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    line_image = np.copy(warped)
    hough_lines = []

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            hough_lines.append(((x1, y1), (x2, y2)))

    return line_image, hough_lines

def draw_lines_and_labels(warped, hough_lines):
    """Draws the grid lines and labels on the warped image based on the Hough lines."""
    text_color = (0, 0, 255)  
    cell_size = 50

    # Sort and find the grid intersection points
    vertical_lines = sorted([line for line in hough_lines if line[0][0] == line[1][0]], key=lambda x: x[0][0])
    horizontal_lines = sorted([line for line in hough_lines if line[0][1] == line[1][1]], key=lambda x: x[0][1])

    if len(vertical_lines) >= 8 and len(horizontal_lines) >= 8:
        for i in range(8):
            for j in range(8):
                x_pos = vertical_lines[i][0][0] + cell_size // 2
                y_pos = horizontal_lines[j][0][1] + cell_size // 2
                cell_label = chr(65 + i) + str(8 - j)
                cv2.putText(warped, cell_label, (x_pos - 10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return warped

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
                if labeled_points is not None:
                    # 4. Aplicar transformação de perspectiva
                    warped, matrix = apply_perspective_transform(img, labeled_points)
                    # 5. Detectar linhas usando Hough Transform
                    line_image, hough_lines = detect_hough_lines(warped)
                    # 6. Desenhar linhas e rótulos no tabuleiro
                    final_image = draw_lines_and_labels(line_image, hough_lines)
                    cv2.imshow('Warped Frame with Hough Lines', final_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print("Imagem não encontrada. Verifique o caminho fornecido.")

if __name__ == "__main__":
    main()
