import cv2
import numpy as np
from cv2 import aruco

def find_aruco_markers(img, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
    return corners, ids

def find_closest_point_to_center(img, corners):
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
    id_positions = [
        {'id': 169, 'label': 'P1', 'position': (0, 0)},
        {'id': 302, 'label': 'P2', 'position': (1, 0)},
        {'id': 876, 'label': 'P3', 'position': (1, 1)},
        {'id': 1001, 'label': 'P4', 'position': (0, 1)}
    ]

    id_map = {item['id']: item for item in id_positions}
    required_ids = set(id_map.keys())

    if not required_ids.issubset(set(ids.flatten())):
        print("Erro: Nem todos os ArUcos necessários foram encontrados.")
        return None

    ordered_points = [None] * 4
    for i, point in enumerate(closest_points):
        aruco_id = ids[i][0]
        if aruco_id in id_map:
            item = id_map[aruco_id]
            position_index = item['position'][0] + item['position'][1] * 2
            ordered_points[position_index] = point
            cv2.putText(img, f"{item['label']} ({aruco_id})", 
                        (int(point[0]), int(point[1]) + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if any(p is None for p in ordered_points):
        print("Erro: Falha ao associar corretamente os pontos aos IDs dos ArUcos.")
        return None

    return np.array(ordered_points)

def detect_colored_objects(img, green_thresholds, purple_thresholds, min_distance):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    lower_green = np.array(green_thresholds[0], dtype="uint8")
    upper_green = np.array(green_thresholds[1], dtype="uint8")
    lower_purple = np.array(purple_thresholds[0], dtype="uint8")
    upper_purple = np.array(purple_thresholds[1], dtype="uint8")

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def draw_contours(contours, color):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                point = np.array([cx, cy])

                if all(np.linalg.norm(point - np.array(center)) > min_distance for center in centers):
                    centers.append(point)
                    cv2.circle(img, (cx, cy), 10, color, -1)

    draw_contours(green_contours, (0, 255, 0))
    draw_contours(purple_contours, (128, 0, 128))

    return img

def apply_perspective_transform(img, src_points):
    dst_points = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, matrix, (480, 480))
    return warped, matrix

def detect_hough_lines(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

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
            cv2.line(warped, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return warped

def draw_lines_and_labels(warped):
    text_color = (50, 50, 50) 
    cell_size = 60 
    
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:  
                cell_label = chr(65 + j) + str(8 - i)
                x_pos = j * cell_size + cell_size // 2 - 15  
                y_pos = i * cell_size + cell_size // 2 + 15 
                cv2.putText(warped, cell_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)

    return warped

def resize_image(warped, scale_factor=1.5):
    return cv2.resize(warped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

def main():
    choice = input("Digite a opção:\n 1 - Imagem\n 2 - Vídeo\n 3 - Webcam Realtime\n")
    if choice == '1':
        path = input("Digite o caminho da imagem: ")
        img = cv2.imread(path)
        if img is not None:
            aruco_corners, ids = find_aruco_markers(img)
            if ids is not None and aruco_corners:
                closest_points = find_closest_point_to_center(img, aruco_corners)
                labeled_points = associate_points_with_ids(closest_points, ids, img)
                if labeled_points is not None:
                    aruco_polygon = np.array(labeled_points, dtype=np.int32)
                  
                    
                   
                    green_thresholds = ([85, 197, 31], [120, 255, 255])  
                    purple_thresholds = ([116, 92, 60], [211, 187, 183])  
                    min_distance = 50  
                    img_with_colors = detect_colored_objects( green_thresholds, purple_thresholds, min_distance)

                    warped, matrix = apply_perspective_transform(img_with_colors, labeled_points)
                    hough_image = detect_hough_lines(warped)
                    final_image = draw_lines_and_labels(hough_image)
                    resized_final_image = resize_image(final_image, scale_factor=1.5)
                    cv2.imshow('Warped Frame with Hough Lines, Labels, and Colored Objects', resized_final_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print("Imagem não encontrada. Verifique o caminho fornecido.")

if __name__ == "__main__":
    main()
