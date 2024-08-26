import cv2
import numpy as np
from cv2 import aruco

def find_aruco_markers(img, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, draw=True):
    """Detects ArUco markers in the provided image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
    return corners, ids

def find_closest_point_to_center(img, corners):
    """Finds the point closest to the center of the image for each ArUco marker."""
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

def associate_points_with_ids(corners, ids):
    """Associates corners with predefined board positions based on ArUco IDs."""
    position_map = {
        1: (0, 0),   # ID 1 (0, 0)
        2: (800, 0), # ID 2 (X, 0)
        3: (800, 800), # ID 3 (X, Y)
        4: (0, 800)  # ID 4  (0, Y)
    }
    src_points = np.zeros((4, 2), dtype=np.float32)
    for i, id in enumerate(ids.flatten()):
        if id in position_map:
            corner = corners[i][0]
            closest_point = corner[np.argmin(np.linalg.norm(corner - np.array([400, 400]), axis=1))] 
            src_points[i] = closest_point
        else:
            print(f"Unmapped ArUco ID: {id}")

    return src_points

def apply_perspective_transform(img, src_points):
    """Applies a perspective transform based on predefined positions."""
    dst_points = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype='float32') 
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, matrix, (800, 800))
    return warped, matrix


def detect_hough_lines(warped):
    """Detects lines in the warped image using the Hough Transform with improved filtering."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0) 
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
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
            # cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            hough_lines.append(((x1, y1), (x2, y2)))
    return line_image, hough_lines

def draw_lines_and_labels(warped, hough_lines):
    """Draws the grid lines and labels on the warped image."""
    text_color = (150, 150, 150)  
    cell_size = 100  
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1: 
                x_pos = j * cell_size + cell_size // 2 - 15
                y_pos = i * cell_size + cell_size // 2 + 15
                cell_label = chr(65 + j) + str(8 - i)
                cv2.putText(warped, cell_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    return warped

def main():
    choice = input("Digite a opção:\n 1 - Imagem\n 2 - Vídeo\n 3 - Webcam Realtime\n")
    if choice == '1':
        path = input("Digite o caminho da imagem: ")
        img = cv2.imread(path)
        if img is not None:
            aruco_corners, ids = find_aruco_markers(img)
            if ids is not None and aruco_corners:
                closest_points = find_closest_point_to_center(img, aruco_corners)
                ordered_points = closest_points[:4] if len(closest_points) >= 4 else None
                if ordered_points is not None:
                    warped, matrix = apply_perspective_transform(img, ordered_points)
                    line_image, hough_lines = detect_hough_lines(warped)
                    final_image = draw_lines_and_labels(line_image, hough_lines)
                    cv2.imshow('Warped Image with Grid and Labels', final_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Failed to order points correctly for perspective transform.")
            else:
                print("No ArUco markers found.")
        else:
            print("Image not found. Please check the path provided.")

if __name__ == "__main__":
    main()
