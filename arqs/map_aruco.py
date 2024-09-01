import cv2
from cv2 import aruco
import numpy as np

def find_aruco_markers(img, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
    
    return corners, ids

def draw_min_polygon(img, points):
    if len(points) > 2:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        cv2.polylines(img, [np.int32(hull)], True, (0, 255, 0), 2)

def find_closest_point_to_center(img, corners, ids):
    image_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    closest_points = []
    for corner, id in zip(corners, ids):
        distances = np.linalg.norm(corner[0] - image_center, axis=1)
        min_index = np.argmin(distances)
        closest_point = corner[0][min_index]
        closest_points.append(closest_point)
        cv2.circle(img, tuple(closest_point.astype(int)), 4, (0, 0, 255), -1)
        label = f'ID: {id[0]}'
        cv2.putText(img, label, tuple(closest_point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return closest_points

if __name__ == "__main__":
    choice = input("Digite a opção:\n1 - Imagem\n2 - Vídeo\n3 - Webcam Realtime\n")
    if choice == '1':
        path = input("Digite o caminho da imagem: ")
        img = cv2.imread(path)
        if img is not None:
            aruco_corners, ids = find_aruco_markers(img)
            if ids is not None:
                closest_points = find_closest_point_to_center(img, aruco_corners, ids)
                draw_min_polygon(img, closest_points)
            cv2.imshow('Frame', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Imagem não encontrada. Verifique o caminho fornecido.")

