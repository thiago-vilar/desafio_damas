import cv2
from cv2 import aruco
import numpy as np

def find_aruco_markers(img, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, draw=True):
    """Detects ArUco markers in the provided image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)
    return corners, ids

def draw_min_polygon(img, points):
    """Draws the minimum enclosing polygon around given points."""
    if points and len(points) > 2:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        cv2.polylines(img, [np.int32(hull)], True, (0, 255, 0), 2)  
        return hull
    return None

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
    return closest_points

def detect_contrast_points(img, mask):
    """Detects high contrast points within the given mask using adaptive thresholding and goodFeaturesToTrack."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(equalized_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    corners = cv2.goodFeaturesToTrack(thresh, maxCorners=200, qualityLevel=0.01, minDistance=10, mask=mask)
    if corners is not None:
        for point in corners:
            pt = tuple(np.int32(point.ravel()))
            cv2.circle(img, pt, 3, (0, 255, 0), -1) 
    return corners

def detect_corners(img, max_corners=100, quality_level=0.01, min_distance=20):
    """Detects corners in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    return corners.astype(int) if corners is not None else None

def display_checkersboard_features(img, corners):
    """Draw lines between detected corners to visualize the chessboard structure."""
    if corners is not None:
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                corner1 = tuple(map(int, corners[i][0])) 
                corner2 = tuple(map(int, corners[j][0]))
                color = tuple(map(int, np.random.randint(0, 255, size=3)))
            

def main():
    choice = input("Digite a opção:\n 1 - Imagem\n 2 - Vídeo\n 3 - Webcam Realtime\n")
    if choice == '1':
        path = input("Digite o caminho da imagem: ")
        img = cv2.imread(path)
        if img is not None:
            aruco_corners, ids = find_aruco_markers(img)
            if ids is not None and aruco_corners:
                closest_points = find_closest_point_to_center(img, aruco_corners)
                hull = draw_min_polygon(img, closest_points)
                if hull is not None and len(hull) > 0: 
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(hull), 255)
                    contrast_points = detect_contrast_points(img, mask)
                    display_checkersboard_features(img, contrast_points)
                    
           
                    extra_corners = detect_corners(img)
                    if extra_corners is not None:
                        for corner in extra_corners:
                            x, y = corner.ravel()
                            cv2.circle(img, (x, y), 5, (255, 0, 0), -1) 
                    
            cv2.imshow('Frame', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Imagem não encontrada. Verifique o caminho fornecido.")

if __name__ == "__main__":
    main()
