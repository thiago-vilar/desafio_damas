import cv2
import numpy as np

def dist(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

def detect_circles():
    image_path = input("Digite o caminho completo da imagem: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro: Não foi possível abrir a imagem.")
        return

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)

    circles = cv2.HoughCircles(blur_frame, cv2.HOUGH_GRADIENT, 1.2, 100, 
                               param1=100, param2=30, minRadius=75, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        prev_circle = None

        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            else:
                if prev_circle is not None:
                    if dist(chosen[0], chosen[1], prev_circle[0], prev_circle[1]) > dist(i[0], i[1], prev_circle[0], prev_circle[1]):
                        chosen = i

            cv2.circle(image, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3) 
            cv2.circle(image, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)  
            prev_circle = chosen

    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_circles()
