import cv2
import numpy as np

def detect_colored_objects(image_path, green_thresholds, purple_thresholds, min_distance):
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem.")
        return


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
                    cv2.circle(image, (cx, cy), 10, color, -1)


    draw_contours(green_contours, (0, 255, 0))


    draw_contours(purple_contours, (128, 0, 128))

  
    cv2.imshow("Detected Colors", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

green_thresholds = ([80, 150, 80], [110, 250, 110])  
purple_thresholds = ([130, 50, 50], [160, 255, 255])  
min_distance = 60  

image_path = input("Digite o caminho do arquivo da imagem: ")
detect_colored_objects(image_path, green_thresholds, purple_thresholds, min_distance)
