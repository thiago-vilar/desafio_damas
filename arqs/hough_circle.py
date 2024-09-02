import cv2
import numpy as np

def detect_pieces():
    image_path = input("Digite o caminho completo da imagem: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Erro: Não foi possível abrir a imagem.")
        return

    image = cv2.resize(image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    
  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    green_thresholds = ([85, 197, 31], [120, 255, 255])
    purple_thresholds = ([116, 92, 60], [211, 187, 183])


    green_mask = cv2.inRange(hsv, np.array(green_thresholds[0]), np.array(green_thresholds[1]))
    purple_mask = cv2.inRange(hsv, np.array(purple_thresholds[0]), np.array(purple_thresholds[1]))


    combined_mask = cv2.add(green_mask, purple_mask)

 
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)


    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        area = cv2.contourArea(contour)
        if area > 100: 
            (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(approximation)
            if major_axis < 2 * minor_axis: 
                cv2.ellipse(image, (int(x), int(y)), (int(major_axis/2), int(minor_axis/2)), angle, 0, 360, (0, 255, 0), 2)

    cv2.imshow('Detected Pieces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_pieces()
