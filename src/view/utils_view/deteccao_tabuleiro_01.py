import cv2
import datetime
import os

def detect_board_edges(image_path):

   
    img = cv2.imread(image_path)

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  
    edges = cv2.Canny(gray, 100, 200)

  
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    largest_contour = max(contours, key=cv2.contourArea)

   
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img

if __name__ == "__main__":
 
    image_path = input("Enter the absolute path to the chessboard image: ")


    labeled_image = detect_board_edges(image_path)

 
    now = datetime.datetime.now()
    filename = f"src\\view\\frames\\img_{now.strftime('%Y%m%d_%H%M%S')}.jpg"

    cv2.imwrite(filename, labeled_image)

    print(f"Image saved to: {filename}")