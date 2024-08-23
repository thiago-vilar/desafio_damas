import cv2
import numpy as np
from datetime import datetime
import os

def detect_chessboard_corners(img_path):

    if not os.path.isfile(img_path):
        print(f"Arquivo não encontrado: {img_path}")
        return None, None

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao carregar a imagem")
        return None, None

   
    pattern_size = (7, 7)  

  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not found:
        print("Cantos do tabuleiro não encontrados")
        return img, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return img, corners_refined

def draw_chessboard_border(img, corners):
    if corners is None:
        return img

    if corners.shape[0] != 7 * 7:
        print("Número de cantos detectados não corresponde ao padrão esperado")
        return img


    points = np.array([corners[0], corners[6], corners[-1], corners[-7]], dtype=np.float32).reshape((-1, 1, 2))
    color = (0, 255, 0)  

   
    img = cv2.polylines(img, [points.astype(int)], True, color, 2, cv2.LINE_AA)
    return img

def main():
    img_path = input("Digite o caminho completo da imagem: ") 

    img, corners = detect_chessboard_corners(img_path)
    
    if img is None or corners is None:
        return
    
    img_with_border = draw_chessboard_border(img, corners)
    
  
    output_dir = 'src/view/frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/img_{timestamp}.jpg'
    cv2.imwrite(filename, img_with_border)
    
    print(f"Imagem salva como {filename}")

if __name__ == "__main__":
    main()
