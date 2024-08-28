import cv2
import numpy as np
import pyperclip  # Importar a biblioteca para manipular a área de transferência

def nothing(x):
    pass

def apply_brightness_contrast(input_img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 / (510 - 0)) - 128)
    contrast = int((contrast - 127) * (127 / (254 - 127)))
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(input_img, al_pha,
                              input_img, 0, ga_mma)
    else:
        cal = input_img

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        al_pha = f
        ga_mma = 127 * (1 - f)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, al_pha,
                              cal, 0, ga_mma)

    return cal

def main():
    # Solicitar ao usuário o caminho da imagem
    image_path = input("Digite o caminho completo ou relativo da imagem: ")
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem. Verifique se o caminho está correto.")
        return

    # Configurações iniciais
    resize_factor = 0.5
    trackbar_window = 'Color Detectors'

    # Redimensionar a imagem para o tamanho desejado
    image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)

    # Criar janela para os trackbars
    cv2.namedWindow(trackbar_window)

    # Criar trackbars para ajustar a cor, brilho e contraste
    cv2.createTrackbar('Low Hue', trackbar_window, 0, 179, nothing)
    cv2.createTrackbar('High Hue', trackbar_window, 179, 179, nothing)
    cv2.createTrackbar('Low Saturation', trackbar_window, 0, 255, nothing)
    cv2.createTrackbar('High Saturation', trackbar_window, 255, 255, nothing)
    cv2.createTrackbar('Low Value', trackbar_window, 0, 255, nothing)
    cv2.createTrackbar('High Value', trackbar_window, 255, 255, nothing)
    cv2.createTrackbar('Brightness', trackbar_window, 255, 2 * 255, nothing)
    cv2.createTrackbar('Contrast', trackbar_window, 127, 2 * 127, nothing)

    while True:
        # Ler valores dos trackbars
        lh = cv2.getTrackbarPos('Low Hue', trackbar_window)
        hh = cv2.getTrackbarPos('High Hue', trackbar_window)
        ls = cv2.getTrackbarPos('Low Saturation', trackbar_window)
        hs = cv2.getTrackbarPos('High Saturation', trackbar_window)
        lv = cv2.getTrackbarPos('Low Value', trackbar_window)
        hv = cv2.getTrackbarPos('High Value', trackbar_window)
        brightness = cv2.getTrackbarPos('Brightness', trackbar_window)
        contrast = cv2.getTrackbarPos('Contrast', trackbar_window)

        # Aplicar brilho e contraste
        adjusted_image = apply_brightness_contrast(image, brightness, contrast)

        # Converter para HSV e aplicar threshold
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
        lower_color = np.array([lh, ls, lv])
        upper_color = np.array([hh, hs, hv])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        result = cv2.bitwise_and(adjusted_image, adjusted_image, mask=mask)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convertendo resultado para RGB

        # Mostrar resultado
        cv2.imshow(trackbar_window, result_rgb)

        # Copiar os valores para a área de transferência quando 'c' for pressionado
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            thresholds = f"Low: {lower_color.tolist()}, High: {upper_color.tolist()}"
            pyperclip.copy(thresholds)
            print("Thresholds copied to clipboard:", thresholds)
        elif k == 27:  # Aguardar até que a tecla 'ESC' seja pressionada
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
