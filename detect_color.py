import cv2
import numpy as np
import pyperclip

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
        alpha = (max - shadow) / 255
        gamma = shadow
        
        cal = cv2.addWeighted(input_img, alpha, input_img, 0, gamma)
    else:
        cal = input_img

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha = f
        gamma = 127 * (1 - f)

        cal = cv2.addWeighted(cal, alpha, cal, 0, gamma)
    return cal

def main():
    image_path = input("Digite o caminho completo ou relativo da imagem: ")
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Não foi possível abrir a imagem. Verifique se o caminho está correto.")
        return

    resize_factor = 0.6
    trackbar_window = 'Color Detectors'
    image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
    cv2.namedWindow(trackbar_window)

    cv2.createTrackbar('Low Hue', trackbar_window, 0, 255, nothing)
    cv2.createTrackbar('High Hue', trackbar_window, 0, 255, nothing)
    cv2.createTrackbar('Low Saturation', trackbar_window, 0, 255, nothing)
    cv2.createTrackbar('High Saturation', trackbar_window, 255, 255, nothing)
    cv2.createTrackbar('Low Value', trackbar_window, 0, 255, nothing)
    cv2.createTrackbar('High Value', trackbar_window, 255, 255, nothing)
    cv2.createTrackbar('Brightness', trackbar_window, 255, 2 * 255, nothing)
    cv2.createTrackbar('Contrast', trackbar_window, 127, 2 * 127, nothing)

    while True:

        lh = cv2.getTrackbarPos('Low Hue', trackbar_window)
        hh = cv2.getTrackbarPos('High Hue', trackbar_window)
        ls = cv2.getTrackbarPos('Low Saturation', trackbar_window)
        hs = cv2.getTrackbarPos('High Saturation', trackbar_window)
        lv = cv2.getTrackbarPos('Low Value', trackbar_window)
        hv = cv2.getTrackbarPos('High Value', trackbar_window)
        brightness = cv2.getTrackbarPos('Brightness', trackbar_window)
        contrast = cv2.getTrackbarPos('Contrast', trackbar_window)

        adjusted_image = apply_brightness_contrast(image, brightness, contrast)
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)

        lower_color = np.array([lh, ls, lv])
        upper_color = np.array([hh, hs, hv])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        result = cv2.bitwise_and(hsv, hsv, mask=mask)

        cv2.imshow(trackbar_window, cv2.cvtColor(result, cv2.COLOR_HSV2BGR))

        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            thresholds = f"Low: {lower_color.tolist()}, High: {upper_color.tolist()}"
            pyperclip.copy(thresholds)
            print("Thresholds copied to clipboard:", thresholds)
        elif k == 27:  
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
