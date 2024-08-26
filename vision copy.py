import numpy as np
import cv2

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Loop de captura contínua
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertendo a imagem para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definindo o intervalo para a cor branca no espaço HSV
    # Isso pode precisar de ajustes dependendo das condições de iluminação
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)

    # Criando a máscara para capturar áreas com cor branca
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Aplicando a máscara para obter o resultado final
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Exibindo o resultado
    cv2.imshow('frame', result)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpando os recursos
cap.release()
cv2.destroyAllWindows()
