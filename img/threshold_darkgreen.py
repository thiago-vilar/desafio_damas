from PIL import Image
import numpy as np

def dark_green_rgb_range(image_path, percentile):
    # Carrega a imagem
    img = Image.open(image_path)

    # Converte a imagem para um array numpy
    img_array = np.array(img)

    # Extrai o canal verde
    green_channel = img_array[:, :, 1]

    # Define o threshold baseado no percentil fornecido
    threshold = int(np.percentile(green_channel, percentile))

    # Encontrar os pixels que são considerados verde escuro
    dark_green_pixels = img_array[(green_channel <= threshold)]

    # Calcula o intervalo RGB para esses pixels
    min_rgb = dark_green_pixels.min(axis=0)
    max_rgb = dark_green_pixels.max(axis=0)

    return min_rgb, max_rgb

# Solicitando entrada do usuário para o caminho da imagem e o percentil
image_path = input("Por favor, insira o caminho da imagem: ")
percentile = float(input("Por favor, insira o percentil para definir 'verde escuro' (exemplo: 20): "))

# Calculando o intervalo RGB e imprimindo
min_rgb, max_rgb = dark_green_rgb_range(image_path, percentile)
print("O intervalo RGB para verde escuro é de mínimo:", min_rgb, "e máximo:", max_rgb)
