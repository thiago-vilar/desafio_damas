from PIL import Image
from datetime import datetime

def resize_image():
    file_path = input("Digite o caminho do arquivo de imagem (relativo ou absoluto): ")
    try:
        with Image.open(file_path) as img:
            original_width, original_height = img.size
            print(f"A resolução atual da imagem é: {img.size} (largura x altura)")
            
            resolucoes = {
                1: (640, 480),
                2: (1280, 720),
                3: (1920, 1080),
                4: (2560, 1440),
                5: (3840, 2160),
                6: "Personalizada"
            }
            
            print("Escolha uma resolução para redimensionar a imagem:")
            for key, value in resolucoes.items():
                if key == 6:
                    print(f"{key}: Personalizada (digite a largura e a altura)")
                else:
                    print(f"{key}: {value} (largura x altura)")
            escolha = int(input("Digite o número correspondente à resolução desejada: "))
            
            if escolha == 6:
                largura = int(input("Digite a largura desejada: "))
                altura = int(input("Digite a altura desejada: "))
                nova_resolucao = (largura, altura)
            else:
                nova_resolucao = resolucoes.get(escolha)

            if nova_resolucao:
              
                scale = nova_resolucao[0] / original_width
                new_width = nova_resolucao[0]
                new_height = int(original_height * scale)
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
               
                if new_height > nova_resolucao[1]:
                    crop_top = (new_height - nova_resolucao[1]) // 2
                    img_cropped = img_resized.crop((0, crop_top, new_width, crop_top + nova_resolucao[1]))
                else:
                    img_cropped = img_resized

             
                current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                save_path = f"src\\view\\frame_img\\resized_image_{current_time}.jpg"
                img_cropped.save(save_path, "JPEG")
                print(f"Imagem redimensionada e cortada salva em: {save_path}")
            else:
                print("Escolha inválida, tente novamente.")
                
    except FileNotFoundError:
        print("Arquivo não encontrado. Por favor, verifique o caminho fornecido.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    resize_image()
