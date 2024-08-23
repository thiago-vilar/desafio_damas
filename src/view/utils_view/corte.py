import os
import cv2
import numpy as np

class TabuleiroCutter:
    def __init__(self):
        self.input_dir = os.path.join(os.path.dirname(__file__), 'frame0')
        self.output_dir = os.path.join(os.path.dirname(__file__), 'frames')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada no caminho: {file_path}")
        return image

    def save_image(self, image):
        if image is not None:
            numero_imagens = len([f for f in os.listdir(self.output_dir) if f.startswith('corte') and f.endswith('.jpg')])
            nome_imagem = f'corte{numero_imagens + 1:02d}.jpg'
            save_path = os.path.join(self.output_dir, nome_imagem)
            cv2.imwrite(save_path, image)
            print(f"Imagem salva em: {save_path}")
        else:
            print("Nenhuma imagem para salvar.")

    def cortar_imagem(self, imagem, pontos):
    
        largura, altura = 600, 600
        dst = np.array([[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(pontos, dst)

        recorte = cv2.warpPerspective(imagem, M, (largura, altura))
        return recorte

    def processar_kinova(self, escolha_kinova):
        if escolha_kinova == '1':
            file_path = os.path.join(self.input_dir, 'webCam view K1.jpg')
            pontos = np.array([
                [1032, 1976],  
                [961, 450],    
                [2488, 420],   
                [2547, 1911]   
            ], dtype="float32")
        elif escolha_kinova == '2':
            file_path = os.path.join(self.input_dir, 'webCam view K2.jpg')
            pontos = np.array([
                [1187, 1871],  
                [1204, 351],   
                [2682, 360],   
                [2703, 1861]   
            ], dtype="float32")
        else:
            raise ValueError("Escolha inválida para Kinova. Escolha 1 ou 2.")
        
        imagem = self.load_image(file_path)
        recorte = self.cortar_imagem(imagem, pontos)
        self.save_image(recorte)

    def menu(self):
        while True:
            print("\nEscolha o Kinova:")
            print("[1] Kinova 1")
            print("[2] Kinova 2")
            print("[3] Sair")
            escolha_kinova = input("Escolha uma opção: ")

            if escolha_kinova == '3':
                break
            if escolha_kinova not in ['1', '2']:
                print("Opção inválida. Tente novamente.")
                continue

            try:
                self.processar_kinova(escolha_kinova)
                print("Imagem processada e salva com sucesso.")
            except Exception as e:
                print(f"Erro: {e}")

if __name__ == "__main__":
    app = TabuleiroCutter()
    app.menu()
