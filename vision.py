import cv2
from src.view.update_board import UpdateBoard

def capture_image():
    update_board = UpdateBoard()
    return update_board.capture_image()

def detect_pieces(image):
    update_board = UpdateBoard()
    update_board.processar_kinova(escolha_kinova='1', modo='2')  # Assumindo kinova 1 e modo 2 para captura de foto
    pieces = update_board.detectar_pecas(image)
    return pieces

def main():
    update_board = UpdateBoard()

    print("Escolha o Kinova:")
    print("[1] Kinova 1")
    print("[2] Kinova 2")
    escolha_kinova = input("Escolha uma opção: ")
    if escolha_kinova not in ['1', '2']:
        print("Opção inválida. Encerrando o programa.")
        return

    print("Pressione 'q' para sair, 'space' para atualizar o tabuleiro.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            update_board.processar_kinova(escolha_kinova=escolha_kinova, modo='2')
            update_board.game_update()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
