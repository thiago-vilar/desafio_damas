"# desafio_damas" 

Neste projeto serão desenvolvidas técnicas de visão computacional clássica baseada em OpenCV-Python.
REVISADO 28/08/24

ESTRATÉGIA DE CODIFICAÇÃO:

Cirar uma classe estruturada UpdateBoard

    1. Achar arucos;
    2. Na área dos arucos achar pontos internos(closets) mais proximos ao centro da imagem;
    3. Desenhar o mínimo polígono com base nos 4 arucos(borda do tabuleiro):
        - Garantir a ordem dos pontos para mapeamento corretos(P1, P2, P3, P4) com mesma distância euclidiana para formar o quadrado do tabuleiro, contudo evitar mapeamento invertido, 'dobrado' (ex. P1, P3, P2, P4); OBS: tomar como base o posicionamento dos pixels na imagem
    4. Associar pontos aos ids para criar sistema de coordenadas:
    5. Detectar as linhas de hough dentro da área do tabuleiro;
    6. Detectar peças verdes e roxas usando limiar threshold HSV;
    7. Aplicação de wraped(deformada) + transformação de perspectiva;
    8. Desenho do tabuleiro:
        - Rotular casas brancas do tabuleiro
        - Identificar quais casas cada peça está
    9. Retorno da matriz numpy no terminal CLI/identificar onde estão as peças
        - Retornar da função opencv imshow
    10. Atualização do tabuleiro
        - Identificar uma jogada
        - Movimentação
        - Peça P move para casa C
        - Qual foi o jogador que fez o movimento
        - Alguma peça foi executada?
        - Definir qual jogador ganhou ao fim do vídeo

    PROGRAMA PRINCIPAL, ACEITAR ENTRADAS DE IMAGEM, VÍDEO E WEBCAM REALTIME