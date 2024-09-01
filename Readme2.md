"# desafio_damas" 

Neste projeto serão desenvolvidas técnicas de visão computacional clássica baseada em OpenCV-Python.

DESAFIO:

a. Identificar Tag.
i. Cada Tag deverá conter a “posição” dela no mapa
1. Tag1 -> (0, 0)
2. Tag2 -> (X, 0)
3. Tag3 -> (X, Y)
4. Tag4 -> (0, Y)
b. Focar a imagem no tabuleiro
c. Identificar as casas do tabuleiro
d. Identificar qual jogador está de qual lado
e. Identificar todas as peças
i. Identificá-las de acordo com o formato/cor
ii. Identificar quais casas cada peça está
f. Identificar uma jogada
i. Movimentação
ii. Peça P move para casa C
iii. Qual foi o jogador que fez o movimento
iv. Alguma peça foi executada?
g. Definir qual jogador ganhou ao fim do vídeo

REVISADO 28/08/24
ESTRATÉGIA DE CODIFICAÇÃO:

Cirar uma classe estruturada UpdateBoard

    1. Achar arucos;
    2. Na área dos arucos achar pontos internos(closets) mais proximos ao centro da imagem;
    3. Desenhar o mínimo polígono com base nos 4 arucos(mapa da borda do tabuleiro):
        - Garantir a ordem dos pontos para mapeamento corretos(P1, P2, P3, P4) com mesma distância euclidiana para formar o quadrado do tabuleiro, contudo evitar mapeamento invertido, 'dobrado' (ex. P1, P3, P2, P4); OBS: tomar como base o posicionamento dos pixels na imaem
    4. Associar pontos aos ids para criar sistema de coordenadas:
    5. Detectar as linhas de hough dentro da área do tabuleiro;
    6. Detectar peças verdes e roxas usando limiar threshold;
    7. Aplicar transformação de perspectiva;
    8. Desenho do tabuleiro:
        - Rotular casas brancas do tabuleiro
        - Verificar se imagem real ou ilustração na função  opencv imshow;
    9. Retorno da matriz numpy no terminal CLI
        - Retornar da função opencv imshow

    PROGRAMA PRINCIPAL, ACEITAR ENTRADAS DE IMAGEM, VÍDEO E WEBCAM REALTIME

