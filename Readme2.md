"# desafio_damas" 

Neste projeto serão desenvolvidas técnicas de visão computacional clássica baseada em OpenCV, abordando também técnicas de detecção via ArUco.


1 - achar arucos;
2 - achar os pontos internos(closets) mais proximos ao centro da imagem
3 - associar pontos(closets) ao id do arucos para garantir a orientação do tauleiro siga a mesma marcação com rótulos nos pontos(P1,P2,P3,P4)
4 - delimitar e desenhar o o polígono mínimo ('draw_min_polygon')
5 - desenho de linhas(labels) usando a função de houges
6 - rotular casas brancas

cv2. getcreateTrackbar("L-H", "Trackbars", 0, 255, nothing)
l_h = cv2.getTrackbarPos("L-H", Trackbars")



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