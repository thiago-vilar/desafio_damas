"# desafio_damas" 

Neste projeto serão desenvolvidas técnicas de visão computacional clássica baseada em OpenCV-Python.
REVISADO 28/08/24

ESTRATÉGIA DE CODIFICAÇÃO:

Cirar um script de detecção de jogo de damas com inteligêcia artificial clássica usando OpenCV-Python

    1. Achar arucos;
    2. Na área dos arucos achar pontos internos(closets) mais proximos ao centro da imagem;
    3. Associar pontos internos(closest) aos ids dos arucos; 
    4. Causar transformação de perspectiva(warped);
    5. Detectar peças de 'damas' no tabuleiro;
    6. Desenho dos contornos;
    7. Desenhar linhas;
    8. Detectar status do tabuleiro;
    9. Calcular e disponibilizar a média da detecção do tabuleiro;
    10. Processar os frames do vídeo para retirar uma média para retorno Nympy no terminal;
    11. Processa os frames para retorno na janela imshow()
    12. Comparar os tabuleiros descarta os não detectados (sem visão dos arucos)
    13. Detecta movimentação de jogada
    14. Checa o vencedor do jogo.

    PROGRAMA PRINCIPAL,
    * Aceita entrada de vídeo
    * Faz resize e cropp
    * Exibe 3 saídas: 1. Janela com vídeo original; 2.Janela com tabuleiro; 3. Retorno da Matriz numpy no terminal com reconhecimento de jogadas, reconhecimento de peças capturadas e reconhecimento de vencedor.