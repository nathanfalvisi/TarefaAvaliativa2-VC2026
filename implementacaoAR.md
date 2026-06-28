# Documentação do Projeto

## Objetivo

Desenvolver uma aplicacao de Realidade Aumentada baseada em Visao Computacional capaz de identificar objetos em tempo real utilizando uma webcam.

## Funcionamento

1. Captura imagens da webcam.
2. Cada frame e enviado ao modelo YOLOv8.
3. O modelo identifica objetos presentes na imagem.
4. O sistema desenha caixas delimitadoras e exibe informacoes sobre cada objeto.
5. O usuário pode alterar modos de visualizacao durante a execucao.

## Foto de teste
[!Foto](img/test.png)

## Principais Recursos

- Detecção em tempo real
- Interface gráfica com HUD
- Modos INFO, SIMPLES e DEBUG
- Ajuste dinâmico da confianca
- Captura de screenshots

## Possíveis Melhorias

- Suporte a GPU (CUDA)
- Estimativa de distancia
- Reconhecimento de texto (OCR)
- Segmentacao de objetos
- Exportacao de resultados

## Aplicações

- Educação
- Inventario
- Industria
- Robótica
- Realidade Aumentada
- Assistência visual

## Conclusão

O projeto demonstra a utilizacao de modelos modernos de deteccao de objetos aliados ao OpenCV para criar uma aplicacao interativa e de facil utilizacao.
