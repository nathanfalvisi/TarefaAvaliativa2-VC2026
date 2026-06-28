# Documentacao do Projeto

## Objetivo

Desenvolver uma aplicacao de Realidade Aumentada baseada em Visao Computacional capaz de identificar objetos em tempo real utilizando uma webcam.

## Funcionamento

1. Captura imagens da webcam.
2. Cada frame e enviado ao modelo YOLOv8.
3. O modelo identifica objetos presentes na imagem.
4. O sistema desenha caixas delimitadoras e exibe informacoes sobre cada objeto.
5. O usuario pode alterar modos de visualizacao durante a execucao.

## Principais Recursos

- Deteccao em tempo real
- Interface grafica com HUD
- Modos INFO, SIMPLES e DEBUG
- Ajuste dinamico da confianca
- Captura de screenshots

## Possiveis Melhorias

- Suporte a GPU (CUDA)
- Estimativa de distancia
- Reconhecimento de texto (OCR)
- Segmentacao de objetos
- Exportacao de resultados

## Aplicacoes

- Educacao
- Inventario
- Industria
- Robotica
- Realidade Aumentada
- Assistencia visual

## Conclusao

O projeto demonstra a utilizacao de modelos modernos de deteccao de objetos aliados ao OpenCV para criar uma aplicacao interativa e de facil utilizacao.
