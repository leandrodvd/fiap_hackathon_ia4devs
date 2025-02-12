# fiap_ia4devs_hackathon
Projeto para a hackathon da pos graduação IA para desenvolvedores da FIAP

## Descrição do Projeto

O objetivo é criar uma solução que reconheça a presença de armas brancas (ex. facas) em um video e envie um email de alerta quando ocorrer a identificação.

## Dados

Para treinamento de um modelo de detecção de objetos, utilizamos o dataset disponível no Kaggle: [Weapon Detection Dataset for YOLOv5](https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5/)

## Modelo

O modelo foi treinado utilizando o Ultralitics YOLOv8, detalhes em https://docs.ultralytics.com/pt


## Como executar o treinamento com o dataset selecionado

Para executar o treinamento do modelo execute

```python
python train.py
```
Para referência o treinamento leva cerca de 5 horas em um computador com processador Intel Core i7 sem GPU.

O modelo treinado está disponível no arquivo `runs/detect/train/weights/best.pt`

## Como executar a detecção de objetos em um video

Para executar a detecção de objetos em um video execute

```python
python analyze.py <caminho parar o arquivo de video.mp4> <mailgun api key> <mailgun domain> <email alert destination>
```
exemplo:
```python
python analyze.py test-video1.mp4 XXXYYYY xxxyyy@mailgun.cc destination@gmail.com
```