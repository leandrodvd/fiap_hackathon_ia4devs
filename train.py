from ultralytics import YOLO

# Carregar modelo pré-treinado YOLOv8 (versão nano para eficiência)
model = YOLO("yolov8n.pt")  # Alternativa: usar "yolov8m.pt" para maior precisão

# Treinar o modelo
print("Treinando modelo...")
model.train(
    data="datasets/dataset.yaml",  # Caminho para o datasets.yaml
    epochs=50,            # Número de épocas (ajuste conforme necessário)
    imgsz=640,            # Tamanho das imagens
    batch=16,             # Tamanho do batch
    device="cpu"         # Usa GPU se disponível, senão usa CPU automaticamente
)

print("Treino finalizado...")
