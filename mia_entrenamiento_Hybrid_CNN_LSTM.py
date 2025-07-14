'''
graph TD
    A[Input Video\nB×3×T×H×W] --> B[CNN 2D]
    B --> C[Features\nB×T×cnn_out×6×6]
    C --> D[LSTM Bidireccional]
    D --> E[Features Temporales\nB×T×lstm_hidden*2]
    E --> F[Mecanismo Atención]
    E --> G[Context Vector]
    F -->|Pesos| G
    G --> H[Clasificador]
    H --> I[Output\nB×num_classes]

    style A fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style D fill:#fbb,stroke:#333
    style F fill:#bfb,stroke:#333
    style H fill:#ffb,stroke:#333
'''
# Importa el módulo para suprimir o manejar advertencias
import warnings

# Importa el módulo para interactuar con el sistema de archivos y rutas
import os

# Importa el módulo para operaciones aleatorias, como selección o permutación
import random

# Importa OpenCV, usado para lectura y procesamiento de videos e imágenes
import cv2

# Importa PyTorch para construir y entrenar redes neuronales
import torch

# Importa funciones adicionales de PyTorch, como activaciones y pérdidas
import torch.nn.functional as F

# Importa NumPy, una biblioteca para manipulación de arreglos y operaciones matemáticas
import numpy as np

# Importa pandas para análisis y manipulación de datos tabulares
import pandas as pd

# Importa módulos de PyTorch para definir modelos y algoritmos de optimización
from torch import nn, optim

# Importa herramientas para crear conjuntos de datos y cargarlos en lotes
from torch.utils.data import DataLoader, Dataset

# Importa transformaciones de imágenes (escalado, recorte, normalización, etc.)
from torchvision import transforms

# Importa funciones para evaluación del modelo: matriz de confusión y reporte de clasificación
from sklearn.metrics import confusion_matrix, classification_report

# Importa matplotlib para visualización de datos y gráficos
import matplotlib.pyplot as plt

# Importa glob para búsqueda de archivos con patrones (como *.mp4)
import glob

# Importa time para medir duraciones o registrar timestamps
import time

# Ignorar todos los warnings para evitar mensajes molestos en consola
warnings.filterwarnings("ignore")

# Clase personalizada que hereda de torch.utils.data.Dataset para cargar videos como secuencias de frames
class VideoDataset(Dataset):
    # Constructor de la clase: recibe ruta de los videos, transformaciones, cantidad de frames y si es entrenamiento
    def __init__(self, directory, transform=None, frame_sample=30, is_train=False):
        self.directory = directory  # Directorio raíz que contiene subcarpetas por clase
        self.transform = transform  # Transformaciones opcionales para aumento de datos
        self.frame_sample = frame_sample  # Número de frames a muestrear por video
        self.is_train = is_train  # Indica si es conjunto de entrenamiento (para aplicar transformaciones)
        self.videos = []  # Lista de rutas a videos
        self.labels = []  # Lista de etiquetas correspondientes

        # Recorre las carpetas de clases: 'Violence' y 'NonViolence'
        for label in ['Violence', 'NonViolence']:
            path = os.path.join(self.directory, label)  # Construye la ruta a la carpeta de clase
            for video_file in os.listdir(path):  # Itera sobre los archivos en esa carpeta
                if video_file.endswith('.mp4'):  # Solo se consideran archivos .mp4
                    self.videos.append(os.path.join(path, video_file))  # Guarda la ruta completa del video
                    self.labels.append(1 if label == 'Violence' else 0)  # Etiqueta 1 para violencia, 0 para no

    # Devuelve la cantidad de muestras en el dataset
    def __len__(self):
        return len(self.videos)

    # Devuelve un video y su etiqueta correspondiente dado un índice
    def __getitem__(self, idx):
        video_path = self.videos[idx]  # Ruta del video actual
        label = self.labels[idx]  # Etiqueta correspondiente
        cap = cv2.VideoCapture(video_path)  # Abre el video con OpenCV
        frames = []  # Lista para almacenar los frames extraídos

        # Extrae los frames del video uno por uno
        while cap.isOpened():
            ret, frame = cap.read()  # Lee un frame
            if not ret:  # Si no hay más frames, termina el bucle
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB
            frame = cv2.resize(frame, (224, 224))  # Redimensiona el frame a 224x224
            frame = torch.from_numpy(frame).float() / 255.0  # Convierte a tensor float normalizado [0, 1]
            frame = frame.permute(2, 0, 1)  # Cambia el orden de dimensiones: [H, W, C] -> [C, H, W]

            # Normaliza con los valores promedio y desviación estándar de ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame = (frame - mean) / std  # Aplica la normalización

            frames.append(frame)  # Añade el frame procesado a la lista

        cap.release()  # Cierra el video

        # Si hay más frames que los necesarios, se seleccionan equidistantes
        if len(frames) > self.frame_sample:
            indices = np.linspace(0, len(frames)-1, num=self.frame_sample, dtype=int)
            frames = [frames[i] for i in indices]
        # Si hay menos frames, se repite el último frame hasta completar el número requerido
        elif len(frames) < self.frame_sample:
            frames += [frames[-1]] * (self.frame_sample - len(frames))

        # Aumento de datos solo si es entrenamiento y pasa la probabilidad aleatoria
        if self.transform and self.is_train and random.random() > 0.5:
            frames = [self.transform(frame) for frame in frames]

        # Convierte la lista de frames a tensor y ajusta dimensiones
        frames = torch.stack(frames)  # Forma: [T, C, H, W]
        frames = frames.permute(1, 0, 2, 3)  # Reordena a [C, T, H, W] para modelos 3D CNN

        return frames, label  # Devuelve el tensor de frames y su etiqueta

# Define una clase de red neuronal que combina una CNN con una LSTM y atención
class Hybrid_CNN_LSTM(nn.Module):
    # Constructor del modelo: recibe el número de clases, tamaño de salida del CNN, tamaño oculto del LSTM y número de capas LSTM
    def __init__(self, num_classes=2, cnn_out=128, lstm_hidden=128, lstm_layers=2):
        super().__init__()  # Llama al constructor de la clase base nn.Module
        
        # Guarda parámetros de configuración como atributos de clase
        self.cnn_out = cnn_out
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Define los bloques convolucionales usando una secuencia de capas
        self.cnn = nn.Sequential(
            self._make_conv_block(3, 32),         # Primer bloque: de 3 canales (RGB) a 32 filtros
            self._make_conv_block(32, 64),        # Segundo bloque
            self._make_conv_block(64, 128),       # Tercer bloque
            self._make_conv_block(128, cnn_out, pool=False),  # Cuarto bloque sin maxpooling
            nn.AdaptiveAvgPool2d((6, 6))          # Reduce la salida a tamaño fijo (6x6)
        )

        # Define una LSTM bidireccional para procesar secuencias de frames
        self.lstm = nn.LSTM(
            input_size=cnn_out * 6 * 6,  # Entrada: vector aplanado por frame (output del CNN)
            hidden_size=lstm_hidden,     # Tamaño del estado oculto
            num_layers=lstm_layers,      # Número de capas LSTM apiladas
            bidirectional=True,          # Activa la bidireccionalidad (permite contexto futuro y pasado)
            batch_first=True,            # Entrada con forma (batch, time, features)
            dropout=0.3 if lstm_layers > 1 else 0  # Dropout solo si hay más de una capa
        )

        # Módulo de atención para ponderar la importancia de cada paso temporal
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),  # Reduce dimensión después de concatenar direcciones
            nn.Tanh(),                                # Activación no lineal
            nn.Linear(lstm_hidden, 1, bias=False)     # Produce un peso escalar por paso de tiempo
        )

        # Capa final de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),  # Reduce dimensión (doble porque LSTM es bidireccional)
            nn.BatchNorm1d(lstm_hidden),              # Normalización por lotes
            nn.LeakyReLU(0.2),                        # Activación LeakyReLU
            nn.Dropout(0.5),                          # Dropout para regularización
            nn.Linear(lstm_hidden, num_classes)       # Capa de salida con tantas unidades como clases
        )

        # Inicializa los pesos de forma personalizada
        self._init_weights()

    # Método auxiliar para crear un bloque convolucional con BatchNorm, activación y (opcional) MaxPool
    def _make_conv_block(self, in_c, out_c, pool=True):
        block = [
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),  # Convolución 2D con kernel 3x3
            nn.BatchNorm2d(out_c),                             # Normalización por lotes
            nn.LeakyReLU(0.2, inplace=True)]                   # Activación LeakyReLU

        if pool:
            block.append(nn.MaxPool2d(2, ceil_mode=True))      # Agrega MaxPooling con redondeo hacia arriba

        return nn.Sequential(*block)  # Devuelve el bloque como una secuencia

    # Método auxiliar para inicializar los pesos de las capas del modelo
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Inicialización He (Kaiming) para convoluciones con LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Inicialización estándar para BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Inicialización Xavier para capas lineales
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # Método de propagación hacia adelante del modelo
    def forward(self, x):
        # x tiene forma [batch, channels, frames, height, width]
        batch_size = x.size(0)
        seq_length = x.size(2)

        # Reorganiza y aplana los frames para pasarlos por la CNN
        c_in = x.permute(0, 2, 1, 3, 4)  # Reordena a [batch, frames, C, H, W]
        c_in = c_in.reshape(-1, x.size(1), x.size(3), x.size(4))  # Aplana batch y frames: [batch*frames, C, H, W]

        c_out = self.cnn(c_in)  # Pasa por CNN: [batch*frames, cnn_out, 6, 6]
        c_out = c_out.view(batch_size, seq_length, -1)  # Reestructura a [batch, frames, features]

        # Pasa por la LSTM bidireccional
        lstm_out, _ = self.lstm(c_out)  # [batch, frames, lstm_hidden*2]

        # Mecanismo de atención: produce una ponderación por cada frame
        attention_scores = self.attention(lstm_out)  # [batch, frames, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # Softmax sobre frames
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # Suma ponderada: [batch, lstm_hidden*2]

        # Clasificación final
        output = self.classifier(context_vector)

        return output

if __name__ == "__main__":

    # Ruta para guardar resultados de métricas por época
    output_path = '/home/michell-alvarez/e_modelos/EF/archivos_txt/resultados_epoch.txt'

    # Inicialización de variables para early stopping
    best_accuracy = 0.0               # Mejor accuracy observado
    epochs_no_improve = 0             # Conteo de épocas sin mejora
    early_stop_limit = 7              # Límite de paciencia para detener entrenamiento
    min_delta = 0.005                 # Mejora mínima requerida para reiniciar contador

    # Selección del dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directorios de entrenamiento, prueba y nuevos datos
    train_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/Train'
    test_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/Test'
    new_data_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/NewData'

    # Definición de transformaciones para los frames (aumento de datos y normalización)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                        # Volteo aleatorio
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Variación de color
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),     # Rotación y traslación aleatoria
        transforms.Resize((224, 224)),                                 # Redimensionado a 224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406],               # Normalización con media y std de ImageNet
                             std=[0.229, 0.224, 0.225])
    ])

    # Carga de datasets con transformaciones
    train_dataset = VideoDataset(train_dir, transform)
    test_dataset = VideoDataset(test_dir, transform)

    # Preparación de DataLoaders para entrenamiento y prueba
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)



    # Instancia el modelo Hybrid_CNN_LSTM y lo mueve al dispositivo (GPU o CPU)
    model = Hybrid_CNN_LSTM().to(device)

    # Busca archivos de checkpoint guardados con el patrón especificado (ordenados por época)
    checkpoint_files = glob.glob('/home/michell-alvarez/e_modelos/EF/model_checkpoint_ef_epoch_*.pth')

    # Configura el optimizador AdamW con regularización L2 (weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Define un scheduler OneCycleLR para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,                    # Tasa de aprendizaje máxima
        steps_per_epoch=len(train_loader),  # Número de pasos por época
        epochs=50,                          # Número total de épocas
        pct_start=0.3                       # Porcentaje del ciclo en el que se alcanza la tasa máxima
    )

    # Función de pérdida con suavizado de etiquetas para reducir sobreajuste y ruido en las etiquetas
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Requiere PyTorch 1.10 o superior

    # Si se encuentran archivos de checkpoint
    if checkpoint_files:
        # Extrae la época de cada archivo usando el nombre del archivo
        checkpoints = [(int(f.split('_')[-1].split('.')[0]), f) for f in checkpoint_files]

        # Selecciona el checkpoint con la época más alta
        last_epoch, checkpoint_path = max(checkpoints, key=lambda x: x[0])

        # Carga el checkpoint desde el archivo
        checkpoint = torch.load(checkpoint_path)

        # Restaura el estado del modelo y del optimizador
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Determina la época a partir de la cual se continuará el entrenamiento
        start_epoch = checkpoint['epoch'] + 1

        # Recupera información opcional del checkpoint
        train_loss = checkpoint.get('train_loss', None)
        train_accuracy = checkpoint.get('train_accuracy', None)

        print(f"Checkpoint loaded from {checkpoint_path}. Starting from epoch {start_epoch}")
        if train_loss is not None:
            print(f"Train Loss loaded: {train_loss}")
        if train_accuracy is not None:
            print(f"Train Accuracy loaded: {train_accuracy}")

    else:
        # Si no se encontró checkpoint, iniciar desde cero
        start_epoch = 0
        print("No checkpoint found. Starting from epoch 0.")

# Itera desde la época cargada (o 0) hasta la 49 (50 épocas en total)
for epoch in range(start_epoch, 50):
    # Pone el modelo en modo entrenamiento
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    # Recorre cada lote del DataLoader de entrenamiento
    for inputs, labels in train_loader:
        # Mueve los datos al dispositivo (GPU o CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Reinicia los gradientes del optimizador
        optimizer.zero_grad()

        # Propagación hacia adelante
        outputs = model(inputs)

        # Cálculo de la pérdida
        loss = criterion(outputs, labels)

        # Propagación hacia atrás
        loss.backward()

        # Actualización de pesos
        optimizer.step()

        # Acumula pérdida para promedio
        running_loss += loss.item()

        # Obtiene las predicciones
        _, predicted = torch.max(outputs.data, 1)

        # Acumula totales y aciertos para cálculo de precisión
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Cálculo de pérdida y precisión promedio de entrenamiento
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # Imprime resultados de la época actual
    print(f'Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # Cambia el modelo a modo evaluación (desactiva dropout, batchnorm en modo entrenamiento)
    model.eval()

    test_loss = 0.0
    total = 0
    correct = 0


    # Desactiva el cálculo de gradientes para acelerar la inferencia y ahorrar memoria
    with torch.no_grad():
        # Itera sobre los lotes del conjunto de prueba
        for inputs, labels in test_loader:
            # Mueve los datos al dispositivo (GPU o CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Propagación hacia adelante
            outputs = model(inputs)

            # Cálculo de la pérdida de prueba
            loss = criterion(outputs, labels)

            # Acumula la pérdida total de prueba
            test_loss += loss.item()

            # Obtiene las predicciones con mayor probabilidad
            _, predicted = torch.max(outputs.data, 1)

            # Acumula el total de ejemplos
            total += labels.size(0)

            # Cuenta cuántas predicciones fueron correctas
            correct += (predicted == labels).sum().item()

        # Calcula la pérdida promedio en validación
        test_loss /= len(test_loader)

        # Calcula la precisión de prueba
        test_accuracy = correct / total

        # Muestra los resultados de la época actual
        print(f'Epoch [{epoch+1}/50], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Guarda un checkpoint del modelo en cada época (se puede ajustar a guardar cada N épocas)
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,  # Época actual
                'model_state_dict': model.state_dict(),  # Pesos del modelo
                'optimizer_state_dict': optimizer.state_dict(),  # Estado del optimizador
                'test_loss': test_loss,  # Pérdida de prueba
                'test_accuracy': test_accuracy,  # Precisión de prueba
            }, f"model_checkpoint_ef_epoch_test_{epoch + 1}.pth")  # Nombre del archivo de checkpoint

            print(f"Checkpoint saved at epoch {epoch+1}")  # Confirmación por consola

        # Guarda las métricas de la época actual en un archivo de texto
        with open(output_path, 'a') as f:
            f.write(f'Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n')
            f.write(f'Epoch [{epoch+1}/50], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n')


        # Ajusta la tasa de aprendizaje según la pérdida de prueba
        scheduler.step(test_loss)