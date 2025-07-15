'''
graph TD
    A[Input Video\n3×T×H×W] --> B[Backbone 3D CNN]
    B --> C[Features\n256×T×7×7]
    C --> D[Temporal Attention]
    C --> E[Weighted Sum]
    D -->|Attention Weights| E
    E --> F[Classifier]
    F --> G[Output\nnum_classes]
'''

# Importa el módulo `warnings` para controlar o suprimir advertencias en tiempo de ejecución
import warnings
# Importa el módulo `os` para operaciones del sistema operativo, como manipulación de archivos
import os
# Importa OpenCV para operaciones de procesamiento de video e imagen
import cv2
# Importa PyTorch, una biblioteca para redes neuronales
import torch
# Importa NumPy para manejo de arrays y operaciones matemáticas
import numpy as np
# Importa los submódulos de PyTorch necesarios para definir modelos y entrenarlos
from torch import nn, optim
# Importa herramientas para manejar datasets personalizados y cargar datos por lotes
from torch.utils.data import DataLoader, Dataset
# Importa transformaciones estándar para imágenes desde torchvision
from torchvision import transforms
# Importa utilidades para entrenamiento con precisión mixta (mixed precision)
from torch.cuda.amp import GradScaler, autocast
# Importa función para dividir datos en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Importa función para generar reporte de métricas de clasificación
from sklearn.metrics import classification_report
# Importa PIL para manipulación de imágenes
from PIL import Image
# Importa `random` para realizar muestreo aleatorio o reproducibilidad
import random
# Importa funciones adicionales como activaciones desde PyTorch
import torch.nn.functional as F
# Importa tqdm para mostrar barras de progreso durante ciclos largos
from tqdm import tqdm
# Importa time para medir tiempos de ejecución
import time
# Importa `make_dot` para visualizar arquitecturas de redes neuronales
from torchviz import make_dot
# Importa hiddenlayer para visualizar y depurar arquitecturas de redes
import hiddenlayer as hl
# Importa función para generar matrices de confusión
from sklearn.metrics import confusion_matrix
# (Comentado) Importaría seaborn, biblioteca para visualización estadística
# import seaborn as sns
# Importa matplotlib para generar gráficos
import matplotlib.pyplot as plt
# Importa pandas para análisis y manipulación de datos tabulares
import pandas as pd
# (Comentado) Importaría dataframe_image para exportar DataFrames como imágenes
# import dataframe_image as dfi

# Desactiva la visualización de advertencias para evitar saturar la salida con mensajes irrelevantes
warnings.filterwarnings("ignore")

# Selecciona el dispositivo a usar: GPU si está disponible, de lo contrario CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Activa una optimización en cudnn para mejorar el rendimiento en modelos estáticos
torch.backends.cudnn.benchmark = True
# Fija la semilla de PyTorch para resultados reproducibles
torch.manual_seed(42)
# Fija la semilla de NumPy para garantizar la reproducibilidad
np.random.seed(42)

# Clase personalizada para el dataset, hereda de Dataset de PyTorch
class ViolenceDataset(Dataset):
    # Constructor de la clase: recibe rutas de videos, etiquetas, transformaciones, longitud de clip y tamaño del frame
    def __init__(self, video_paths, labels, transform=None, clip_len=30, frame_size=112):
        self.video_paths = video_paths  # Lista con rutas a los videos
        self.labels = labels            # Lista con etiquetas asociadas a cada video
        self.transform = transform      # Transformaciones a aplicar a cada frame
        self.clip_len = clip_len        # Número de frames por clip de video
        self.frame_size = frame_size    # Tamaño al que se redimensionará cada frame

    # Devuelve el número total de muestras del dataset
    def __len__(self):
        return len(self.video_paths)

    # Devuelve la muestra en la posición `idx`
    def __getitem__(self, idx):
        # Carga el video en la ruta correspondiente
        cap = cv2.VideoCapture(self.video_paths[idx])
        frames = []

        # Obtiene el número total de frames del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Si hay suficientes frames, selecciona clip_len frames aleatorios sin repetición
        if total_frames >= self.clip_len:
            indices = sorted(random.sample(range(total_frames), self.clip_len))
        # Si no hay suficientes, usa todos los frames disponibles
        else:
            indices = list(range(total_frames))

        # Recorre los índices seleccionados
        for i in indices:
            # Posiciona el video en el frame i
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()  # Lee el frame
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte BGR a RGB
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))  # Redimensiona
                frame = Image.fromarray(frame)  # Convierte a objeto PIL para aplicar transformaciones

                # Si se definió transformación, la aplica
                if self.transform:
                    frame = self.transform(frame)
                # Si no, convierte a tensor y normaliza manualmente
                else:
                    frame = transforms.ToTensor()(frame)
                    frame = transforms.Normalize([0.432, 0.398, 0.377], [0.228, 0.224, 0.225])(frame)

                # Agrega el frame procesado a la lista
                frames.append(frame)

        cap.release()  # Libera el recurso del video

        # Si no se alcanzó clip_len, se rellenan con tensores de ceros
        while len(frames) < self.clip_len:
            frames.append(torch.zeros(3, self.frame_size, self.frame_size))

        # Devuelve el clip como tensor de forma (C, T, H, W) y su etiqueta
        return torch.stack(frames[:self.clip_len], dim=1).float(), torch.tensor(self.labels[idx], dtype=torch.long)

# Clase del modelo de detección de violencia, hereda de nn.Module
class ViolenceDetector(nn.Module):
    # Constructor del modelo
    def __init__(self, num_classes=2):
        super().__init__()  # Llama al constructor de la clase base

        # Bloque de extracción de características con convoluciones 3D
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1,3,3), padding=(0,1,1)),  # Conv3D con kernel 1x3x3
            nn.BatchNorm3d(64),     # Normalización por lotes
            nn.GELU(),              # Activación GELU
            nn.MaxPool3d((1,2,2)),  # Max pooling espacial

            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),  # Conv3D con kernel 3x3x3
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((None, 7, 7))  # Reduce las dimensiones espaciales a 7x7
        )

        # Módulo de atención temporal: decide qué frames son más relevantes
        self.temp_attention = nn.Sequential(
            nn.Linear(256*7*7, 256),  # Proyecta a espacio intermedio
            nn.GELU(),                # Activación no lineal
            nn.Dropout(0.3),          # Dropout para regularización
            nn.Linear(256, 1)         # Produce un peso de atención por frame
        )

        # Capa de clasificación final
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 512),  # Capa totalmente conectada
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)  # Capa final con número de clases
        )

    # Propagación hacia adelante
    def forward(self, x):
        x = x.float()  # Asegura que el tipo de dato sea float
        features = self.backbone(x)  # Extrae características (B, C, T, H, W)
        B, C, T, H, W = features.shape  # Desempaqueta dimensiones
        features = features.permute(0, 2, 1, 3, 4).reshape(B, T, C*H*W)  # Reordena y aplana (B, T, D)
        attn_weights = torch.softmax(self.temp_attention(features), dim=1)  # Calcula pesos de atención (B, T, 1)
        context = (features * attn_weights).sum(dim=1)  # Calcula la representación contextual (B, D)
        return self.classifier(context)  # Pasa al clasificador final


# Función que carga rutas y etiquetas de videos desde un directorio organizado por clases
def cargar_datos_desde_directorio(directorio):
    paths, etiquetas = [], []  # Listas vacías para almacenar rutas de video y etiquetas
    for clase in ['Violence', 'NonViolence']:  # Recorre las dos clases
        clase_dir = os.path.join(directorio, clase)  # Construye la ruta completa para la clase
        for archivo in os.listdir(clase_dir):  # Itera sobre los archivos del directorio de la clase
            if archivo.endswith('.mp4'):  # Solo se consideran archivos de video .mp4
                paths.append(os.path.join(clase_dir, archivo))  # Agrega la ruta del video
                etiquetas.append(1 if clase == 'Violence' else 0)  # Etiqueta: 1 para violencia, 0 para no violencia
    return paths, etiquetas  # Retorna las rutas y las etiquetas como listas

# Función principal que entrena el modelo de detección de violencia
def train_model():
    # Define las rutas para los conjuntos de entrenamiento, prueba y salida de resultados
    train_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/Train'
    test_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/Test'
    output_path = '/home/michell-alvarez/e_modelos/EF/archivos_txt/resultados.txt'

    # Carga rutas y etiquetas de los conjuntos de entrenamiento y validación
    train_paths, train_labels = cargar_datos_desde_directorio(train_dir)
    val_paths, val_labels = cargar_datos_desde_directorio(test_dir)

    # Transformaciones para aumentar la robustez del modelo (solo para entrenamiento)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Volteo horizontal aleatorio
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variación de brillo y contraste
        transforms.RandomResizedCrop(size=112, scale=(0.8, 1.0)),  # Recorte y reescalado aleatorio
        transforms.ToTensor(),  # Conversión a tensor
        transforms.Normalize([0.432, 0.398, 0.377], [0.228, 0.224, 0.225])  # Normalización por canal
    ])

    # Transformaciones para validación (más conservadoras)
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Redimensionado fijo
        transforms.ToTensor(),
        transforms.Normalize([0.432, 0.398, 0.377], [0.228, 0.224, 0.225])
    ])

    # Crea instancias de los datasets personalizados
    train_dataset = ViolenceDataset(train_paths, train_labels, train_transform)
    val_dataset = ViolenceDataset(val_paths, val_labels, val_transform)

    # Calcula pesos para cada clase con el fin de manejar desbalance
    class_counts = np.bincount(train_labels)  # Cuenta la cantidad de ejemplos por clase
    pos_weight = torch.tensor([class_counts[1]/class_counts[0], 1.0], dtype=torch.float32).to(device)
    # Define la función de pérdida con pesos y suavizado de etiquetas (label smoothing)
    criterion = nn.CrossEntropyLoss(weight=pos_weight, label_smoothing=0.1)

    # Carga los datos en lotes (batch) usando DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    # Inicializa el modelo y lo mueve al dispositivo (GPU/CPU)
    model = ViolenceDetector().to(device)

    # Genera un grafo visual del modelo con entrada ficticia para inspección
    dummy_input = torch.randn(1, 3, 30, 112, 112).to(device)  # Entrada de ejemplo con dimensiones del modelo
    output = model(dummy_input)  # Propagación hacia adelante para generar salida
    dot = make_dot(output, params=dict(model.named_parameters()))  # Crea grafo visual con parámetros
    dot.render("model_architecture_make_dot", format="png")  # Guarda el grafo como imagen PNG

    # Define el optimizador con regularización L2 (weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Planificador de tasa de aprendizaje (scheduler) con forma de coseno
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Inicializa el escalador para entrenamiento con precisión mixta
    scaler = GradScaler()

    # Variables para early stopping
    best_acc = 0.0  # Mejor accuracy validado hasta el momento
    early_stop_patience = 7  # Número máximo de épocas sin mejora permitidas
    min_delta = 0.005  # Cambio mínimo de mejora para considerar progreso
    patience_counter = 0  # Contador de épocas sin mejora

    # Repite el entrenamiento por 50 épocas como máximo
    for epoch in range(50):
        # Cambia el modelo a modo entrenamiento
        model.train()
        # Inicializa pérdida acumulada y contadores de aciertos
        train_loss = 0.0
        correct = 0
        total = 0

        # Itera sobre lotes de entrenamiento
        for inputs, labels in train_loader:
            # Mueve los datos al dispositivo (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            # Reinicia los gradientes acumulados
            optimizer.zero_grad()

            # Usa autocast para precisión mixta (más eficiente en GPU)
            with autocast():
                outputs = model(inputs)  # Propagación hacia adelante
                loss = criterion(outputs, labels)  # Cálculo de la pérdida

            # Backpropagation con escalado de gradiente
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Actualiza los pesos
            scaler.update()  # Actualiza el escalador

            # Acumula la pérdida
            train_loss += loss.item()
            # Obtiene la predicción con mayor probabilidad
            _, predicted = outputs.max(1)
            total += labels.size(0)  # Número total de muestras
            correct += predicted.eq(labels).sum().item()  # Número de predicciones correctas

        # Calcula precisión de entrenamiento
        train_acc = correct / total

        # Cambia a modo evaluación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Desactiva el cálculo de gradientes para validación
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calcula precisión de validación
        val_acc = correct / total
        # Actualiza el scheduler (planificador de tasa de aprendizaje)
        scheduler.step()

        # Si la validación mejora lo suficiente, guarda el modelo y reinicia el contador
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            patience_counter = 0
            # Guarda el mejor modelo como checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
        else:
            # Si no mejora, incrementa el contador de paciencia
            patience_counter += 1
            # Si se excede la paciencia, se activa early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Genera el texto del log de la época actual
        log_str = (f"Epoch {epoch+1:03d}: "
                   f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f} | "
                   f"Dif Acc: {train_acc-val_acc:.4f}")

        # Muestra el log en consola
        print(log_str)
        # Guarda el log en el archivo de resultados
        with open(output_path, 'a') as f:
            f.write(log_str + '\n')

    # Muestra la mejor precisión de validación alcanzada
    print(f"Best Validation Accuracy: {best_acc:.4f}")

# Ejecuta la función principal solo si este archivo se ejecuta directamente (no si se importa)
if __name__ == "__main__":
    train_model()