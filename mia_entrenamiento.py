# Importa funciones del sistema operativo, como manejo de rutas
import os
# Importa OpenCV para leer y procesar videos
import cv2
# Importa PyTorch
import torch
# Importa NumPy para operaciones numéricas
import numpy as np
# Importa pandas para manipulación de datos (aunque no se usa en este fragmento)
import pandas as pd
# Importa módulos de redes neuronales y optimización de PyTorch
from torch import nn, optim
# Importa clases para crear DataLoaders y Datasets personalizados
from torch.utils.data import DataLoader, Dataset
# Importa transformaciones de imágenes de torchvision
from torchvision import transforms
# Importa métricas de evaluación de sklearn
from sklearn.metrics import confusion_matrix, classification_report
# Importa matplotlib para visualización de gráficos
import matplotlib.pyplot as plt
# Importa glob para buscar archivos usando patrones (aunque no se usa en este fragmento)
import glob

# Define una clase personalizada de Dataset para cargar videos
class VideoDataset(Dataset):
    # Constructor: recibe el directorio raíz y una posible transformación
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.videos = []  # Lista para almacenar rutas de los videos
        self.labels = []  # Lista para almacenar etiquetas

        # Recorre dos carpetas: 'Violence' y 'NonViolence'
        for label in ['Violence', 'NonViolence']:
            path = os.path.join(self.directory, label)  # Ruta completa a la carpeta
            for video_file in os.listdir(path):
                # Solo considera archivos de video con extensión .mp4
                if video_file.endswith('.mp4'):
                    self.videos.append(os.path.join(path, video_file))  # Guarda ruta del video
                    self.labels.append(1 if label == 'Violence' else 0)  # Asigna 1 a violencia, 0 a no violencia

    # Devuelve el número total de videos en el dataset
    def __len__(self):
        return len(self.videos)

    # Carga y procesa el video en la posición idx
    def __getitem__(self, idx):
        video_path = self.videos[idx]  # Ruta al video
        label = self.labels[idx]       # Etiqueta correspondiente

        # Abrir el video con OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []  # Lista para almacenar los frames del video

        # Leer todos los frames del video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Redimensionar frame a 224x224
            if self.transform:
                frame = self.transform(frame)  # Aplicar transformación si se definió
            frames.append(frame)
        cap.release()  # Cerrar el archivo de video

        # Asegurar que cada video tenga 30 frames:
        # Si tiene menos, se rellena con tensores de ceros
        if len(frames) < 30:
            frames += [torch.zeros(3, 224, 224)] * (30 - len(frames))
        # Si tiene más, se recortan los primeros 30
        frames = frames[:30]

        # Convertir lista de frames en un solo tensor [T, C, H, W]
        frames = torch.stack(frames)

        # Rearreglar dimensiones a formato [C, T, H, W] para modelos 3D-CNN
        frames = frames.permute(1, 0, 2, 3)
        # Retornar el tensor de video y su etiqueta
        return frames, label  

# Define una clase para un modelo de red neuronal convolucional 3D mejorado
class Improved3DCNN(nn.Module):
    # Método constructor del modelo
    def __init__(self):
        super(Improved3DCNN, self).__init__()

        # Primera capa convolucional 3D: entrada con 3 canales (RGB), salida con 16 canales
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        # Dropout para prevenir overfitting después de la primera convolución
        self.dropout1 = nn.Dropout(0.2)
        # MaxPooling 3D que reduce resolución espacial (no temporal) a la mitad
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Segunda capa convolucional: de 16 a 32 canales
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        # Dropout tras la segunda convolución
        self.dropout2 = nn.Dropout(0.2)

        # Tercera capa convolucional: de 32 a 64 canales
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        # Dropout tras la tercera convolución
        self.dropout3 = nn.Dropout(0.2)

        # Cálculo dinámico del tamaño de entrada para la primera capa completamente conectada
        self.fc_input_size = self._get_fc_input_size()

        # Capa completamente conectada: de fc_input_size a 128 neuronas
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        # Dropout tras la capa fc1 para regularización
        self.dropout_fc = nn.Dropout(0.2)
        # Capa de salida: 2 clases (Violencia y No violencia)
        self.fc2 = nn.Linear(128, 2)

    # Método auxiliar para calcular el tamaño de entrada a la primera capa fully connected
    def _get_fc_input_size(self):
        with torch.no_grad():  # No calcular gradientes aquí
            # Simula una entrada con dimensiones [Batch=1, Channels=3, Time=30, H=224, W=224]
            x = torch.zeros(1, 3, 30, 224, 224)
            # Aplicar convoluciones y pooling como en el forward
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            # Retorna el número de elementos totales (flatten)
            return x.numel()

    # Método forward: define cómo fluye la información en la red
    def forward(self, x):
        # Primera convolución + ReLU + pooling + dropout
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout1(x)
        # Segunda convolución + ReLU + pooling + dropout
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout2(x)
        # Tercera convolución + ReLU + pooling + dropout
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout3(x)
        # Aplanar el tensor para pasar a la capa fully connected
        x = x.view(-1, self.fc_input_size)
        # Fully connected con ReLU + dropout
        x = self.dropout_fc(nn.functional.relu(self.fc1(x)))
        # Capa final (sin activación porque se suele aplicar CrossEntropyLoss luego)
        x = self.fc2(x)
        return x

# Punto de entrada principal del script
if __name__ == "__main__":

    # Ruta donde se guardarán los resultados por época en un archivo de texto
    output_path = '/home/michell-alvarez/e_modelos/EF/archivos_txt/resultados_epoch.txt'

    # Inicializa variables para Early Stopping
    best_accuracy = 0.0  # Mejora máxima registrada de accuracy en test
    epochs_no_improve = 0  # Cuenta de épocas sin mejora
    early_stop_limit = 3  # Límite de épocas sin mejora antes de detener el entrenamiento

    # Usa GPU si está disponible, de lo contrario usa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rutas a los directorios de entrenamiento, prueba y nuevos datos
    train_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/Train'
    test_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/Test'
    new_data_dir = '/home/michell-alvarez/e_modelos/EF/Violencia/NewData'

    # Transformaciones aplicadas a los frames del video (tensor + normalización)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Carga los datasets personalizados con las transformaciones
    train_dataset = VideoDataset(train_dir, transform)
    test_dataset = VideoDataset(test_dir, transform)

    # Crea los dataloaders para entrenamiento y prueba
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Inicializa el modelo mejorado y lo mueve al dispositivo (GPU/CPU)
    model = Improved3DCNN().to(device)

    # Define la función de pérdida (CrossEntropy para clasificación multiclase)
    criterion = nn.CrossEntropyLoss()

    # Inicializa el optimizador Adam con tasa de aprendizaje inicial 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler que reduce la tasa de aprendizaje si la pérdida no mejora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Busca archivos de checkpoint guardados previamente en el sistema
    checkpoint_files = glob.glob('/home/michell-alvarez/e_modelos/EF/model_checkpoint_ef_epoch_*.pth')

    # Si existen checkpoints previos
    if checkpoint_files:
        # Extrae el número de época de cada archivo para identificar el más reciente
        checkpoints = [(int(f.split('_')[-1].split('.')[0]), f) for f in checkpoint_files]
        # Selecciona el checkpoint con mayor número de época
        last_epoch, checkpoint_path = max(checkpoints, key=lambda x: x[0])

        # Carga el checkpoint desde el archivo
        checkpoint = torch.load(checkpoint_path)
        # Restaura el estado del modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        # Restaura el estado del optimizador
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Define la siguiente época desde la que se continuará el entrenamiento
        start_epoch = checkpoint['epoch'] + 1

        # Intenta recuperar métricas previas de entrenamiento si están guardadas
        train_loss = checkpoint.get('train_loss', None)
        train_accuracy = checkpoint.get('train_accuracy', None)

        # Imprime confirmación de carga del checkpoint y estado restaurado
        print(f"Checkpoint loaded from {checkpoint_path}. Starting from epoch {start_epoch}")
        if train_loss is not None:
            print(f"Train Loss loaded: {train_loss}")
        if train_accuracy is not None:
            print(f"Train Accuracy loaded: {train_accuracy}")

    else:
        # Si no se encuentra checkpoint, empieza desde cero
        start_epoch = 0
        print("No checkpoint found. Starting from epoch 0.")
     
# Bucle principal de entrenamiento: desde la última época guardada hasta la 100
for epoch in range(start_epoch, 100):
    # Cambiar el modelo a modo entrenamiento (activa Dropout, etc.)
    model.train()

    # Inicializa variables acumuladoras para pérdida y precisión
    running_loss = 0.0
    total = 0
    correct = 0

    # Itera sobre los lotes del conjunto de entrenamiento
    for inputs, labels in train_loader:
        # Mover datos y etiquetas al dispositivo (GPU o CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Reinicia los gradientes del optimizador
        optimizer.zero_grad()

        # Propagación hacia adelante: predicción del modelo
        outputs = model(inputs)

        # Cálculo de la pérdida entre predicción y etiquetas reales
        loss = criterion(outputs, labels)

        # Propagación hacia atrás: cálculo de gradientes
        loss.backward()

        # Actualiza los pesos del modelo
        optimizer.step()

        # Acumula la pérdida del lote
        running_loss += loss.item()

        # Obtiene la clase con mayor probabilidad por muestra
        _, predicted = torch.max(outputs.data, 1)

        # Cuenta total de muestras en el lote
        total += labels.size(0)

        # Cuenta cuántas predicciones fueron correctas
        correct += (predicted == labels).sum().item()

    # Calcula la pérdida promedio de entrenamiento en toda la época
    train_loss = running_loss / len(train_loader)

    # Calcula la precisión promedio de entrenamiento en toda la época
    train_accuracy = correct / total

    # Imprime resultados de la época
    print(f'Epoch [{epoch+1}/100], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # Guardar checkpoint del modelo en cada época
    if (epoch + 1) % 1 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # Estado del modelo
            'optimizer_state_dict': optimizer.state_dict(),  # Estado del optimizador
            'train_loss': train_loss,  # Pérdida de entrenamiento guardada
            'train_accuracy': train_accuracy,  # Precisión de entrenamiento guardada
        }, f"model_checkpoint_ef_epoch_{epoch + 1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

    # Validación (evaluación en conjunto de prueba)
    model.eval()  # Cambia a modo evaluación (desactiva dropout, etc.)

    # Inicializa variables para pérdida y precisión en test
    test_loss = 0.0
    total = 0
    correct = 0

    # No se calculan gradientes durante la validación
    with torch.no_grad():
        # Itera sobre el conjunto de prueba
        for inputs, labels in test_loader:
            # Mueve los datos al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass para predicciones
            outputs = model(inputs)

            # Cálculo de la pérdida de prueba
            loss = criterion(outputs, labels)

            # Acumula la pérdida
            test_loss += loss.item()

            # Predicción de clase más probable
            _, predicted = torch.max(outputs.data, 1)

            # Cuenta total de muestras en test
            total += labels.size(0)

            # Cuenta cuántas predicciones fueron correctas
            correct += (predicted == labels).sum().item()

        # Calcula la pérdida promedio de validación dividiendo entre la cantidad de lotes
        test_loss /= len(test_loader)

        # Calcula la precisión de validación total
        test_accuracy = correct / total

        # Muestra en consola los resultados de pérdida y precisión en test para la época actual
        print(f'Epoch [{epoch+1}/100], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Guarda un checkpoint del modelo cada época con métricas de validación
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Estado del modelo actual
                'optimizer_state_dict': optimizer.state_dict(),  # Estado del optimizador
                'test_loss': test_loss,  # Pérdida en test para esta época
                'test_accuracy': test_accuracy,  # Precisión en test
            }, f"model_checkpoint_ef_epoch_test_{epoch + 1}.pth")

            # Mensaje de confirmación
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Abrir archivo de resultados en modo de apéndice (no sobrescribe, solo agrega al final)
        with open(output_path, 'a') as f:
            # Escribir los resultados de entrenamiento en el archivo de texto
            f.write(f'Epoch [{epoch+1}/100], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n')
            # Escribir los resultados de validación en el archivo de texto
            f.write(f'Epoch [{epoch+1}/100], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n')

        # Ajusta la tasa de aprendizaje si la pérdida de validación no mejora
        scheduler.step(test_loss)

        # Early Stopping: verifica si hubo mejora significativa en la precisión
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy  # Actualiza mejor precisión obtenida
            epochs_no_improve = 0  # Reinicia el contador de épocas sin mejora
        else:
            # Si la precisión cayó al menos 10 puntos porcentuales, cuenta como no mejora
            if (best_accuracy - test_accuracy) >= 0.1:
                epochs_no_improve += 1
            else:
                # Si la caída fue menor a 0.1, no se considera significativa, no incrementa contador
                epochs_no_improve = 0

        # Si se superó el límite de épocas sin mejora significativa, se detiene el entrenamiento
        if epochs_no_improve >= early_stop_limit:
            print('Early stopping!')
            break  # Sale del bucle principal de entrenamiento
