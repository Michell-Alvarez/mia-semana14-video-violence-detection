# -------------------------------
# STREAMLIT Y COMPONENTES DE UI
# -------------------------------

# Librería principal para crear interfaces web interactivas
import streamlit as st

# Módulo adicional para crear menús laterales personalizados
from streamlit_option_menu import option_menu

# -------------------------------
# SISTEMA DE ARCHIVOS Y UTILIDADES
# -------------------------------

# Librería estándar de Python para manipular rutas y archivos
import os
# Permite la creación de archivos temporales
import tempfile
# Permite ejecutar comandos de terminal desde Python
import subprocess
# Permite copiar archivos o directorios
import shutil

# -------------------------------
# VISIÓN POR COMPUTADOR Y TRANSFORMACIONES
# -------------------------------

# OpenCV para procesamiento de imágenes y lectura de video
import cv2
# Transformaciones para preprocesar imágenes (ToTensor, Normalize, etc.)
from torchvision import transforms

# -------------------------------
# PYTORCH (DEEP LEARNING)
# -------------------------------

# Librería principal para entrenamiento e inferencia con redes neuronales
import torch
# Submódulo de PyTorch para construir modelos de redes neuronales
import torch.nn as nn



# ==== Modelo ====
# Definición de una red neuronal convolucional 3D mejorada
class Improved3DCNN(nn.Module):
    # Constructor del modelo
    def __init__(self):
        # Inicializa la clase base nn.Module
        super(Improved3DCNN, self).__init__()

        # Primera capa convolucional 3D: de 3 canales (RGB) a 16 canales
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        self.dropout1 = nn.Dropout(0.2)  # Dropout para reducir overfitting

        # Pooling para reducir las dimensiones espaciales (alto y ancho)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Segunda capa convolucional: de 16 a 32 canales
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.dropout2 = nn.Dropout(0.2)

        # Tercera capa convolucional: de 32 a 64 canales
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.dropout3 = nn.Dropout(0.2)

        # Calcula automáticamente el tamaño de entrada de la primera capa fully connected
        self.fc_input_size = self._get_fc_input_size()

        # Capa completamente conectada: reduce a 128 neuronas
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout_fc = nn.Dropout(0.2)

        # Capa de salida con 2 neuronas (para clasificación binaria)
        self.fc2 = nn.Linear(128, 2)

    # Método auxiliar para calcular el tamaño que tendrá el tensor tras las convoluciones y poolings
    def _get_fc_input_size(self):
        with torch.no_grad():  # No requiere gradientes
            # Crea un tensor de prueba con forma [batch, channels, time, height, width]
            x = torch.zeros(1, 3, 30, 224, 224)
            # Aplica las tres convoluciones seguidas de pooling y ReLU
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            return x.numel()  # Devuelve el número total de elementos

    # Método de propagación hacia adelante
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout3(x)

        # Aplana el tensor para pasarlo a la capa fully connected
        x = x.view(-1, self.fc_input_size)

        # Capa densa + ReLU + Dropout
        x = self.dropout_fc(nn.functional.relu(self.fc1(x)))

        # Capa de salida sin activación (CrossEntropyLoss lo maneja)
        x = self.fc2(x)
        return x

# ==== Configuración ====
# Selecciona GPU si está disponible, si no usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancia el modelo y lo mueve al dispositivo seleccionado
model = Improved3DCNN().to(device)

# Carga el checkpoint entrenado con mejor desempeño en test (época 6)
checkpoint = torch.load('/home/michell-alvarez/e_tesisI/EF/model_checkpoint_ef_epoch_test_6.pth', map_location=device)

# Restaura los pesos del modelo desde el checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Coloca el modelo en modo evaluación (desactiva dropout y batchnorm)
model.eval()

# Define las transformaciones a aplicar a los frames: tensor y normalización
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte imagen a tensor [C, H, W]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliza al rango [-1, 1]
])

# ==== Funciones ====
# Función para preprocesar un video .mp4 y prepararlo para la predicción
def preprocess_video(video_path):
    # Abre el video con OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []  # Lista para almacenar los frames procesados

    # Lee los frames uno por uno
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Si no hay más frames, salir del bucle

        # Redimensiona cada frame a 224x224 píxeles
        frame = cv2.resize(frame, (224, 224))

        # Convierte el frame de BGR (OpenCV) a RGB (PyTorch)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplica las transformaciones (ToTensor + Normalize)
        frame = transform(frame)

        # Agrega el frame a la lista
        frames.append(frame)

    # Cierra el video
    cap.release()

    # Asegura que haya exactamente 30 frames: rellena con ceros si hay menos
    if len(frames) < 30:
        frames += [torch.zeros(3, 224, 224)] * (30 - len(frames))

    # Si hay más de 30, se queda con los primeros 30
    frames = frames[:30]

    # Convierte la lista de frames en un tensor: [T, C, H, W] → [C, T, H, W]
    frames_tensor = torch.stack(frames).permute(1, 0, 2, 3)

    # Agrega dimensión de batch y mueve al dispositivo
    return frames_tensor.unsqueeze(0).to(device)


# -----------------------------------------------------------
# Función para predecir si un tensor de video contiene violencia
# -----------------------------------------------------------
def predict_tensor(video_tensor):
    # Desactiva el cálculo de gradientes (mejora el rendimiento en inferencia)
    with torch.no_grad():
        # Realiza la predicción usando el modelo
        output = model(video_tensor)
        # Obtiene el índice de la clase con mayor puntuación
        _, predicted = torch.max(output.data, 1)
        # Devuelve la etiqueta correspondiente como string
        return "Violencia" if predicted.item() == 1 else "No violencia"

# -----------------------------------------------------------
# Función para procesar y predecir todos los videos en una carpeta
# -----------------------------------------------------------
def predict_new_videos(directory):
    resultados = []  # Lista para almacenar los resultados (nombre, predicción)

    # Itera sobre todos los archivos del directorio
    for video_file in os.listdir(directory):
        # Solo considera archivos .mp4
        if video_file.endswith('.mp4'):
            # Construye la ruta completa del video
            video_path = os.path.join(directory, video_file)

            # Preprocesa el video para convertirlo en un tensor
            video_tensor = preprocess_video(video_path)

            # Realiza la predicción
            pred = predict_tensor(video_tensor)

            # Almacena el resultado
            resultados.append((video_file, pred))

    # Retorna la lista de predicciones
    return resultados

# -----------------------------------------------------------
# Función para recodificar un video a un formato compatible usando ffmpeg
# -----------------------------------------------------------
def recodificar_video(input_path, output_path):
    # Comando ffmpeg para recodificar el video
    comando = [
        "ffmpeg",
        "-i", input_path,            # archivo de entrada
        "-c:v", "libx264",           # codificación de video en H.264
        "-preset", "fast",           # preset para velocidad de codificación
        "-movflags", "+faststart",   # permite reproducción progresiva
        "-c:a", "aac",               # codificación de audio AAC
        "-b:a", "128k",              # tasa de bits de audio
        output_path                  # archivo de salida
    ]

    # Ejecuta el comando sin mostrar la salida en consola
    subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# -----------------------------------------------------------
# INTERFAZ GRÁFICA CON STREAMLIT
# -----------------------------------------------------------

# Título principal de la aplicación
st.title("Detección de Violencia en Video")

# Menú lateral de navegación
with st.sidebar:
    menu = option_menu(
        "Menú principal",  # Título del menú
        ["Validar videos de Carpeta Local", "Cargar video"],  # Opciones del menú
        icons=["folder-check", "cloud-upload"],  # Iconos para cada opción
        menu_icon="cast",  # Icono del menú principal
        default_index=0,   # Índice por defecto (opción seleccionada inicialmente)

        # Estilos personalizados para el menú lateral
        styles={
            "container": {"padding": "5px", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
        }
    )

# -----------------------------------------------------------
# Opción del menú: Validar todos los videos dentro de una carpeta local
# -----------------------------------------------------------
if menu == "Validar videos de Carpeta Local":
    # Encabezado de la sección
    st.header("📁 Validar carpeta local")

    # Ruta fija donde se encuentran los videos a analizar
    carpeta = '/home/michell-alvarez/e_modelos/EF/Violencia/NewData'

    # Si se presiona el botón de predicción
    if st.button("Iniciar Predicción en Carpeta"):
        # Verifica si la carpeta existe
        if os.path.isdir(carpeta):
            # Llama a la función de predicción por carpeta
            resultados = predict_new_videos(carpeta)

            # Muestra mensaje de éxito
            st.success("Predicción completada.")

            # Muestra los resultados individualmente
            for archivo, prediccion in resultados:
                # Agrega un emoji según la clase predicha
                emoji = "🚨" if prediccion == "Violencia" else "🟢"
                # Muestra el nombre del video y su predicción
                st.write(f"📹 **{archivo}** → {emoji} **{prediccion}**")

            # Línea divisoria
            st.markdown("---")

            # Muestra el total de videos analizados
            st.info(f"✅ Total de videos analizados: **{len(resultados)}**")
        else:
            # Mensaje de error si la carpeta no existe
            st.error(f"No se encontró la carpeta: {carpeta}")

# -----------------------------------------------------------
# Opción del menú: Subir un video individual para análisis
# -----------------------------------------------------------
elif menu == "Cargar video":
    # Encabezado de la sección
    st.header("📤 Cargar un video para análisis")

    # Componente para subir archivos .mp4
    uploaded_file = st.file_uploader("Selecciona un archivo .mp4", type=["mp4"])

    # Verifica si se ha subido un archivo
    if uploaded_file is not None:
        # Crea un archivo temporal para guardar el video subido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            # Copia el contenido del archivo subido al archivo temporal
            shutil.copyfileobj(uploaded_file, tmp)
            tmp.flush()              # Asegura que se escriba todo en disco
            os.fsync(tmp.fileno())   # Sincroniza con el sistema de archivos
            tmp_path = tmp.name      # Obtiene la ruta al archivo temporal

        # Intenta abrir el video con OpenCV para validarlo
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            # Si el video no puede abrirse, muestra un mensaje de error
            st.error("❌ No se pudo abrir el video. El archivo podría estar dañado.")
        else:
            # Obtiene información del video para verificar duración
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = n_frames / fps if fps > 0 else 0
            cap.release()

            # Muestra advertencia si el video es muy corto
            if duration < 1:
                st.warning("⚠️ El video tiene muy poca duración, puede que no se habilite el botón de Play.")

            # Recodifica el video para asegurar compatibilidad en Streamlit
            recod_path = tmp_path.replace(".mp4", "_recode.mp4")
            recodificar_video(tmp_path, recod_path)

            # Muestra el video recodificado en la app
            st.video(recod_path)

            # Si el usuario presiona el botón para predecir el video cargado
            if st.button("Predecir video cargado"):
                # Preprocesa el video y realiza la predicción
                video_tensor = preprocess_video(recod_path)
                pred = predict_tensor(video_tensor)
                emoji = "🚨" if pred == "Violencia" else "🟢"
                # Muestra el resultado
                st.success(f"📹 Video cargado → {emoji} **{pred}**")
