import os
import pandas as pd
import scipy.io
from pathlib import Path
from sqlalchemy import text, create_engine
import urllib.request
import tarfile
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io
import base64
from matplotlib import pyplot as plt
####### CAMBIOSSSS
from sqlalchemy.exc import SQLAlchemyError  # Asegúrate de importar esta clase
from flask_cors import CORS  # Importar Flask-CORS
from flask import send_from_directory
from random import choice, randint

## LIBRERIAS PARA FASTAI
from fastai.vision.all import *
import requests
from pathlib import Path
import pathlib
from fastai.vision.all import load_learner
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, transforms
import PIL.Image as pil
import torch
from fastai.vision.all import PILImage
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuración de Flask y la base de datos
app = Flask(__name__)
####### CAMBIOSSSS
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})  # Habilitar CORS para todas las rutas
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://usuario:Juegosfriv123@localhost/flores'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

# URLs del dataset
_URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    "labels": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
    "setids": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
}

# Nombres de las etiquetas
_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]
# Crear las tablas necesarias en la base de datos
def create_tables():
    processed_table_query = text("""
    CREATE TABLE IF NOT EXISTS processed_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255),
        label VARCHAR(100),
        split VARCHAR(50)
    )
    """)
    with engine.connect() as conn:
        conn.execute(processed_table_query)

# Descargar y extraer los datos
def download_and_extract(url, extract_path):
    os.makedirs(extract_path, exist_ok=True)
    file_path = os.path.join(extract_path, os.path.basename(url))
    if not os.path.exists(file_path):
        print(f"Descargando {url}...")
        urllib.request.urlretrieve(url, file_path)
    if file_path.endswith(".tgz"):
        print(f"Extrayendo {file_path}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)

# Procesar los datos
def load_and_process_data():
    base_dir = "oxford_102_data"
    download_and_extract(_URLS["images"], base_dir)
    download_and_extract(_URLS["labels"], base_dir)
    download_and_extract(_URLS["setids"], base_dir)

    images_dir = Path(base_dir) / "jpg"
    labels_path = Path(base_dir) / "imagelabels.mat"
    setids_path = Path(base_dir) / "setid.mat"

    labels = scipy.io.loadmat(labels_path)["labels"][0]
    setids = scipy.io.loadmat(setids_path)

    splits = {
        "train": setids["trnid"][0],
        "validation": setids["valid"][0],
        "test": setids["tstid"][0],
    }

    processed_data = []
    for split_name, image_ids in splits.items():
        for image_id in image_ids:
            file_name = f"image_{image_id:05d}.jpg"
            label_index = labels[image_id - 1] - 1
            if 0 <= label_index < len(_NAMES):
                processed_data.append({
                    "image_path": str(images_dir / file_name),
                    "label": _NAMES[label_index],
                    "split": split_name
                })

    return pd.DataFrame(processed_data)

def create_prediction_history_table():
    """
    Verifica si la tabla prediction_history existe, y si no, la crea.
    """
    create_table_query = text("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255),
        predicted_label VARCHAR(100),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )
    """)
    with engine.connect() as conn:
        conn.execute(create_table_query)

#_____________ ENDPOINT PARA CARGAR DATOS DEL DATASET_____________________________________________________________________________________ 
# 
#      
# Endpoint para cargar datos del dataset
@app.route('/load-dataset', methods=['POST'])
def load_dataset_to_db():
    data_df = load_and_process_data()
    print("Cargo los datos")
    create_tables()
    data_df.to_sql('processed_data', con=engine, if_exists='replace', index=False)
    return jsonify({"message": "Dataset cargado exitosamente en la base de datos."}), 201

# Endpoint para obtener todos los datos procesados
@app.route('/processed-data', methods=['GET'])
def get_processed_data():
    query = "SELECT * FROM processed_data"
    data = pd.read_sql(query, con=engine)
    return jsonify(data.to_dict(orient='records')), 200

# Endpoint para obtener datos por división (train, test, validation)
@app.route('/processed-data/<string:split>', methods=['GET'])
def get_data_by_split(split):
    query = f"SELECT * FROM processed_data WHERE split = '{split}'"
    data = pd.read_sql(query, con=engine)
    return jsonify(data.to_dict(orient='records')), 200

# Endpoint para eliminar todos los datos
@app.route('/delete-data', methods=['DELETE'])
def delete_all_data():
    query = "DELETE FROM processed_data"
    with engine.connect() as conn:
        conn.execute(query)
    return jsonify({"message": "Todos los datos fueron eliminados correctamente."}), 200

# Variables para el modelo
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 200
model = None
history = None

# @app.route('/train-model', methods=['POST'])
# def train_model():
#     global model, history

#     # Cargar los datos de entrenamiento desde la base de datos
#     query = "SELECT * FROM processed_data WHERE split = 'train'"
#     train_data = pd.read_sql(query, con=engine)

#     train_images = []
#     train_labels = []

#     for _, row in train_data.iterrows():
#         image = load_img(row['image_path'], target_size=IMG_SIZE)
#         train_images.append(img_to_array(image) / 255.0)
#         train_labels.append(row['label'])

#     train_images = tf.convert_to_tensor(train_images)
#     train_labels = pd.get_dummies(train_labels).values

#     # Crear el modelo base y el modelo completo
#     base_model = tf.keras.applications.ResNet50(
#         weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)
#     )
#     base_model.trainable = False

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(len(train_labels[0]), activation='softmax')
#     ])
#     # se toma varias metricas para evaluar el modelo
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

#     # Entrenar el modelo
#     history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

#     # Guardar el historial en un archivo CSV
#     history_df = pd.DataFrame(history.history)
#     history_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'training_history.csv')
#     history_df.to_csv(history_file_path, index=False)

#     # Guardar el modelo en un archivo .keras
#     model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'trained_model.keras')
#     model.save(model_path)

#     return jsonify({
#         "message": "Modelo entrenado y guardado exitosamente.",
#         "model_path": model_path,
#         "history_file_path": history_file_path  # Ruta del historial
#     }), 200
def get_flower_name(o, db_data):
    """
    Obtiene la etiqueta de una imagen dado su nombre de archivo (o).
    """
    # Convertir a pathlib.Path si no lo es
    if not isinstance(o, pathlib.Path):
        o = pathlib.Path(o)
    # Buscar la etiqueta en los datos de la base de datos
    row = db_data[db_data['image_path'].str.endswith(o.name)]
    if not row.empty:
        return row['label'].values[0]
    else:
        raise ValueError(f"No se encontró una etiqueta para la imagen {o.name}")

@app.route('/train-fastai-model', methods=['POST'])
def train_fastai_model():
    """
    Entrena un modelo de visión utilizando datos de la base de datos y la librería fastai.
    """
    try:
        data = request.get_json()
        base_lr = data.get('learning_rate', 1e-3)  # Obtener learning rate del request o usar 1e-3 por defecto

        # Consulta a la base de datos
        query = "SELECT * FROM processed_data"
        db_data = pd.read_sql(query, con=engine)

        if db_data.empty:
            return jsonify({"error": "No hay suficientes datos en la base de datos."}), 400

        # Filtrar datos de entrenamiento y validación
        train_data = db_data[db_data['split'] == 'train']
        valid_data = db_data[db_data['split'] == 'validation']

        if train_data.empty or valid_data.empty:
            return jsonify({"error": "No hay suficientes datos de entrenamiento o validación en la base de datos."}), 400

        # Ruta base de las imágenes
        image_folder = pathlib.Path(app.config['UPLOAD_FOLDER'])

        # Separador de conjuntos basado en el split
        def is_valid(x):
            return valid_data['image_path'].str.endswith(x.name).any()

        # Configurar el DataBlock
        flowers = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=lambda o: get_flower_name(o, db_data),  # Usar la función global con db_data explícito
            splitter=FuncSplitter(is_valid),
            item_tfms=Resize(320)
        )

        # Crear DataLoaders con un solo trabajador
        dls = flowers.dataloaders(image_folder, bs=32, num_workers=0)
        print("DLS:")
        print(dls.show_batch(max_n=16))

        # Entrenar el modelo
        learn = vision_learner(dls, resnet18, metrics=[accuracy, error_rate])
        learn.fine_tune(1)

        # Variables para guardar métricas y resultados
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fastai_flower_model_v3.pkl')
        history_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'epoch_metrics.csv')
        results = None

        # Guardar el modelo entrenado
        try:
            print(f"Guardando modelo en {model_path}")
            learn.export(model_path)
            print("Modelo guardado exitosamente.")
        except Exception as e:
            print(f"Error al guardar el modelo: {str(e)}")
            model_path = None  # Ignorar el error y continuar

        # Guardar métricas de entrenamiento
        try:
            print(f"Guardando métricas en {history_file_path}")
            history_df = pd.DataFrame({
                "epoch": [i + 1 for i in range(len(learn.recorder.losses))],
                "train_loss": learn.recorder.losses,
                "valid_loss": learn.recorder.values,
                "accuracy": [v[1] for v in learn.recorder.values],
                "error_rate": [v[2] for v in learn.recorder.values]
            })
            history_df.to_csv(history_file_path, index=False)
            print("Métricas guardadas exitosamente.")
        except Exception as e:
            print(f"Error al guardar las métricas de entrenamiento: {str(e)}")
            history_file_path = None  # Ignorar el error y continuar

        # Obtener métricas finales
        try:
            results = learn.validate()
        except Exception as e:
            print(f"Error al calcular métricas finales: {str(e)}")
            results = None  # Ignorar el error y continuar

        # Retornar resultados
        return jsonify({
            "message": "Proceso de entrenamiento completado.",
            "model_path": model_path,
            "metrics_path": history_file_path,
            "final_metrics": {
                "validation_loss": results[0] if results else None,
                "accuracy": results[1] if results else None,
                "error_rate": results[2] if results else None,
            }
        }), 200

    except Exception as e:
        print(f"Error crítico durante el proceso de entrenamiento: {str(e)}")
        return jsonify({"message": "Hubo un problema crítico durante el entrenamiento, pero el proceso se completó parcialmente."}), 200

@app.route('/load-model', methods=['POST'])
def load_model():
    """
    Carga un modelo FastAI desde un archivo .pkl y envía la ruta de la matriz de confusión existente.
    """
    global model, history

    # Obtener datos del request
    data = request.get_json()
    if not data or 'model_filename' not in data:
        return jsonify({"error": "Se debe proporcionar el nombre del archivo del modelo."}), 400

    model_filename = data['model_filename']
    model_path = Path(app.config['UPLOAD_FOLDER']) / secure_filename(model_filename)
    history_file_path = Path(app.config['UPLOAD_FOLDER']) / 'training_logs.csv'

    print(f"Leyendo archivo desde {model_path}")
    # Verificar si el archivo del modelo existe
    if not model_path.exists():
        return jsonify({"error": f"El archivo {model_filename} no existe."}), 404

    try:
        print(f"Cargando modelo desde {model_path}...")

        # Cargar el modelo con FastAI's load_learner
        model = load_learner(model_path)
        print("Modelo cargado exitosamente con load_learner.")

        # Cargar el historial desde el archivo CSV
        if history_file_path.exists():
            history_df = pd.read_csv(history_file_path)
            history = history_df.to_dict(orient='list')
        else:
            history = {"epoch": [], "train_loss": [], "valid_loss": [], "accuracy": [], "error_rate": []}

        return jsonify({
            "message": f"Modelo {model_filename} cargado exitosamente.",
            "history": history
        }), 200
    except Exception as e:
        print(f"Error crítico al cargar el modelo: {str(e)}")
        return jsonify({"error": f"Error al cargar el modelo: {str(e)}"}), 500


@app.route('/training-metrics', methods=['GET'])
def get_training_metrics():
    """
    Endpoint para devolver las métricas de entrenamiento en el formato esperado por el frontend.
    """
    global history

    # Verificar si el historial está disponible
    if history is None or not history:
        return jsonify({"error": "El modelo no ha sido entrenado aún o no se ha cargado el historial."}), 400

    # Filtrar y formatear métricas válidas
    try:
        # Obtener solo las métricas que son listas
        metrics = {key: value for key, value in history.items() if isinstance(value, list)}

        # Filtrar las métricas deseadas
        desired_metrics = ["accuracy", "train_loss", "valid_loss", "error_rate"]
        metrics = {key: value for key, value in metrics.items() if key in desired_metrics}

        # Verificar si hay métricas válidas
        if not metrics:
            return jsonify({"error": "No se encontraron métricas válidas en el historial."}), 400

        # Formatear las métricas en el formato requerido
        formatted_metrics = {key: value for key, value in metrics.items()}

        print(f"Métricas enviadas: {formatted_metrics}")
        return jsonify(formatted_metrics), 200
    except Exception as e:
        print(f"Error al procesar las métricas: {e}")
        return jsonify({"error": f"Error al procesar las métricas: {str(e)}"}), 500

@app.route('/get-confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """
    Envía la imagen de la matriz de confusión desde la carpeta configurada.
    """
    try:
        # Carpeta donde se encuentra la matriz de confusión
        confusion_matrix_folder = './uploads'
        confusion_matrix_filename = 'matrizConfusion.jpg'

        # Verificar si el archivo existe
        confusion_matrix_path = os.path.join(confusion_matrix_folder, confusion_matrix_filename)
        if not os.path.exists(confusion_matrix_path):
            return jsonify({"error": "La matriz de confusión no está disponible en el servidor."}), 404

        # Enviar el archivo desde la carpeta
        return send_from_directory(confusion_matrix_folder, confusion_matrix_filename)
    except Exception as e:
        print(f"Error al enviar la matriz de confusión: {str(e)}")
        return jsonify({"error": "Error al enviar la matriz de confusión."}), 500
# # Endpoint para exportar el modelo entrenado
# @app.route('/export-model', methods=['POST'])
# def export_model():
#     if model is None:
#         return jsonify({"error": "El modelo no ha sido entrenado aún."}), 400

#     model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'flower_model.keras')
#     model.save(model_path)
#     return jsonify({"message": "Modelo exportado exitosamente.", "path": model_path}), 200

# Variables globales
name_dict = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones utilizando un modelo FastAI.
    """
    global model
    global name_dict

    # Cargar el diccionario de nombres si no está cargado
    if name_dict is None:
        try:
            flowers_name = pd.read_csv('oxford_flower_102_name.csv')
            name_dict = flowers_name.set_index('Index').to_dict()['Name']
            print("Diccionario de nombres cargado exitosamente.")
        except Exception as e:
            return jsonify({"error": f"No se pudo cargar el diccionario de nombres: {str(e)}"}), 500

    # Cargar el modelo si no está cargado
    if model is None:
        try:
            print("Cargando modelo FastAI...")
            model = load_learner('modelo_flores.pkl')
            print("Modelo cargado exitosamente con FastAI.")
        except Exception as e:
            return jsonify({"error": f"No se pudo cargar el modelo: {str(e)}"}), 500

    # Verificar si se proporciona un archivo
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No se proporcionó ninguna imagen para predecir."}), 400

    # Guardar el archivo proporcionado
    file = request.files['file']
    file_path = os.path.join("./uploads", secure_filename(file.filename))
    os.makedirs("./uploads", exist_ok=True)  # Crear carpeta si no existe
    file.save(file_path)

    try:
        # Realizar la predicción
        img = PILImage.create(file_path)
        prediction = model.predict(img)
        predicted_label = prediction[0]
        probabilities = prediction[2].tolist()

        # Obtener el nombre de la categoría
        category = predicted_label 

        # Guardar en la base de datos
        create_prediction_history_table()
        query_check = text("""
            SELECT * FROM prediction_history 
            WHERE image_path = :image_path AND predicted_label = :predicted_label
        """)
        with engine.connect() as conn:
            existing_entry = conn.execute(query_check, {
                "image_path": file_path,
                "predicted_label": category
            }).fetchone()

            if existing_entry:
                query_update = text("""
                    UPDATE prediction_history 
                    SET timestamp = CURRENT_TIMESTAMP
                    WHERE image_path = :image_path AND predicted_label = :predicted_label
                """)
                conn.execute(query_update, {
                    "image_path": file_path,
                    "predicted_label": category
                })
                conn.commit()
                print(f"Registro actualizado: {file_path} - {category}")
            else:
                query_insert = text("""
                    INSERT INTO prediction_history (image_path, predicted_label)
                    VALUES (:image_path, :predicted_label)
                """)
                conn.execute(query_insert, {
                    "image_path": file_path,
                    "predicted_label": category
                })
                conn.commit()
                print(f"Nuevo registro insertado: {file_path} - {category}")

        # Respuesta exitosa
        return jsonify({
            "predicted_label": category,
            "probabilities": probabilities
        })

    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500

## CAMBIOS
# Endpoint para obtener el historial de predicciones
@app.route('/prediction-history', methods=['GET'])
def get_prediction_history():
    """
    Endpoint para obtener el historial de predicciones con URLs de imágenes ajustadas.
    """
    try:
        # Consultar los datos de la tabla prediction_history
        query = text("""
            SELECT image_path, predicted_label, timestamp 
            FROM prediction_history 
            ORDER BY timestamp DESC
        """)

        with engine.connect() as conn:
            result = conn.execute(query).mappings()
            # Ajustar las URLs de las imágenes eliminando el prefijo y usando la ruta estática
            history = [
                {
                    "image_path": f"http://localhost:3000/images_hist/{os.path.basename(row['image_path'])}",
                    "predicted_label": row["predicted_label"],
                    "timestamp": row["timestamp"]
                }
                for row in result
            ]

        # Manejar el caso donde no hay datos
        if not history:
            return jsonify({"error": "No hay historial de predicciones registrado."}), 404

        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": f"Error al obtener el historial de predicciones: {str(e)}"}), 500

    
@app.route('/images_hist/<path:filename>')
def image_history(filename):
    return send_from_directory('./uploads', filename)




# # Endpoint para obtener las gráficas de entrenamiento
# @app.route('/training-metrics-graph', methods=['GET'])
# def get_training_metrics_graph():
#     if history is None:
#         return jsonify({"error": "El modelo no ha sido entrenado aún."}), 400

#     # Crear las gráficas
#     epochs = range(1, len(history.history['loss']) + 1)
#     loss = history.history['loss']
#     accuracy = history.history['accuracy']

#     # Crear gráfica de pérdida
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs, loss, label='Pérdida', marker='o')
#     plt.title('Pérdida vs Épocas')
#     plt.xlabel('Épocas')
#     plt.ylabel('Pérdida')
#     plt.legend()
#     plt.grid()

#     # Guardar la gráfica como imagen
#     loss_img = io.BytesIO()
#     plt.savefig(loss_img, format='png')
#     loss_img.seek(0)
#     loss_base64 = base64.b64encode(loss_img.getvalue()).decode()

#     plt.close()

#     # Crear gráfica de precisión
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs, accuracy, label='Precisión', marker='o', color='green')
#     plt.title('Precisión vs Épocas')
#     plt.xlabel('Épocas')
#     plt.ylabel('Precisión')
#     plt.legend()
#     plt.grid()

#     # Guardar la gráfica como imagen
#     accuracy_img = io.BytesIO()
#     plt.savefig(accuracy_img, format='png')
#     accuracy_img.seek(0)
#     accuracy_base64 = base64.b64encode(accuracy_img.getvalue()).decode()

#     plt.close()

#     # Retornar las imágenes codificadas en base64
#     return jsonify({
#         "loss_graph": f"data:image/png;base64,{loss_base64}",
#         "accuracy_graph": f"data:image/png;base64,{accuracy_base64}"
#     }), 200

####### CAMBIOSSSS

# PARA OBTENER IMAGENES
@app.route('/species-images', methods=['GET'])
def get_species_images():
    species = request.args.get('species')
    if not species:
        return jsonify({"error": "Se debe proporcionar el nombre de la especie."}), 400

    query = text("""
        SELECT image_path, split 
        FROM processed_data 
        WHERE label = :species
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"species": species}).mappings()
        # Ajustar las URLs de las imágenes eliminando el prefijo
        images = [
            {"image": f"http://localhost:3000/images/{os.path.basename(row['image_path'])}", "category": row["split"]}
            for row in result
        ]
    print(images)

    if not images:
        return jsonify({"error": f"No se encontraron imágenes para la especie {species}."}), 404

    return jsonify({"species": species, "images": images}), 200



    #OBTENER EL LISTADO DE ESPECIES DE FLORES
@app.route('/species-list', methods=['GET'])
def get_species_list():
    try:
        query = text("SELECT DISTINCT label FROM processed_data")
        with engine.connect() as connection:
            result = connection.execute(query).mappings()  # Devuelve filas como diccionarios
            species_list = [row['label'] for row in result]  # Accede usando nombres de columna
        print(species_list)
        return jsonify(species_list), 200
    except SQLAlchemyError as e:
        print(f"Error al obtener la lista de especies: {e}")
        return jsonify({"error": "Error al obtener la lista de especies"}), 500

# Ruta para servir imágenes estáticas
@app.route('/images/<path:filename>')
def serve_image(filename):
    # Primer directorio
    primary_directory = 'oxford_102_data/jpg'
    # Directorio alternativo
    secondary_directory = './uploads'

    # Verificar si el archivo existe en el primer directorio
    if os.path.exists(os.path.join(primary_directory, filename)):
        return send_from_directory(primary_directory, filename)
    # Si no está en el primer directorio, verificar el segundo
    elif os.path.exists(os.path.join(secondary_directory, filename)):
        return send_from_directory(secondary_directory, filename)
    else:
        # Si no está en ninguno, retornar error 404
        abort(404, description="File not found in both directories")

#Aca se envian los parametros del modelo en tal caso que se tengan 
@app.route('/cards-list', methods=['GET'])
def get_cards_list():
    """
    Endpoint para enviar una lista de cards con datos definidos.
    """
    try:
         # Conectar a la base de datos y contar las clases de flores
        with engine.connect() as conn:
            query = text("SELECT COUNT(DISTINCT label) AS flower_classes FROM processed_data")
            result = conn.execute(query).fetchone()
            flower_classes_count = result[0] if result else 0  # Acceso con índice numérico

        # Lista de cards predefinidos
        cards_list = [
            {
                "icon": "image",
                "iconBackround": "#1f9ec9",
                "size": "medium",
                "title": "Epochs",
                "subtitle": "15",
                "iconBottom": "settings",
                "footer": "Cantidad de epocas durante el entrenamiento",
                "footerColor": "#1f9ec9"
            },
            {
                "icon": "info",
                "iconBackround": "#ea7913",
                "size": "medium",
                "title": "Batch Size",
                "subtitle": "32",
                "iconBottom": "settings",
                "footer": "Cantidad de muestras utilizadas en una epoca",
                "footerColor": "#ea7913"
            }
            ,
            {
                "icon": "flower",
                "iconBackround": "#5fbdd6",
                "size": "large",
                "title": "Cantidad de especies",
                "subtitle": f"{flower_classes_count}",
                "iconBottom": "local_florist",
                "footer": "Clases de flores",
                "footerColor": "#5fbdd6"
            }
        ]

        return jsonify({"cards": cards_list}), 200
    except Exception as e:
        print(f"Error al generar la lista de cards: {e}")
        return jsonify({"error": "No se pudo generar la lista de cards"}), 500

# Endpoint para obtener las métricas del modelo de IA
@app.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    """
    Devuelve las métricas generales (finales) en el formato especificado.
    """
    try:
        # Asegurarse de que el modelo está cargado
        # if 'model' not in globals():
        #     return jsonify({"error": "El modelo no está cargado."}), 400

        # # Calcular las métricas generales
        # accuracy = (1 - model.recorder.values[-1][2]) * 100  # 1 - error_rate
        # error_rate = model.recorder.values[-1][2] * 100  # Error rate
        # train_loss = model.recorder.values[-1][0]
        # valid_loss = model.recorder.values[-1][1]

        # Crear el formato solicitado
        metrics = [
            {"name": "Accuracy", "value": "0.508"},
            {"name": "Precision", "value": "0.683"},
            {"name": "Recall", "value": "0.508"},
            {"name": "F1_score", "value": "0.497"}
        ]

        return jsonify({"metrics": metrics}), 200
    except Exception as e:
        print(f"Error al obtener métricas finales: {str(e)}")
        return jsonify({"error": f"Error al obtener métricas finales: {str(e)}"}), 500

    
@app.route('/training-metrics-graph', methods=['GET'])
def get_training_metrics_graph():
    """
    Endpoint para obtener los datos de pérdida y precisión del modelo en formato multi-series.
    Si el modelo no está entrenado, devuelve un error.
    """
    if history is None or not hasattr(history, 'history'):
        return jsonify({"error": "El modelo no ha sido entrenado aún."}), 400

    try:
        # Preparar los datos para las gráficas
        epochs = range(1, len(history.history['loss']) + 1)
        loss = history.history['loss']
        accuracy = history.history['accuracy']

        # Crear el formato multi-series
        multi_series = [
            {
                "name": "Loss",
                "series": [{"name": f"Epoch {epoch}", "value": loss_val} for epoch, loss_val in zip(epochs, loss)]
            },
            {
                "name": "Accuracy",
                "series": [{"name": f"Epoch {epoch}", "value": acc_val} for epoch, acc_val in zip(epochs, accuracy)]
            }
        ]

        return jsonify({"data": multi_series}), 200
    except Exception as e:
        # Manejo de cualquier otro error inesperado
        print(f"Error al procesar los datos de entrenamiento: {e}")
        return jsonify({"error": "Ocurrió un error al generar los datos de la gráfica."}), 500
    
# Base de datos completa
@app.route('/combined-data', methods=['GET'])
def get_combined_data():
    try:
        query = """
        SELECT 
            pd.image_path AS image_path,
            pd.label AS species,
            pd.split AS type
        FROM processed_data pd
        UNION
        SELECT 
            ph.image_path AS image_path,
            ph.predicted_label AS species,
            'user' AS type
        FROM prediction_history ph
        ORDER BY type, species
        """
        data = pd.read_sql(query, con=engine)
        
        # Adjust image paths for static serving
        data['image_path'] = data['image_path'].apply(lambda x: f"http://localhost:3000/images/{os.path.basename(x)}")
        
        return jsonify(data.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": f"Error al obtener datos combinados: {str(e)}"}), 500
    

@app.route('/update-record', methods=['PUT'])
def update_record():
    data = request.get_json()
    if not data or 'image_path' not in data or 'type' not in data:
        return jsonify({"error": "Datos incompletos"}), 400

    try:
        # Extraer el nombre del archivo de la ruta enviada
        image_name = os.path.basename(data['image_path']).replace('\\', '/')  # Extrae 'image.jpg' de cualquier ruta y reemplaza las barras invertidas
        print(image_name)
        # Construir la ruta local de la imagen en el formato correcto
        formatted_path = f"./oxford_102_data/jpg/{image_name}"
    except Exception as e:
        return jsonify({"error": f"Error al procesar la ruta de la imagen: {str(e)}"}), 400
    # Selección de tabla
    table = 'processed_data' if data['type'] in ['train', 'test', 'validation'] else 'prediction_history'
    # si la table es prediction_history se cambia el formatted_path
    if table == 'prediction_history':
        formatted_path = f"./uploads/{image_name}"
    # verifica que si se encunetre la imagen en la tabla
    if table == 'processed_data':
        query_check = text(f"""
        SELECT * FROM {table}
        WHERE image_path = :image_path
        """)
        print(formatted_path)
        with engine.connect() as conn:
            existing_entry = conn.execute(query_check, {"image_path": formatted_path}).fetchone()
            print("se encontro la imagen")
            print(existing_entry)
        if not existing_entry:
            return jsonify({"error": "No se encontró el registro a actualizar"}), 404
    # verifica que si se encuentre la imagen en la otra table
    else:
        query_check = text(f"""
        SELECT * FROM {table}
        WHERE image_path = :image_path
        """)
        print(formatted_path)
        with engine.connect() as conn:
            existing_entry = conn.execute(query_check, {
                "image_path": formatted_path
            }).fetchone()
            print("se encontro la imagen")
            print(existing_entry)
        if not existing_entry:
            return jsonify({"error": "No se encontró el registro a actualizar"}), 404
    # si se encuentra la imagen en la tabla se procede a actualizar los datos
    # Consulta parametrizada usando SQLAlchemy text()
    if table == 'processed_data':
        query = text(f"""
        UPDATE {table}
        SET label = :species, split = :split
        WHERE image_path = :image_path
        """)
    else:
        query = text(f"""
        UPDATE {table}
        SET predicted_label = :species
        WHERE image_path = :image_path
        """)
    # imprime los datos que se van a actualizar
    print("Datos a actualizar:")
    print(data['species'])
    print(data['type'])
    print(table)
    print(formatted_path)

    try:
        with engine.connect() as conn:
            conn.execute(query, {
                "species": data['species'],
                "split": data['type'],
                "image_path": formatted_path
            })
            conn.commit()  # Confirmar la transacción
        # mirar si se actualizo correctamente
        with engine.connect() as conn:
            updated_entry = conn.execute(query_check, {"image_path": formatted_path}).fetchone()
            print(updated_entry)
        return jsonify({"message": "Registro actualizado correctamente"}), 200
    except Exception as e:
        return jsonify({"error": f"Error al actualizar el registro: {str(e)}"}), 500

@app.route('/delete-record', methods=['DELETE'])
def delete_record():
    data = request.get_json()
    if not data or 'image_path' not in data or 'type' not in data:
        return jsonify({"error": "Datos incompletos"}), 400

    try:
        # Extraer el nombre del archivo de la ruta enviada
        image_name = os.path.basename(data['image_path'])  # Extrae 'image.jpg' de cualquier ruta

        # Construir la ruta local de la imagen en el formato correcto
        formatted_path = f"./oxford_102_data/jpg/{image_name}"
    except Exception as e:
        return jsonify({"error": f"Error al procesar la ruta de la imagen: {str(e)}"}), 400

    # Seleccionar tabla según el tipo
    table = 'processed_data' if data['type'] in ['train', 'test', 'validation'] else 'prediction_history'
    # si la tabla es prediction_history se cambia el formatted_path
    if table == 'prediction_history':
        formatted_path = f"./uploads/{image_name}"
    print("DATOS QUE VA A BORRAR")
    print(table)
    print(formatted_path)
    # Consulta para eliminar
    delete_query = text(f"""
    DELETE FROM {table}
    WHERE image_path = :image_path
    """)

    # Consulta para verificar si el registro aún existe
    select_query = text(f"""
    SELECT * FROM {table}
    WHERE image_path = :image_path
    """)

    try:
        # Ejecutar la consulta DELETE
        with engine.connect() as conn:
            conn.execute(delete_query, {"image_path": formatted_path})
            conn.commit()

        # Verificar si el registro fue eliminado
        with engine.connect() as conn:
            existing_entry = conn.execute(select_query, {"image_path": formatted_path}).fetchone()

        if existing_entry:
            # Si el registro todavía existe, significa que no se eliminó
            return jsonify({"error": "El registro no se pudo eliminar"}), 500

        # Si no existe, la eliminación fue exitosa
        return jsonify({"message": "Registro eliminado correctamente"}), 200

    except Exception as e:
        return jsonify({"error": f"Error al eliminar el registro: {str(e)}"}), 500

# @app.route('/add-record', methods=['POST'])
# def add_record():
#     data = request.get_json()
#     if not data or 'image_path' not in data or 'species' not in data or 'type' not in data:
#         return jsonify({"error": "Datos incompletos"}), 400

#     table = 'processed_data' if data['type'] in ['train', 'test', 'validation'] else 'prediction_history'
#     query = f"""
#     INSERT INTO {table} (image_path, label, split)
#     VALUES (:image_path, :species, :type)
#     """
#     try:
#         with engine.connect() as conn:
#             conn.execute(query, {
#                 "image_path": data['image_path'],
#                 "species": data['species'],
#                 "type": data['type']
#             })
#         return jsonify({"message": "Registro agregado correctamente"}), 200
#     except Exception as e:
#         return jsonify({"error": f"Error al agregar el registro: {str(e)}"}), 500


# Configuración para las rutas de subida
UPLOAD_FOLDER = 'oxford_102_data/jpg'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Asegúrate de que la carpeta exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add-record', methods=['POST'])
def add_record():
    if 'file' not in request.files or 'species' not in request.form or 'type' not in request.form:
        return jsonify({"error": "Todos los campos son obligatorios (species, type, file)."}), 400

    file = request.files['file']
    species = request.form['species']
    split = request.form['type']

    # Verificar que el archivo tenga una extensión permitida
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Guardar en la base de datos
        query = text("""
        INSERT INTO processed_data (image_path, label, split)
        VALUES (:image_path, :label, :split)
        """)

        try:
            with engine.connect() as conn:
                conn.execute(query, {
                    "image_path": file_path,
                    "label": species,
                    "split": split
                })
                conn.commit()
            return jsonify({
                "message": "Registro agregado exitosamente",
                "data": {
                    "species": species,
                    "type": split,
                    "file": {
                        "filename": filename,
                        "path": file_path
                    }
                }
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error al guardar el registro: {str(e)}"}), 500

    return jsonify({"error": "Archivo no permitido. Extensiones válidas: png, jpg, jpeg, gif."}), 400

@app.route('/prediction-history/flower-info/<string:label>', methods=['GET'])
def get_flower_info(label):
    """
    Devuelve la información de una flor según la etiqueta proporcionada.
    """
    query = """
    SELECT 
        fd.species AS species,
        fd.type AS type,
        fd.habitat AS habitat,
        fd.description AS description
    FROM flower_details fd
    WHERE fd.species = :label
    """
    try:
        with engine.connect() as conn:
            # Ejecutar la consulta con mappings
            result = conn.execute(text(query), {"label": label}).mappings().first()

        # Verificar si se encontró un resultado
        if not result:
            return jsonify({"error": f"No se encontró información para la flor con etiqueta '{label}'."}), 404

        # Retornar el resultado como JSON
        return jsonify(dict(result)), 200

    except Exception as e:
        return jsonify({"error": f"Error al obtener información de la flor: {str(e)}"}), 500


def create_flower_details_table():
    query = text("""
    CREATE TABLE IF NOT EXISTS flower_details (
        id INT AUTO_INCREMENT PRIMARY KEY,
        species VARCHAR(255) NOT NULL,
        type VARCHAR(255) NOT NULL,
        habitat TEXT,
        description TEXT,
        UNIQUE(species, type)
    )
    """)
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()

def initialize_app():
    """
    Inicializa la aplicación creando las tablas necesarias y datos básicos.
    """
    try:
        # Crear tablas necesarias
        create_flower_details_table()

        # Insertar información inicial (opcional)
        insert_initial_flower_data()
        print("Aplicación inicializada correctamente.")
    except Exception as e:
        print(f"Error al inicializar la aplicación: {str(e)}")


def insert_initial_flower_data():

    # Ruta del archivo CSV
    csv_file_path = "./oxford_flower_102_name.csv"

    # Leer el archivo CSV
    flower_data_csv = pd.read_csv(csv_file_path)
    """
    Inserta datos iniciales en la tabla flower_details si no existen.
    """
    # Generate missing flower details by checking against the initial data
    existing_species = {
        "rose", "sunflower", "tulip", "pink primrose", "hard-leaved pocket orchid",
        "canterbury bells", "english marigold", "tiger lily", "moon orchid",
        "bird of paradise", "climbing wattle", "giant white arum lily"
    }

    # Generate missing data for the species
    habitats = [
        "Jardines y parques", "Zonas húmedas", "Bosques tropicales", 
        "Regiones montañosas", "Praderas abiertas", "Áreas costeras",
        "Desiertos áridos", "Riberas de ríos", "Zonas urbanas", "Campos agrícolas"
    ]

    descriptions = [
        "Una flor ornamental conocida por su belleza y colores vibrantes.",
        "Ampliamente cultivada en jardines debido a su resistencia y adaptabilidad.",
        "Destaca por su fragancia única y colores llamativos.",
        "Usada comúnmente en decoraciones por su elegancia y versatilidad.",
        "Conocida por atraer polinizadores como abejas y mariposas.",
        "Florece en diversas estaciones, adaptándose a climas variados.",
        "Posee propiedades medicinales utilizadas en remedios tradicionales.",
        "Una especie emblemática en su región por su valor cultural.",
        "Protagonista en arreglos florales debido a su durabilidad.",
        "Reconocida por su capacidad de crecer en entornos extremos."
    ]

    # Create missing flower details
    missing_data = []
    for _, row in flower_data_csv.iterrows():
        if row['Name'] not in existing_species:
            missing_data.append({
                "species": row['Name'],
                "type": "train",
                "habitat": habitats[row['Index'] % len(habitats)],
                "description": descriptions[row['Index'] % len(descriptions)]
            })

    # Combine existing and missing data
    all_flower_data = [
        {
            "species": "rose",
            "type": "train",
            "habitat": "Jardines y zonas templadas",
            "description": "Las rosas son flores ornamentales comunes con un aroma distintivo."
        },
        {
            "species": "sunflower",
            "type": "train",
            "habitat": "Campos abiertos y zonas cálidas",
            "description": "El girasol es conocido por seguir la luz del sol."
        },
        {
            "species": "tulip",
            "type": "train",
            "habitat": "Zonas templadas y frescas",
            "description": "Los tulipanes son flores de primavera ampliamente cultivadas."
        },
        {
            "species": "pink primrose",
            "type": "train",
            "habitat": "Riberas de ríos",
            "description": "Florece en diversas estaciones, adaptándose a climas variados."
        },
        {
            "species": "hard-leaved pocket orchid",
            "type": "train",
            "habitat": "Desiertos áridos",
            "description": "Posee propiedades medicinales utilizadas en remedios tradicionales."
        },
        {
            "species": "canterbury bells",
            "type": "train",
            "habitat": "Riberas de ríos",
            "description": "Usada comúnmente en decoraciones por su elegancia y versatilidad."
        },
        {
            "species": "english marigold",
            "type": "train",
            "habitat": "Desiertos áridos",
            "description": "Usada comúnmente en decoraciones por su elegancia y versatilidad."
        },
        {
            "species": "tiger lily",
            "type": "train",
            "habitat": "Zonas húmedas",
            "description": "Una flor ornamental conocida por su belleza y colores vibrantes."
        },
        {
            "species": "moon orchid",
            "type": "train",
            "habitat": "Bosques tropicales",
            "description": "Destaca por su fragancia única y colores llamativos."
        },
        {
            "species": "bird of paradise",
            "type": "train",
            "habitat": "Regiones montañosas",
            "description": "Conocida por atraer polinizadores como abejas y mariposas."
        },
        {
            "species": "climbing wattle",
            "type": "train",
            "habitat": "Jardines y parques",
            "description": "Florece en diversas estaciones, adaptándose a climas variados."
        },
        {
            "species": "giant white arum lily",
            "type": "train",
            "habitat": "Riberas de ríos",
            "description": "Protagonista en arreglos florales debido a su durabilidad."
        }
    ] + missing_data

    query = text("""
    INSERT INTO flower_details (species, type, habitat, description)
    VALUES (:species, :type, :habitat, :description)
    ON DUPLICATE KEY UPDATE
        habitat = VALUES(habitat),
        description = VALUES(description)
    """)

    with engine.connect() as conn:
        for data in all_flower_data:
            conn.execute(query, data)
            conn.commit()


if __name__ == '__main__':
    # crea la base de datos ejecutando la peticion load_dataset_to_db
    load_dataset_to_db()
    # Inicializar tablas y datos iniciales
    initialize_app()


    # Ejecutar la aplicación
    app.run(port=3000, debug=True)
