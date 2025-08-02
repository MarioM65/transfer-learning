import os
import zipfile
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
import matplotlib.pyplot as plt

# ==== Configurações ====
DATASET_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
DATASET_ZIP = "cats_and_dogs_filtered.zip"
DATASET_DIR = "cats_and_dogs_filtered"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = "modelo_transfer.h5"

def baixar_dataset():
    if not os.path.exists(DATASET_ZIP):
        print("🔽 Baixando dataset...")
        response = requests.get(DATASET_URL)
        with open(DATASET_ZIP, "wb") as f:
            f.write(response.content)
    else:
        print("✅ Dataset já existe.")

def extrair_dataset():
    if not os.path.exists(DATASET_DIR):
        print("📦 Extraindo dataset...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        print("✅ Dataset já extraído.")

def carregar_dados():
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'validation')

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_gen, val_gen

def construir_modelo():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def treinar_modelo(model, train_gen, val_gen):
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    return history

def avaliar_modelo(model, val_gen):
    loss, acc = model.evaluate(val_gen)
    print(f"📊 Acurácia de validação: {acc:.2%}")

def mostrar_previsao_aleatoria(model, val_gen):
    img_path = val_gen.filepaths[0]
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    plt.imshow(img)
    plt.title("Classe prevista: " + ("Dog 🐶" if prediction > 0.5 else "Cat 🐱"))
    plt.axis('off')
    plt.show()

def salvar_modelo(model):
    model.save(MODEL_PATH)
    print(f"💾 Modelo salvo como {MODEL_PATH}")

# ==== Execução ====
if __name__ == "__main__":
    baixar_dataset()
    extrair_dataset()
    train_gen, val_gen = carregar_dados()
    model = construir_modelo()
    treinar_modelo(model, train_gen, val_gen)
    avaliar_modelo(model, val_gen)
    salvar_modelo(model)
    mostrar_previsao_aleatoria(model, val_gen)
