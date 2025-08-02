import zipfile
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

if not os.path.exists("cats_and_dogs"):
    with zipfile.ZipFile("kagglecatsanddogs_5340.zip", 'r') as zip_ref:
        zip_ref.extractall("cats_and_dogs")

src_dir = "cats_and_dogs/PetImages"
base_dir = "dataset"
cat_dir = os.path.join(base_dir, "cats")
dog_dir = os.path.join(base_dir, "dogs")

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for category in ["Cat", "Dog"]:
    src_path = os.path.join(src_dir, category)
    dest_path = os.path.join(base_dir, category.lower() + "s")

    for fname in os.listdir(src_path):
        fsrc = os.path.join(src_path, fname)
        fdst = os.path.join(dest_path, fname)
        try:
            if os.path.getsize(fsrc) > 0:
                shutil.copy(fsrc, fdst)
        except:
            pass  # ignora arquivos corrompidos

IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, validation_data=val_gen, epochs=5)

loss, acc = model.evaluate(val_gen)
print(f"Acurácia de validação: {acc:.2f}")

img_path = val_gen.filepaths[0]
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

plt.imshow(img)
plt.title("Classe prevista: " + ("Dog 🐶" if prediction > 0.5 else "Cat 🐱"))
plt.axis('off')
plt.show()
