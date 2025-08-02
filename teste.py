from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('modelo_transfer.h5')

img_path = 'minha_imagem.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

plt.imshow(image.load_img(img_path))
plt.title("Classe prevista: " + ("Dog 🐶" if prediction > 0.5 else "Cat 🐱"))
plt.axis('off')
plt.show()
