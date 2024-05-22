from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

#cargar modelo
model = keras.models.load_model("keras_model.h5", compile=False)

# Cargar Etiquetas

class_names = open("labels.txt", "r").readlines()


@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        #determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


        #obtener la imagen de la solicitud
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

           # Redimensiona la imagen
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Convierte la imagen en un arreglo numpy
        image_array = np.asarray(image)

        # Normaliza la imagen
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data[0] = normalized_image_array

        # Realiza la predicción
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][2:]
        confidence_score = prediction[0][index]

        return jsonify({'class': class_name, 'confidence': float(confidence_score)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)