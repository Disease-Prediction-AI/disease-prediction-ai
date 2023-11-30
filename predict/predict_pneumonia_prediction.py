from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from PIL import Image, ImageOps  
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

class PneumoniaModel:
    def __init__(self):
        self.model = None

    def load_pneumonia_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, image_path):
        img_array = self.preprocess_image(image_path)
        pred_result = self.model.predict(img_array)
        return pred_result
    
    def imagerecognise(self,uploadedfile,modelpath,labelpath):
        np.set_printoptions(suppress=True)
        model = load_model(modelpath, compile=False)
        class_names = open(labelpath, "r").readlines()
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(uploadedfile).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        print('prediction value',prediction)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        return(class_name[2:],confidence_score)

if __name__ == "__main__":
    model = PneumoniaModel()
    model.load_pneumonia_model("model/pneumonia_prediction.keras")
    result = model.predict("data/chest_xray/train/PNEUMONIA/person2_bacteria_4.jpeg")
    print(f'result prediction: { result }\n')
    print("...................////")
    model.imagerecognise("data/chest_xray/train/PNEUMONIA/person2_bacteria_4.jpeg","model/pneumonia_prediction.keras","data/labelsPnemonia.txt")
