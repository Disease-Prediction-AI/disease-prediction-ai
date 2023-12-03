

import cv2
import numpy as np
import sys
from keras.models import load_model

class PneumoniaClassifier:
    def __init__(self, model_filepath='model/model_pneumonia.h5'):
        self.model = load_model(model_filepath)

    def preprocess_image(self, img_path, img_size=150):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=-1)
        return np.expand_dims(img_expanded, axis=0)

    def predict(self, img_path):
        preprocessed_img = self.preprocess_image(img_path)
        prediction = self.model.predict(preprocessed_img)
        class_label = "PNEUMONIA" if prediction[0][0] < 0.5 else "NORMAL"
        return class_label


if __name__ == "__main__":
    
    image_path = sys.argv[1]
    # Initialize the classifier
    classifier = PneumoniaClassifier()

    # Make a prediction for the given image
    result = classifier.predict(image_path)

    # Display the result
    print(f"The image is predicted as: {result}")
