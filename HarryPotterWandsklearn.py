from PIL import Image
import numpy as np
import joblib
import os

# === Function to perform spell prediction ===
def predict_spell(img_path, model_path):
    # Open the image from the given path and convert it to grayscale
    img = Image.open(img_path).convert("L")
    
    # Convert the image to a NumPy array and flatten it to a 1D vector (shape: 1 x 784)
    img = np.array(img).reshape(1, -1)
    
    # Load the pre-trained classifier model from disk
    clf = joblib.load(model_path)
    
    # Predict the class (0 = open spell, 1 = close spell) and return the result
    prediction = clf.predict(img)
    return prediction[0]

# === Script entry point (used when run directly) ===
if __name__ == "__main__":
    # Define the image and model file paths
    img_path = "/home/gloworm72/WandProject/lastframe.jpg"
    model_path = "/home/gloworm72/WandProject/new_custom_classifier.pkl"

    # Call the prediction function and print the result
    result = predict_spell(img_path, model_path)
    print(result)
