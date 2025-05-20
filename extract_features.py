import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Directory paths
DATA_DIR = 'data'
FEATURE_DIR = 'features'

os.makedirs(FEATURE_DIR, exist_ok=True)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def extract_features():
    features = []
    filenames = []

    # Recursively walk through subfolders
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, fname)
                try:
                    img_tensor = preprocess_img(path)
                    feature = model.predict(img_tensor, verbose=0)[0]
                    features.append(feature)
                    filenames.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

    features = np.array(features)
    filenames = [f.replace("\\", "/") for f in filenames]
    filenames = np.array(filenames)
    
    np.save(os.path.join(FEATURE_DIR, 'features.npy'), features)
    np.save(os.path.join(FEATURE_DIR, 'filenames.npy'), filenames)

    print(f"Saved { len(features)} features from images in {DATA_DIR}/")

if __name__ == "__main__":
    extract_features()
