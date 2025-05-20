import numpy as np
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Load model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Load saved features and filenames
FEATURES_PATH = 'features/features.npy'
FILENAMES_PATH = 'features/filenames.npy'
features = np.load(FEATURES_PATH)
filenames = np.load(FILENAMES_PATH)

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def get_image_embedding(img_path):
    img_tensor = preprocess_img(img_path)
    embedding = model.predict(img_tensor, verbose=0)[0]
    return embedding

def recommend(query_img_path, top_k=5):
    query_embedding = get_image_embedding(query_img_path).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, features)[0]

    # Load filenames
    filenames_arr = np.load(FILENAMES_PATH)

    # Exclude the image that's an exact match to the uploaded one
    similarity_list = []
    for i, fname in enumerate(filenames_arr):
        if os.path.abspath(fname) != os.path.abspath(query_img_path):  # avoid self-match
            similarity_list.append((fname, similarities[i]))

    # Sort and return top_k
    similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
    return similarity_list[:top_k]

# Example usage
if __name__ == "__main__":
    query_path = 'data/sneakers/0ed14658bcd730081c79a5224bfb4100.jpg'  
    results = recommend(query_path, top_k=5)
    for path, score in results:
        print(f"{path} - Similarity: {score:.4f}")
