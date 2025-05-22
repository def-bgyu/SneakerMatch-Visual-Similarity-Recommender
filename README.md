SneakerMatch is a visual recommendation system that helps users find sneakers that look similar to a reference image. Upload a picture of a sneaker, and the app returns the top visually similar sneakers based on deep feature embeddings using a pre-trained ResNet50 model.

---

<!-- ## 🌐 Live Demo
📍 *[Optional: Add your Hugging Face or Streamlit Space URL once it's live]*  
Example: [https://NidhiS09-sneakermatch.hf.space](https://huggingface.co/spaces/NidhiS09/SneakerMatch) -->

---

##  Features
- Upload a sneaker image and get top 5 visually similar results.
- Uses pre-trained `ResNet50` (ImageNet) for feature extraction.
- Computes cosine similarity between feature embeddings.
- Clean and interactive Streamlit interface.
- Hugging Face Spaces compatible (Docker-ready setup).

---

##  How It Works

1. **Image Upload:** User uploads a sneaker image.
2. **Feature Extraction:** ResNet50 extracts feature vectors from both uploaded and dataset images.
3. **Similarity Calculation:** Cosine similarity is computed against a precomputed dataset.
4. **Results Displayed:** Top 5 similar sneakers are shown with similarity scores.

---

## 🗂️ Project Structure

```
SneakerMatch/
│
├── src/                        # Source code directory
│   ├── streamlit_app.py        # Main Streamlit UI app
│   ├── extract_features.py     # Dataset loading and feature extraction
│   └── recommender.py          # Cosine similarity and recommendation logic
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker config (for Hugging Face Spaces)
├── .gitignore                  # Files and folders to ignore
├── .runtime.txt                # Python version pin for deployment
└── README.md                   # This file
```

---

## 🧪 Dataset

You can use any sneaker image dataset structured like:

Used the **[Sneakers Image Dataset](https://www.kaggle.com/datasets/saadghojaria/sneakers-image-dataset-pinterest)** from Kaggle, which contains images of sneakers and boots.


---

## ⚙️ Setup (Local)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/SneakerMatch.git
cd SneakerMatch

# Create virtual env and install dependencies
pip install -r requirements.txt

# Run locally
streamlit run src/streamlit_app.py
```

---


## 📦 Requirements

All packages are listed in `requirements.txt`. Key dependencies include:
- `tensorflow`
- `keras`
- `numpy`
- `opencv-python-headless`
- `streamlit`
- `scikit-learn`
- `Pillow`
- `tqdm`
- `gdown` (for downloading zipped datasets)

---


## 👩‍💻 Author

**Nidhi Sankhe**  
Master's in Computer Science  
💼 [LinkedIn](https://www.linkedin.com/in/nidhi-sankhe/)  
🧠 [Hugging Face](https://huggingface.co/NidhiS09)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
