## Brain Tumor Multimodal Classifier App

This repository contains a Streamlit application and supporting code for a multimodal deep learning model that classifies brain tumor presence based on CT and/or MRI scans.

## 🧠 Overview
- Multimodal classifier that supports CT, MRI, or both inputs.
- Dual DenseNet201 architecture (feature-level fusion).
- Trained using the PyTorch deep learning framework.
- Integrated with Weights & Biases for experiment tracking.

## 🚀 Live Demo
You can run the app locally using:
```bash
streamlit run app.py
```
or you can try out the app on streamlit cloud [here](https://ctmri-net.streamlit.app/)

## 📁 Project Structure
```plaintext
brain-tumor-multimodal-app/
├── app.py                 # Streamlit interface for predictions
├── model.py               # Model architecture 
├── utils.py               # Helper functions 
├── requirements.txt       # Dependencies
├── README.md              # This documentation
├── inference_examples/    # Example CT and MRI inputs (optional)
└── notebooks/
    └── training_notebook.ipynb   # Kaggle-style notebook with training pipeline
```

## 🧾 Dataset
The model was trained using the public Kaggle dataset:
📂 [Brain Tumor Multimodal Image CT and MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-multimodal-image-ct-and-mri)

- Format: ImageFolder (`Healthy/`, `Tumour/`)
- CT and MRI stored in separate parent folders
- Samples randomly paired by label category

## 🧠 Model Architecture
- Two DenseNet201 encoders for CT and MRI images
- Global average pooling and feature fusion
- Fully connected classifier for binary prediction
- Softmax output for confidence scoring

## 📦 Model Weights
The pretrained model is hosted on Hugging Face:
📍 [Hugging Face Repo](https://huggingface.co/your-username/brain-tumor-multimodal)

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="lukmanaj/brain-tumor-multimodal",
    filename="multimodal_brain_tumor_model.pth"
)
```

## 📊 Training Performance

| Epoch | Train Loss | Accuracy |
|-------|------------|----------|
| 1     | 0.1552     | 94.82%   |
| 5     | 0.0368     | 98.78%   |

⚠️ The model shows signs of overfitting — further validation or regularization is recommended.

## 🧑‍⚕️ Intended Use
- For educational and research purposes only.
- Not suitable for clinical diagnosis or real-world deployment without further validation.

## 📚 Citation
```
Aliyu, L. (2025). Brain Tumor Classification using Multimodal Deep Learning (ArewaDS Capstone Project).
```

## 🤝 Acknowledgement
- [Masoud Nickparvar (Kaggle dataset)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-multimodal-image-ct-and-mri)

---

Feel free to contribute or open an issue for improvements or questions.
