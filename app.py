# ## Brain Tumor Multimodal Classifier App
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download

# Streamlit interface to load a saved model and predict on uploaded CT and/or MRI images
class MultiModalDenseNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_ct = models.densenet201(weights=None)
        base_mri = models.densenet201(weights=None)
        self.ct_features = base_ct.features
        self.mri_features = base_mri.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_ct, x_mri):
        f_ct = self.ct_features(x_ct)
        f_mri = self.mri_features(x_mri)
        fused = torch.cat((f_ct, f_mri), dim=1)
        return self.classifier(fused)

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalDenseNet(num_classes=2).to(device)
    model_path = hf_hub_download(
        repo_id="lukmanaj/brain-tumor-multimodal",
        filename="multimodal_brain_tumor_model.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🧠 Brain Tumor Multimodal Classifier")
st.markdown("You may upload a CT scan, MRI scan, or both. The model will predict based on available modality.")

ct_file = st.file_uploader("Upload CT Image (optional)", type=["jpg", "png", "jpeg"])
mri_file = st.file_uploader("Upload MRI Image (optional)", type=["jpg", "png", "jpeg"])

if not ct_file and not mri_file:
    st.info("Please upload at least one image (CT or MRI).")

if ct_file or mri_file:
    if ct_file:
        ct_img = Image.open(ct_file).convert("RGB")
        ct_tensor = transform(ct_img).unsqueeze(0).to(device)
    else:
        ct_tensor = torch.zeros(1, 3, 224, 224).to(device)  # dummy input

    if mri_file:
        mri_img = Image.open(mri_file).convert("RGB")
        mri_tensor = transform(mri_img).unsqueeze(0).to(device)
    else:
        mri_tensor = torch.zeros(1, 3, 224, 224).to(device)  # dummy input

    with torch.inference_mode():
        output = model(ct_tensor, mri_tensor)
        prob = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(prob).item()
        label = "Tumour" if pred == 1 else "Healthy"

    st.image([img for img in [ct_file and ct_img, mri_file and mri_img] if img],
             caption=[cap for cap in [ct_file and "CT Scan", mri_file and "MRI Image"] if cap])
    st.success(f"Prediction: {label} (Confidence: {prob[pred]:.2f})")
