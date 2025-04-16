import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# Page setup
st.set_page_config(page_title="Chest X-ray Disease Detector", layout="centered")

# Disease labels
class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

# Load the model
def load_model():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(1024, 15)
    )
    model.load_state_dict(torch.load("chexnet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Predict with confidence scores
def predict(image, model):
    image_tensor = transform_image(image)
    outputs = model(image_tensor)
    probs = torch.sigmoid(outputs).squeeze().detach().numpy()
    return sorted([(class_names[i], float(probs[i])) for i in range(len(probs))], key=lambda x: x[1], reverse=True)

# UI Elements
st.title("🔬 Chest X-ray Disease Prediction App")
st.subheader("Detect multiple lung diseases using AI (DenseNet121)")
st.markdown("---")

st.markdown("### 📁 Upload a Chest X-ray")
uploaded_file = st.file_uploader("Choose an image (JPG or PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🩻 Uploaded Chest X-ray", use_container_width=True)

    model = load_model()
    with st.spinner("⏳ Running model..."):
        prediction = predict(image, model)

    st.markdown("### 🩺 Prediction Results")
    has_positive = False
    for disease, score in prediction:
        if score > 0.5:
            has_positive = True
            st.success(f"✅ {disease} — **{score:.2%}**")

    if not has_positive:
        st.info("✅ No disease detected with current threshold.")

    st.markdown("### 📈 Prediction Confidence (All Labels)")
    for disease, score in prediction:
        st.write(f"**{disease}**: {score:.2%}")
        st.progress(score)

# Sidebar info
st.sidebar.title("📘 Project Info")
st.sidebar.markdown("""
- **Model:** DenseNet121
- **Dataset:** NIH Chest X-ray14
- **Frameworks:** PyTorch + Streamlit
- **Task:** Multi-label Disease Detection
- **Authors:** Ayush Makade
""")

# Footer
st.markdown("---")
st.caption("Created as part of Final Year Project • © 2025")
