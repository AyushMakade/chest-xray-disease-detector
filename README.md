# chest-xray-disease-detector
DeepLearning-powered Streamlit app for predicting chest diseases from X-ray images

# 🩺 Chest X-ray Disease Prediction App

This is an AI-powered web app built with **Streamlit** and **PyTorch** that predicts the presence of **15 chest diseases** from uploaded X-ray images. The model is based on **DenseNet121** and trained using the **NIH ChestXray14** dataset.

---

## 🚀 Try It Out

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_USERNAME/chest-xray-disease-detector/main/st.py)

> ⚠️ Replace the link above with your actual repo path after you deploy!

---

## 📂 Features

- ✅ Detects 15 chest-related diseases
- 🧠 Powered by DenseNet121 deep learning model
- 📊 Multi-label classification with confidence scores
- 🖼️ X-ray image viewer
- ⚡ Built with Streamlit and PyTorch

---

## 🧪 Example Predictions

| Disease              | Confidence |
|----------------------|------------|
| Effusion             | 93.6%      |
| Cardiomegaly         | 85.2%      |
| Pneumothorax         | 14.0%      |
| **No Finding**       | ✖          |

---

## 🛠 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run st.py
