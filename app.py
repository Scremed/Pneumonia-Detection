import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import traceback

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Pneumonia X-ray Classifier",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**.")

DEVICE = "cpu"
CLASS_NAMES = ["Normal", "Pneumonia"]
MODEL_PATH = "checkpoints/efficientnet_b3_best.pt"

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model():
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=2
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ----------------------
# File uploader
# ----------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # ----------------------
        # Load & resize image
        # ----------------------
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((300, 300))

        # ----------------------
        # Layout: Image | Results
        # ----------------------
        col_img, col_result = st.columns([1, 2])

        with col_img:
            st.image(
                image,
                caption="Uploaded X-ray",
                width=220
            )

        # ----------------------
        # Preprocess
        # ----------------------
        input_tensor = transforms.ToTensor()(image)
        input_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(input_tensor)

        input_tensor = input_tensor.unsqueeze(0)

        # ----------------------
        # Inference
        # ----------------------
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # ----------------------
        # Prediction results
        # ----------------------
        with col_result:
            st.subheader("🔍 Prediction Results")

            pred_idx = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx]

            if pred_class == "Pneumonia":
                st.error(
                    f"⚠️ **PNEUMONIA DETECTED**\n\n"
                    f"Confidence: **{confidence*100:.1f}%**"
                )
            else:
                st.success(
                    f"✅ **NORMAL**\n\n"
                    f"Confidence: **{confidence*100:.1f}%**"
                )

            # Confidence bar
            st.markdown("### Confidence Level")
            st.progress(int(round(confidence * 100)))

    except Exception as e:
        st.error("❌ Error while processing the image")
        st.code(traceback.format_exc())

st.caption(
"⚠️ Disclaimer: This app is for educational purposes only and must not be used for medical diagnosis."
)   
