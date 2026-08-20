import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import cv2
import time
import pandas as pd
import altair as alt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="wide"
)

# ==========================
# THEME
# ==========================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme = st.sidebar.radio(
    "🎨 Appearance",
    ["Dark", "Light"],
    index=0
)

st.session_state.theme = theme

# ==========================
# CUSTOM CSS
# ==========================

if theme == "Dark":

    st.markdown("""
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }

    /* TEXT */
    h1, h2, h3, h4, h5, h6,
    p, label, span, div {
        color: #ffffff;
    }

    /* METRIC */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363d;
    }

    /* FILE UPLOADER */
    div[data-testid="stFileUploader"] {
        background-color: #161b22;
        border-radius: 10px;
        padding: 10px;
    }

    /* ALERT */
    .stAlert {
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background-color: #ffffff;
        color: #111111;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #f5f7fa;
    }

    /* TEXT */
    h1, h2, h3, h4, h5, h6,
    p, label, span, div {
        color: #111111;
    }

    /* METRIC */
    div[data-testid="stMetric"] {
        background-color: #f5f7fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #d9dee5;
    }

    /* FILE UPLOADER */
    div[data-testid="stFileUploader"] {
        background-color: #f5f7fa;
        border-radius: 10px;
        padding: 10px;
    }

    /* ALERT */
    .stAlert {
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

# ======================================================
# CLASS
# ======================================================

CLASS_NAMES = [
    "Blight",
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy"
]

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, len(CLASS_NAMES))
    )

    model.load_state_dict(
        torch.load(
            "model/best_resnet50.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

cam = GradCAM(
    model=model,
    target_layers=[model.layer4[-1]]
)

# ======================================================
# TRANSFORM
# ======================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ======================================================
# PREDICT
# ======================================================

def predict(image):

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        prob = torch.softmax(output, dim=1)

        conf, pred = torch.max(prob, dim=1)

    return pred.item(), conf.item(), prob.squeeze()

# ======================================================
# GRAD-CAM
# ======================================================

def generate_gradcam(image):

    rgb_img = image.resize((224,224))

    rgb_img = np.array(rgb_img).astype(np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0)

    pred, _, _ = predict(image)

    targets = [ClassifierOutputTarget(pred)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (224,224)
    )

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🌽 Corn Leaf Disease")

st.sidebar.markdown("---")

st.sidebar.success("🤖 Model : ResNet50")

st.sidebar.info("🔥 Explainable AI : Grad-CAM")

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Current Theme : {theme}"
)

uploaded = st.sidebar.file_uploader(
    "Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

# ======================================================
# TITLE
# ======================================================

st.title("🌽 Corn Leaf Disease Detection")

st.caption(
    "Detection of Corn Leaf Disease using ResNet50 and Explainable AI (Grad-CAM)"
)

st.markdown("---")

# ======================================================
# CHECK IMAGE
# ======================================================

if uploaded is None:

    st.info("Upload gambar daun jagung terlebih dahulu.")

    st.stop()

# ======================================================
# PREDICTION
# ======================================================

image = Image.open(uploaded).convert("RGB")

start = time.time()

pred, conf, prob = predict(image)

elapsed = time.time() - start

# ======================================================
# VALIDATION
# ======================================================

CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.05

top2 = torch.topk(prob, 2)

margin = (
    top2.values[0].item()
    - top2.values[1].item()
)

# ======================================================
# CHECK VALID / INVALID IMAGE
# ======================================================

if conf < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:

    prediction = "Unknown"
    is_valid = False

else:

    prediction = CLASS_NAMES[pred]
    is_valid = True

# ======================================================
# GENERATE GRAD-CAM ONLY IF VALID
# ======================================================

gradcam = None

if is_valid:

    try:

        gradcam = generate_gradcam(image)

    except Exception as e:

        st.warning(
            f"Grad-CAM tidak dapat dibuat: {e}"
        )

        gradcam = None

# ======================================================
# STATUS
# ======================================================

if prediction == "Unknown":

    status = "🔴 Invalid Image"

elif conf >= 0.90:

    status = "🟢 High Confidence"

elif conf >= 0.75:

    status = "🟡 Medium Confidence"

else:

    status = "🔴 Low Confidence"

# ======================================================
# DASHBOARD
# ======================================================

col1, col2 = st.columns([1, 1])

# ==========================
# ORIGINAL IMAGE
# ==========================

with col1:

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


# ==========================
# GRAD-CAM
# ==========================

with col2:

    st.subheader("🔥 Grad-CAM Visualization")

    if is_valid and gradcam is not None:

        st.image(
            gradcam,
            use_container_width=True
        )

    else:

        st.warning(
            "⚠️ Invalid Image\n\n"
            "Grad-CAM tidak ditampilkan karena "
            "gambar tidak memenuhi kriteria validasi."
        )

# ======================================================
# RESULT
# ======================================================

st.subheader("📊 Prediction Result")

metric1,metric2,metric3=st.columns(3)

metric1.metric(
    "Prediction",
    prediction
)

metric2.metric(
    "Confidence",
    f"{conf*100:.2f}%"
)

metric3.metric(
    "Inference Time",
    f"{elapsed:.3f} sec"
)

st.progress(float(conf))

if conf>=0.90:

    st.success(status)

elif conf>=0.75:

    st.warning(status)

else:

    st.error(status)

st.markdown("---")

# ======================================================
# PROBABILITY
# ======================================================

st.subheader("📈 Prediction Probability")

values=prob.detach().numpy()*100

df=pd.DataFrame({

    "Disease":CLASS_NAMES,

    "Probability":values

})

chart=alt.Chart(df).mark_bar().encode(

    x=alt.X(
        "Probability:Q",
        title="Probability (%)"
    ),

    y=alt.Y(
        "Disease:N",
        sort="-x"
    ),

    tooltip=[
        "Disease",
        alt.Tooltip(
            "Probability",
            format=".2f"
        )
    ]

).properties(

    height=250

)

st.altair_chart(
    chart,
    use_container_width=True
)

st.markdown("---")

# ======================================================
# TOP PREDICTION
# ======================================================

st.subheader("🏆 Top Prediction")

sorted_idx=np.argsort(values)[::-1]

for idx in sorted_idx:

    c1,c2=st.columns([4,1])

    c1.write(CLASS_NAMES[idx])

    c2.write(f"{values[idx]:.2f}%")

st.markdown("---")

# ======================================================
# MODEL INFORMATION
# ======================================================

st.subheader("🤖 Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.info("""
**Architecture**

- ResNet50
- Transfer Learning
- Fine-Tuning
""")

with info2:
    st.info("""
**Explainable AI**

- Grad-CAM
- CNN Visualization
- Heatmap Analysis
""")

with info3:
    st.info("""
**Dataset**

- Corn Leaf Disease
- 4 Classes
- Image Size 224×224
""")

st.markdown("---")

# ======================================================
# EXPLANATION
# ======================================================

st.subheader("📝 Prediction Explanation")

if prediction == "Healthy":

    st.success(
        """
The uploaded corn leaf is predicted as **Healthy**.

The Grad-CAM visualization highlights the image region
that contributes most to the prediction, showing no
significant disease characteristics.
        """
    )

else:

    st.warning(
        f"""
The uploaded corn leaf is predicted as **{prediction}**.

The highlighted region in the Grad-CAM visualization
indicates the area that most influenced the ResNet50
model during the classification process.
        """
    )

st.markdown("---")

# ======================================================
# CLASS DESCRIPTION
# ======================================================

st.subheader("🌽 Disease Information")

descriptions = {

    "Blight":
    """
    Northern Corn Leaf Blight is characterized by elongated
    gray or brown lesions that develop on corn leaves.
    Severe infection may significantly reduce crop yield.
    """,

    "Common Rust":
    """
    Common Rust is caused by fungal infection and appears
    as reddish-brown pustules scattered across the leaf
    surface.
    """,

    "Gray Leaf Spot":
    """
    Gray Leaf Spot produces rectangular gray lesions that
    gradually expand and may cause premature leaf death.
    """,

    "Healthy":
    """
    The corn leaf shows no visible symptoms of disease
    and appears to be in healthy condition.
    """

}

st.write(descriptions[prediction])

st.markdown("---")

# ======================================================
# FOOTER
# ======================================================

st.caption(
"""
Corn Leaf Disease Detection using ResNet50 and Explainable AI (Grad-CAM)

Developed with Streamlit & PyTorch
"""
)