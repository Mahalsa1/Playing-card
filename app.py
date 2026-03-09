import csv
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import onnxruntime as ort

st.set_page_config(page_title="Playing Card Classification", layout="centered")

st.title("Playing Card Classification")
st.write("Upload a playing card image to predict its class.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cards_model.onnx"
CSV_PATH = BASE_DIR / "cards-image-datasetclassification" / "cards.csv"

DEFAULT_IMAGE_SIZE = (200, 200)
LOW_CONFIDENCE_THRESHOLD = 30.0


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("cards_model.onnx not found in repository")

    session = ort.InferenceSession(str(MODEL_PATH))
    return session


@st.cache_data
def load_class_names():
    if CSV_PATH.exists():
        class_lookup = {}
        with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    idx = int(row["class index"])
                    label = row["labels"]
                    class_lookup[idx] = label
                except:
                    pass
        return [class_lookup[i] for i in sorted(class_lookup)]

    return []


def preprocess_image(image: Image.Image, target_size):
    image = image.convert("RGB")
    img = np.array(image)

    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def detect_playing_card(image: Image.Image):
    frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 25, 90)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8))

    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:10]:
        area=cv2.contourArea(contour)
        if area>5000:
            return True

    return False


session = load_model()
class_names = load_class_names()

input_name = session.get_inputs()[0].name


uploaded_file = st.file_uploader(
    "Choose a card image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_container_width=True)

    try:

        card_detected = detect_playing_card(image)

        processed = preprocess_image(image,DEFAULT_IMAGE_SIZE)

        outputs = session.run(None,{input_name:processed})
        prediction = outputs[0]

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)*100)

        if class_names and class_index < len(class_names):
            predicted_label = class_names[class_index]
        else:
            predicted_label = f"Class {class_index}"

        st.subheader("Prediction Result")

        if (not card_detected) and confidence < LOW_CONFIDENCE_THRESHOLD:
            st.error("No playing card detected in the uploaded image.")
            st.stop()

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning(f"Low confidence ({confidence:.2f}%).")
            st.write("**Predicted Class (best guess):**", predicted_label)
        else:
            st.write("**Predicted Class:**", predicted_label)

        st.write(f"**Confidence:** {confidence:.2f}%")

        probs = prediction[0]
        top_k = min(5,len(probs))

        top_indices = np.argsort(probs)[::-1][:top_k]

        st.write("**Top Predictions:**")
        for idx in top_indices:

            if class_names and idx < len(class_names):
                label = class_names[idx]
            else:
                label = f"Class {idx}"

            st.write(f"- {label}: {probs[idx]*100:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
