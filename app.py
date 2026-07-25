import streamlit as st
import numpy as np
import pickle
from PIL import Image
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

st.set_page_config(page_title="MNIST Live Digit Classifier", page_icon="✏️")


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('artifacts/mnist_ann_model.keras')
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('artifacts/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


model, scaler, encoder = load_artifacts()


def preprocess_like_mnist(image_data):
    """Convert a raw RGBA canvas drawing into an MNIST-style 28x28 array:
    cropped to the digit's bounding box, resized to fit a 20x20 box
    (preserving aspect ratio), centered on a 28x28 canvas, and then
    re-centered by center of mass -- matching how MNIST digits were
    originally prepared."""
    img = Image.fromarray(image_data.astype('uint8'), mode='RGBA').convert('L')
    arr = np.array(img).astype('float32')

    # Remove faint anti-aliasing noise so the bounding box is accurate
    arr[arr < 20] = 0

    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return None  # nothing drawn yet

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]

    # Resize so the longer side becomes 20px, preserving aspect ratio
    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, round(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, round(h * (20.0 / w)))

    cropped_img = Image.fromarray(cropped.astype('uint8')).resize((new_w, new_h), Image.LANCZOS)
    cropped_arr = np.array(cropped_img).astype('float32')

    # Paste centered into a blank 28x28 canvas
    canvas28 = np.zeros((28, 28), dtype='float32')
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas28[top:top + new_h, left:left + new_w] = cropped_arr

    # Fine-tune centering using center of mass (this is what MNIST actually did)
    cy, cx = ndimage.center_of_mass(canvas28)
    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))
    canvas28 = ndimage.shift(canvas28, shift=(shift_y, shift_x), mode='constant', cval=0)

    return canvas28


st.title("✏️ Draw a Digit")
st.write("Draw a digit (0-9) below. The prediction updates automatically after each stroke.")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if st.button("🗑️ Clear Canvas"):
    st.session_state.canvas_key += 1
    st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    # A 280x280 canvas (10x the MNIST 28x28 size) so drawing is comfortable;
    # preprocess_like_mnist() handles cropping/resizing/centering to 28x28.
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

with col2:
    if canvas_result.image_data is not None:
        processed = preprocess_like_mnist(canvas_result.image_data)

        if processed is not None:
            flat = processed.reshape(1, -1)
            scaled = scaler.transform(flat)

            probs = model.predict(scaled, verbose=0)
            pred_label = encoder.categories_[0][np.argmax(probs)]
            confidence = np.max(probs) * 100

            st.markdown(f"## Prediction: **{pred_label}**")
            st.write(f"Confidence: {confidence:.1f}%")
            st.bar_chart(probs[0])

            with st.expander("See what the model actually sees (28x28 input)"):
                st.image(processed.astype('uint8'), width=140)
        else:
            st.info("Start drawing to see a live prediction.")