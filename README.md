# MNIST Digit Classifier — ANN + Live Drawing App

A complete, end-to-end deep learning project: an Artificial Neural Network (ANN) trained on the MNIST handwritten digit dataset, wrapped in a Streamlit app where you draw a digit on a live canvas and get an instant prediction.

**Live demo:** [mnistnumbers-4rr2rdnbmjjxywoqzfhdrv.streamlit.app](https://mnistnumbers-4rr2rdnbmjjxywoqzfhdrv.streamlit.app)

![Draw a digit demo](artifacts/demo_screenshot.png)
<!-- Optional: drop a screenshot of the app in artifacts/ or a new assets/ folder and update this path -->

---

## What's in this repo

| File | Purpose |
|---|---|
| `ProjectMNIST.ipynb` | Full training notebook — data loading, preprocessing, model building, evaluation, regression demo, and hyperparameter tuning |
| `app.py` | Streamlit app with a live drawing canvas that predicts digits in real time |
| `requirements.txt` | Python dependencies for both the notebook and the app |
| `artifacts/mnist_ann_model.keras` | Trained ANN model |
| `artifacts/scaler.pkl` | Fitted `MinMaxScaler` used to scale pixel values |
| `artifacts/encoder.pkl` | Fitted `OneHotEncoder` used for digit labels |
| `LICENSE` | MIT License |

---

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits — 60,000 training images and 10,000 test images, each a 28×28 grayscale image labeled 0–9. Loaded directly via `tf.keras.datasets.mnist.load_data()` (no manual download needed).

---

## Project Pipeline (Notebook)

1. **Getting Started** — load MNIST, explore shapes, visualize sample digits
2. **Preprocessing & Cleaning** — check for missing values and duplicates, check class balance, flatten each 28×28 image to a 784-length vector
3. **Encoding, Scaling & Split** — `MinMaxScaler` on pixel values (0–1), `OneHotEncoder` on labels, stratified train/validation split
4. **Model Building** — a Dense ANN (see architecture below), compiled with Adam and categorical cross-entropy
5. **Prediction & Evaluation** — test accuracy, confusion matrix, classification report, visualized predictions
6. **Save Artifacts** — model, scaler, and encoder saved to disk for reuse by the app
7. **Streamlit App** — `app.py`, built with a live drawing canvas
8. **Cloud Deployment** — deployed via Streamlit Community Cloud
9. **Regression with ANN** — a side demo applying the same ANN framework to a regression task (California Housing dataset), since MNIST has no continuous target
10–13. **Hyperparameter Tuning** — progressively deeper KerasTuner searches: general tuning → node counts → number of hidden layers → full combined search (layers + nodes + dropout + optimizer)

### Model architecture (final ANN)

| Layer | Units | Activation |
|---|---|---|
| Input | 784 | — |
| Dense | 256 | ReLU |
| Dropout | 0.2 | — |
| Dense | 128 | ReLU |
| Dropout | 0.2 | — |
| Dense | 64 | ReLU |
| Dense (output) | 10 | Softmax |

- **Optimizer:** Adam
- **Loss:** categorical cross-entropy
- **Batch size:** 128
- **Epochs:** up to 50, with early stopping (patience 5 on validation loss)

---

## The Streamlit App

`app.py` renders a 280×280 freehand drawing canvas ([`streamlit-drawable-canvas`](https://github.com/andfanilo/streamlit-drawable-canvas)). Every stroke triggers a rerun, which:

1. Crops the drawing to its bounding box
2. Resizes it to fit a 20×20 box, preserving aspect ratio
3. Centers it on a blank 28×28 canvas
4. Re-centers it by center of mass — matching how MNIST digits were originally prepared
5. Scales it with the saved `MinMaxScaler` and feeds it to the model
6. Displays the predicted digit, confidence %, and a per-class probability chart

This preprocessing step matters: a naive resize of a freehand drawing to 28×28 (without cropping/centering) performs noticeably worse, since real MNIST digits are small and centered within their frame rather than filling it edge to edge.

---

## Running Locally

```bash
git clone https://github.com/IoNiCx1/MNIST_numbers.git
cd MNIST_numbers
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

To retrain the model yourself, open `ProjectMNIST.ipynb` in Jupyter, Colab, or Kaggle and run it top to bottom — it will regenerate the files in `artifacts/`.

---

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io):
- Repository: `IoNiCx1/MNIST_numbers`
- Branch: `main`
- Main file: `app.py`

---

## Tech Stack

- **Model:** TensorFlow / Keras
- **Preprocessing:** scikit-learn (`MinMaxScaler`, `OneHotEncoder`), scipy (`ndimage`)
- **Tuning:** KerasTuner (`RandomSearch`, `Hyperband`)
- **App:** Streamlit, `streamlit-drawable-canvas`
- **Data/viz:** NumPy, pandas, Matplotlib, seaborn

---

## Next Steps

- Swap the flat ANN for a CNN — typically more robust to the position/stroke variation of hand-drawn digits
- Add data augmentation (rotation, shift, zoom) during training
- Try `kt.BayesianOptimization` for a smarter hyperparameter search

---

## License

MIT — see [LICENSE](LICENSE).
