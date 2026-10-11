# Plant Disease Prediction

A convolutional neural network that classifies plant leaf images as **Healthy**, **Powdery** (powdery mildew) or **Rust**. It comes with a Flask web app: upload a leaf photo and get a prediction.

## How it works

1. `Model_Training.ipynb` trains a Keras CNN on the leaf images in `Dataset/` and saves it as `model.h5`.
2. `app.py` loads `model.h5` and serves a web page.
3. When an image is uploaded:
   - it is resized to 225×225 and scaled to 0–1;
   - the model returns probabilities for the three classes;
   - the most likely label is returned.

## Model

| Layer | Configuration | Output shape |
|---|---|---|
| Input | RGB image | 225 × 225 × 3 |
| Conv2D | 32 filters, 3×3, ReLU | 223 × 223 × 32 |
| MaxPooling2D | 2×2 | 111 × 111 × 32 |
| Conv2D | 64 filters, 3×3, ReLU | 109 × 109 × 64 |
| MaxPooling2D | 2×2 | 54 × 54 × 64 |
| Flatten | | 186,624 |
| Dense | 64, ReLU | 64 |
| Dense | 3, softmax | 3 |

## Dataset

Images are organised by split and class:

```
Dataset/
  Train/Train/{Healthy,Powdery,Rust}/            1,322 images
  Validation/Validation/{Healthy,Powdery,Rust}/     60 images
  Test/Test/{Healthy,Powdery,Rust}/                150 images
```

## Results

Trained for 5 epochs (from `Model_Training.ipynb`):

| Epoch | Training accuracy | Validation accuracy |
|---|---|---|
| 1 | 0.5038 | 0.6000 |
| 2 | 0.6891 | 0.7167 |
| 3 | 0.7413 | 0.8000 |
| 4 | 0.8222 | 0.8167 |
| 5 | 0.8328 | 0.8333 |

These are early results from a small model and a 60-image validation set:

- The test split has not been evaluated as a whole yet.
- In the notebook's sample prediction, a Rust test image was classified as Healthy (0.51 vs 0.49).

Data augmentation, more epochs, and transfer learning from a pretrained backbone are the obvious next steps.

## Tech stack

- Python, TensorFlow / Keras
- NumPy, Pillow, OpenCV
- Flask, Jinja2 templates, Bootstrap, jQuery
- Jupyter Notebook for training

## Getting started

`model.h5` (about 144 MB) is stored with [Git LFS](https://git-lfs.com). Install it before cloning, or the file will only be a small pointer:

```bash
git lfs install
git clone https://github.com/Om-Salunkhe/Plant-Disease-Prediction.git
cd Plant-Disease-Prediction
```

Install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install tensorflow numpy pillow opencv-python flask
```

Run the web app:

```bash
python app.py
```

Open http://127.0.0.1:5000 and upload a leaf image. Uploaded files are saved to `uploads/`.

To retrain, open `Model_Training.ipynb` in Jupyter and run all cells. This overwrites `model.h5`.

## Project structure

```
Model_Training.ipynb   data loading, model definition and training
model.h5               trained model (Git LFS)
app.py                 Flask app: upload → preprocess → predict
templates/             upload page
static/                CSS and JS
Dataset/               train / validation / test images
uploads/               images uploaded through the web app
```
