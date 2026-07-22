# Object Detection with a CNN (Classification + Bounding Box Regression)

A small end-to-end object detection project built with TensorFlow/Keras. A single CNN backbone with two output heads learns to **classify a handwritten digit** and **regress its bounding box** after the digit is randomly placed on a larger blank canvas.

## Overview

- **Task:** given a 75x75 grayscale image containing one digit pasted at a random position, predict (1) which digit it is (0-9) and (2) the normalized bounding box `[xmin, ymin, xmax, ymax]` around it.
- **Model:** 3 convolutional blocks (Conv2D + AveragePooling2D) shared backbone → dense layer → two output heads:
  - `class_output`: 10-way softmax (digit classification)
  - `bbox_output`: 4-value linear regression (bounding box)
- **Loss:** categorical cross-entropy (classification) + MSE (bounding box), trained jointly.
- **Evaluation:** classification accuracy and Intersection-over-Union (IoU) between predicted and ground-truth boxes.

## Dataset note

This was originally built around **MNIST** (via `tf.keras.datasets.mnist` / `tensorflow_datasets`). The version in this repo was executed in a sandboxed environment without access to the Google Cloud Storage endpoints those loaders use, so it substitutes **scikit-learn's built-in `load_digits`** dataset (1,797 8x8 digit images, ships with `scikit-learn`, no download required) so the notebook could be run genuinely end-to-end rather than left unexecuted.

The modeling code — architecture, losses, training loop, IoU calculation — is unchanged. To use full MNIST instead (recommended if you have internet access, since 60,000 images will noticeably improve both accuracy and IoU), just replace the data-loading cell in Section 3 with:

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

and adapt the padding/bounding-box logic to the 28x28 MNIST digits (the notebook already does this padding for the scikit-learn digits after upscaling them to 28x28).

## Results (this run, scikit-learn digits)

| Metric | Value |
|---|---|
| Training classification accuracy | ~94.0% |
| Validation classification accuracy | ~89.6% |
| Validation bounding-box loss (MSE) | 0.0083 |
| Average IoU (validation) | 0.52 |
| Fraction of boxes with IoU > 0.6 | ~33% |

IoU is modest mainly due to dataset size (1,797 vs. MNIST's 60,000 images) — the model sees much less positional and appearance variety.

## Files

- `OBJECT_DETECTION.ipynb` — the full notebook, already executed with real outputs (plots, model summary, training logs, predictions).
- `object_detection_model.keras` — the trained model, saved in Keras v3 format. Load it with:

```python
from tensorflow.keras.models import load_model
model = load_model("object_detection_model.keras")
```

- `training_graph.png` — classification loss and accuracy curves (train vs. validation) from this run.
- `sample_prediction.png` — ground-truth (green) vs. predicted (green if IoU > 0.6, else red) bounding boxes on 10 validation samples.
- `requirements.txt` — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook OBJECT_DETECTION.ipynb
```


## Future Scope

- Detect multiple objects per image
- Real-time webcam detection
- YOLO-style implementation for full multi-object detection
- Mobile deployment
- TensorFlow Lite deployment
