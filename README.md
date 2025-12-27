# Pneumonia Detection on Chest X-Rays with DenseNet, EfficientNet, and Swin Transformer

This project trains and evaluates deep learning models to classify chest X-ray images as NORMAL or PNEUMONIA using the Kaggle dataset. Models explored:
- DenseNet-121 (CNN)
- EfficientNet-B3 (CNN)
- Swin Transformer Tiny (ViT)

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

## Repository structure

- .gitignore (ignoring large files, e.g. saved models and datasets)
- README.md
- Notebooks
  - [DenseNet-121_model.ipynb](DenseNet-121_model.ipynb)
  - [EfficientNet-B3_model.ipynb](EfficientNet-B3_model.ipynb)
  - [Swin-Transformer_model.ipynb](Swin_Transformer_model.ipynb)

## Quick start

1) Environment (Python 3.9–3.11 recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Dataset
- Download from Kaggle and extract into `Dataset/chest_xray/` so it matches the structure above.

3) Run in VS Code
- Open one of the notebooks:
  - [DenseNet-121_model.ipynb](DenseNet-121_model.ipynb)
  - [EfficientNet-B3_model.ipynb](EfficientNet-B3_model.ipynb)
  - [Swin-Transformer_model.ipynb](Swin-Transformer_model.ipynb)
- Ensure any data path variables point to `Dataset/chest_xray/`.
- Run all cells to train; best weights are saved to [checkpoints/](checkpoints/).

## What each notebook does

- DenseNet-121: Trains a compact CNN that reuses features via dense connectivity.
- Input size typically 224×224, ImageNet normalization, standard augmentations.
- EfficientNet-B3: Scaled CNN with compound depth/width/resolution scaling.
- Input size typically 300×300.
- Swin Transformer Tiny: Hierarchical transformer with shifted windows for efficiency.
- Input size typically 224×224.

All notebooks include training, validation, test evaluation, and checkpointing to [checkpoints/](checkpoints/).

## Evaluation

Common metrics:
- Accuracy: $ \text{Acc}=\frac{TP+TN}{TP+FP+TN+FN} $
- Precision: $ \text{Prec}=\frac{TP}{TP+FP} $
- Recall (Sensitivity): $ \text{Rec}=\frac{TP}{TP+FN} $
- F1: $ \text{F1}=2\cdot\frac{\text{Prec}\cdot\text{Rec}}{\text{Prec}+\text{Rec}} $

Use the “Evaluation” section in each notebook to:
- Compute metrics on [Dataset/chest_xray/test](Dataset/chest_xray/test)
- Plot confusion matrix and ROC curves

## Streamlit App

An interactive UI to classify chest X-rays using the trained EfficientNet-B3 checkpoint.

### Run the app

```bash
# Install dependencies (after activating your virtual env)
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py
```

Then open the local URL shown in the terminal, and upload a chest X-ray (JPG/PNG). The app displays the predicted class and confidence with a progress bar.
