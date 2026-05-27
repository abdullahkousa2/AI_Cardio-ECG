# CardioAI — ECG Heart Diagnosis

An AI-powered web application that classifies ECG images into 4 cardiac conditions using **EfficientNetV2-S** with **Grad-CAM** heatmap visualization.

![Accuracy](https://img.shields.io/badge/Accuracy-98.61%25-brightgreen)
![Model](https://img.shields.io/badge/Model-EfficientNetV2--S-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%2B%20PyTorch-orange)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Abdullah2026/AI-Cardio-ECG)

---

## Demo

> Upload an ECG image → get the predicted condition + confidence bars + Grad-CAM heatmap showing which regions drove the decision.

<!-- Add a screenshot or GIF here: drag your demo image into the GitHub editor and it will upload automatically -->
<!-- Example: ![CardioAI Demo](assets/demo.gif) -->

---

## Features

- 98.61% test accuracy across 4 cardiac conditions
- Grad-CAM heatmaps highlighting the ECG regions that drove the prediction
- Drag-and-drop web interface with real-time confidence bars
- REST API for programmatic access
- Docker support for CPU and GPU deployment

---

## Cardiac Conditions Detected

| Class | Description |
|-------|-------------|
| Myocardial Infarction | Active heart attack ECG pattern |
| History of MI | Post-infarction ECG changes |
| Abnormal Heartbeat | Arrhythmia / irregular rhythm |
| Normal | Healthy ECG pattern |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | EfficientNetV2-S + SE-Block + Dual Pooling (PyTorch + timm) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Explainability | Grad-CAM heatmap overlay |

---

## Quick Start

### Option 1 — Local (Conda)

```bash
conda activate ocr
cd "heart diagnose"
pip install -r requirements.txt
python -m uvicorn app.main:app
```

Open `http://localhost:8000`

---

### Option 2 — Docker (CPU)

```bash
docker compose up --build
```

Open `http://localhost:8000`

---

### Option 3 — Docker (GPU — requires NVIDIA Container Toolkit)

```bash
docker compose -f docker-compose.gpu.yml up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Server health + class names |
| `POST` | `/predict` | Classify an ECG image |

### `/predict` — Request

```
POST /predict
Content-Type: multipart/form-data

file: <image file>   (JPG, PNG, BMP — max 10 MB)
```

### `/predict` — Response

```json
{
  "predicted_class": "Myocardial Infarction",
  "confidence": 0.987,
  "probabilities": {
    "Myocardial Infarction": 0.987,
    "History of MI": 0.008,
    "Abnormal Heartbeat": 0.003,
    "Normal": 0.002
  },
  "original_image": "<base64 PNG>",
  "gradcam_image": "<base64 PNG with heatmap overlay>"
}
```

---

## Model Architecture

```
tf_efficientnetv2_s (pretrained backbone)
  └── SE-Block (channel attention)
  └── Dual Pooling (GAP + GMP → 2560-dim)
  └── Classifier: Linear → BN → GELU → Dropout(0.4)
                → Linear → BN → GELU → Dropout(0.25)
                → Linear(4)
```

- Input size: 300 × 300 px
- Normalization: mean = std = 0.5
- Training: 3-phase strategy (frozen → partial unfreeze → full fine-tune)
- Final val accuracy: **98.61%**

---

## Dataset

- **3,951 ECG images** — train + test split
- 4 classes: MI, History of MI, Abnormal Heartbeat, Normal
- Source: ECG Image Dataset (Kaggle)

---

## Disclaimer

> This tool is for **research and educational purposes only**.  
> It does not replace professional clinical diagnosis.
