# Oil Spill Detection

![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Object detection model that identifies oil spills in images using YOLOv11, trained on the [oil-detection-2](https://universe.roboflow.com/yolo-m1lny/oil-detection-2) dataset from Roboflow Universe.

![Oil Spill Detection Demo](static\results\result_a70e23d6a0584b67adb685cc853a05df_Screenshot 2026-09-08 232104.png)

## Overview

This project fine-tunes a YOLOv11 model to classify and localize oil spills, distinguishing between `oil` and `no-oil` regions in an image. It includes training, inference, and a simple web interface for testing the model on new images.

## Classes

- `oil`
- `no-oil`

## Dataset

- **Source:** [oil-detection-2 on Roboflow Universe](https://universe.roboflow.com/yolo-m1lny/oil-detection-2)
- **Size:** 1,564 images
- **Format:** YOLOv11

## Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv11 |
| Epochs | 100 |
| Patience | 50 |
| Image size | 640 |

## Results

Validation metrics after training:

| Metric | Score |
|---|---|
| mAP50-95 | 0.7838 |
| mAP50 | 0.9810 |
| mAP75 | 0.8525 |
| Mean Precision | 0.9577 |
| Mean Recall | 0.9013 |

## Project Structure

```
.
├── train.py             # Flask web app — upload an image, run detection, view results
├── static/
│   ├── uploads/          # Uploaded images
│   └── results/          # Annotated output images
└── weights/
    └── best.pt            # Trained model weights
```

## Usage

### Run the web app

```bash
python app.py
```

Then open `http://localhost:5000` in your browser, upload an image, and view the detection results with confidence scores.


## Requirements

```bash
pip install ultralytics roboflow flask opencv-python
```

## License

Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).