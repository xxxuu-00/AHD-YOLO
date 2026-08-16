# AHD-YOLO

Official implementation of **AHD-YOLO: An Efficient Front-End Perception Framework for Vision-Based Drowning-Risk Monitoring in Complex Water-Surface Environments**.

AHD-YOLO is a lightweight **YOLO11n-based** detector for horizontal-view water-surface human-part detection. It integrates layer-wise **DRFD-HWD** downsampling and post-concatenation **ACA** attention.

## Environment

- Python 3.11.14
- PyTorch 2.8.0 + CUDA 12.9
- Ultralytics 8.4.21
- NVIDIA GeForce RTX 4060 Laptop GPU

Install dependencies:

```bash
pip install -r requirements.txt
