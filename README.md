# VisionChat — Complete Local Image Understanding

This project includes:
- **BLIP image captioning** for natural-language image descriptions.
- **YOLO11 nano** for object detection, bounding boxes, and object counts.
- **BLIP visual question answering** for questions about the uploaded image.

No Gemini key or paid API is required. The first use downloads model files; later runs use the local cache.

## Recommended setup: stable Python 3.12 (64-bit)

Your earlier Python 3.13.0b4 is a beta release. Install stable Python 3.12 from:
https://www.python.org/downloads/windows/

Select **Add Python to PATH** during installation.

Open PowerShell in this project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the localhost URL printed in the terminal, usually http://localhost:8501.

If activation is blocked, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```
Then activate again.

## First run and hardware

The first run downloads BLIP captioning, BLIP VQA, and YOLO11 nano model files. Internet is required for these downloads. CPU inference is supported but may be slow. A compatible GPU can improve speed.

## Limitations

Descriptions and answers can be wrong or incomplete. YOLO only detects categories in its training dataset. The project does not guarantee identification of religious figures, people, exact brands/models, or every small object. It is a demonstration tool, not a factual verification system.

If installation fails, share the complete error beginning at the first `ERROR:` line. Use this fresh `.venv`; do not mix it with your old `.venv` or `.venv313`.
