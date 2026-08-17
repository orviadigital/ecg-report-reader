# ECG Report Reader v4

This version is designed for a normal person to understand the extracted ECG numbers.

It provides:
- Your report value
- Typical adult reference range
- Green/yellow/red status
- Plain-language explanation
- Separate section for the machine-generated interpretation
- Special review warning for phrases such as ST elevation / possible anterior injury
- OCR confidence protection
- Sex selection for QTc reference
- PDF/photo support

Important:
This is an educational screening prototype. It cannot confirm that an ECG is normal and cannot diagnose disease. The actual 12-lead waveform must be reviewed by a qualified clinician.

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS Tesseract:
```bash
brew install tesseract
```
