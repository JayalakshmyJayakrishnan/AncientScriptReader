# Ancient Script Reader

## Overview
This project aims to develop a system for translating ancient texts, lost scripts, and decipher them into English. Initially, I'm working with Sanskrit texts. The aim is to convert Sanskrit texts (including ancient scripts like those from the *Valmiki Ramayana*) into English using Optical Character Recognition (OCR) and machine learning. The solution leverages pre-trained models (e.g., Helsinki-NLP/opus-mt-hi-en) for initial translation, with plans to fine-tune and enhance accuracy using advanced models like IndicTrans or ByT5-Sanskrit in future phases. 
## Features
- **OCR Extraction:** Converts Sanskrit text from images (e.g., `editedsan.jpg`) into machine-readable format using Tesseract OCR.
- **Translation:** Implements a dictionary-based translation for initial Sanskrit-to-English conversion, with fine-tuning using Hugging Face Transformers.
- **Training:** Fine-tunes a pre-trained model on a custom dataset (`Sanskrit.txt` and `English.txt`) for improved accuracy.
- **Future Enhancements:** Plans to integrate higher-accuracy models (e.g., IndicTrans, ByT5-Sanskrit) and deploy a web-based UI.

## Project Status
This project is currently under progress. The initial setup, OCR, and training pipeline are being developed and tested. Training is scheduled to complete after May 23, 2025, with ongoing refinements planned for subsequent phases.

## Requirements
- **Python 3.13**
- **Libraries:**
  - `pytesseract`
  - `opencv-python`
  - `pillow`
  - `transformers`
  - `torch`
  - `datasets`
  - `sentencepiece` (required for tokenizer)
- **Tools:**
  - Tesseract-OCR (installed at `C:\Program Files\Tesseract-OCR\`)
  - CMake (installed and added to PATH)
  - Visual Studio Build Tools
 

**Wish me luck ヾ(≧▽≦*)o**


