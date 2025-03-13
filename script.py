import pytesseract
from PIL import Image
import cv2
import os
import re
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Preprocessing and OCR functions (to sort out the yellowing, inconsistencies, clarity etc.)
def preprocess_image(file_path):
    try:
        img = cv2.imread(file_path)

        if img is None:
            raise FileNotFoundError(f"Error: Could not load image at {file_path}")

        # Resize for better OCR (balanced for single-line text)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise with balanced parameters
        gray_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)

        # Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_img = clahe.apply(gray_img)

        # Binarization with enhanced thresholding
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        # Crop edges to remove artifacts
        thresh_img = thresh_img[30:-30, 30:-30]
        return thresh_img
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def extract_text(image):
    try:
        if image is None:
            raise ValueError("No image provided for text extraction.")
        configs = [
            ("--oem 3 --psm 7", "Neural, Line"),
            ("--oem 3 --psm 11", "Neural, Sparse"),
            ("--oem 3 --psm 13", "Neural, Raw Line"),
            ("--oem 3 --psm 6", "Neural, Block"),
            ("--oem 1 --psm 7", "Legacy, Line")
        ]
        best_text = ""
        for config, config_name in configs:
            text = pytesseract.image_to_string(image, lang="san", config=config)
            cleaned_text = re.sub(r'[^\u0900-\u097F\s।॥]', '', text.strip())
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'(?<![\u0900-\u097F])\d+(?![\u0900-\u097F])', '', cleaned_text)
            cleaned_text = re.sub(r'([।॥])\s*([।॥])+', r'\1', cleaned_text)
            cleaned_text = re.sub(r'[\u200c\u200d]', '', cleaned_text)
            cleaned_text = re.sub(r'[^\u0900-\u097F\s।॥]{1,}', '', cleaned_text)
            cleaned_text = cleaned_text.rstrip('।।।।').rstrip('्').rstrip('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह').rstrip('।')
            print(f"{config_name} Extracted text (raw):", repr(text) if text.strip() else "No text detected")
            print(f"{config_name} Extracted text (cleaned):", repr(cleaned_text) if cleaned_text else "No text detected")
            
            devanagari_chars = sum(1 for c in cleaned_text if ord(c) >= 2304 and ord(c) <= 2431)
            if devanagari_chars > 15 and len(cleaned_text) > len(best_text):
                best_text = cleaned_text
        return best_text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return ""

def translate_text(sanskrit_text):
    if not sanskrit_text:
        return "No text to translate"

    # Expanded dictionary-based translation for the verse 
    translation_dict = {
        "एकाकिना": "Alone",
        "तपः": "austerities",
        "द्वाभ्यां": "by two",
        "पठनं": "reading",
        "गायनं": "singing",
        "त्रिभिः": "by three",
        "चतुर्भिः": "by four",
        "चतुर्भिर्": "by four",
        "गमनं": "traveling",
        "क्षेत्रं": "field",
        "पञ्चभिः": "by five",
        "पञ्चभिर्": "by five",
        "बहु": "many",
        "।": "",
        "॥": "",
        "भारतम्": "India",
        "मातृभूमिः": "motherland",
        "भारतीयाः": "Indians",
        "भ्रातरः": "brothers",
        "प्राणेभ्योऽपि": "more than life",
        "एवं": "thus",
        "प्रियतरा": "more beloved",
        "समृद्धौ": "prosperity",
        "विविध": "various",
        "संस्कृतौ": "cultures",
        "अभिमन्यामहे": "we take pride",
        "सदा": "always",
        "प्रयत्नमानाः": "striving",
        "सम्मानयेम": "we respect",
        "शिष्टतया": "with civility",
        "विश्वासपात्रतां": "trustworthiness",
        "संस्कृत": "Sanskrit",
        "प्रतिर्जा": "pledge",
        "रामः": "Rama",
        "सीता": "Sita",
        "हनुमान्": "Hanuman",
        "धर्मः": "duty/righteousness",
        "कुरुक्षेत्रे": "on the field of Kurukshetra",
        "ॐ तपः स्वाध्यायनिरतं": "Om, engaged in austerities and Vedic studies"
    }
    # Match and construct translation for full phrases
    translated_parts = []
    remaining_text = sanskrit_text
    for key in translation_dict:
        if key in remaining_text:
            translated_part = translation_dict[key]
            if translated_part:  # Only add non-empty translations
                translated_parts.append(translated_part)
            remaining_text = remaining_text.replace(key, '').strip()
    if translated_parts:
        return ' '.join(translated_parts) + (' ' + remaining_text if remaining_text and remaining_text not in ['।', '॥'] else '')
    return "Translation not found (no dictionary match)"

# Dataset preparation and training setup
def prepare_dataset():
    # Load existing dataset files
    with open(r"C:\Project1\Sanskrit.txt", 'r', encoding='utf-8') as f:
        sanskrit_lines = f.readlines()
    with open(r"C:\Project1\English.txt", 'r', encoding='utf-8') as f:
        english_lines = f.readlines()

    # Add the current verse and its translation (Tryn out samples here)
    current_sanskrit = "एकाकिना तपो द्वाभ्यां पठनं गायनं त्रिभिः चतुर्भिर्गमनं क्षेत्रं पञ्चभिर्बहु"
    current_english = "Alone austerities by two reading singing by three traveling field by five many"
    sanskrit_lines.append(current_sanskrit + "\n")
    english_lines.append(current_english + "\n")

    # Ensure equal length and create dataset
    min_length = min(len(sanskrit_lines), len(english_lines))
    sanskrit_lines = sanskrit_lines[:min_length]
    english_lines = english_lines[:min_length]
    
    data = {
        "source": [l.strip() for l in sanskrit_lines],
        "target": [e.strip() for e in english_lines]
    }
    dataset = Dataset.from_dict(data)
    return dataset

# Model and tokenizer setup
model_name = "Helsinki-NLP/opus-mt-hi-en"  # Hindi-to-English as a starting point for Sanskrit
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Training preparation (to be executed tomorrow 👾)
def setup_training():
    dataset = prepare_dataset()
    train_test_split = dataset.train_test_split(test_size=0.1)  # 90% train, 10% test
    dataset_dict = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples["source"], truncation=True, padding="max_length", max_length=128)
        targets = tokenizer(examples["target"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # Training arguments (to be used tomorrow)
    training_args = TrainingArguments(
        output_dir="./sanskrit_to_english_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False
    )

    # Trainer setup (to be executed tomorrow)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer
    )
    return trainer, tokenized_datasets

# Main execution for today (OCR, translation, and training setup)
file_path = r"C:\Project1\editedsan.jpg"

try:
    processed_image = preprocess_image(file_path)

    if processed_image is not None:
        # Save processed image for debugging
        cv2.imwrite(r"C:\Project1\processed_editedsan_clean.jpg", processed_image)

        extracted_text = extract_text(processed_image)

        print("Extracted Sanskrit Text:", repr(extracted_text) if extracted_text else "No text detected")

        translated_text = translate_text(extracted_text)

        print("Translated English Text (Dictionary):", translated_text)

        # Prepare dataset for tomorrow's training
        dataset = prepare_dataset()
        print("Dataset preview for training:", dataset[0])
        
        # Set up training (save for tomorrow)
        trainer, tokenized_datasets = setup_training()
        print("Training setup completed. Run `trainer.train()` tomorrow to start training.")
    else:
        print("Image preprocessing failed. Please check the file path and image format.")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
