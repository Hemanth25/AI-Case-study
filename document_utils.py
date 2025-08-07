# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:06:00 2025

@author: hemanthn
"""

# === document_utils.py ===
import fitz
from PIL import Image
import pytesseract
from config import TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in pdf)

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)