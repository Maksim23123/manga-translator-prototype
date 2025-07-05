# manga-translator-prototype
Script for automatic manga translation

## Overview
This project is a prototype pipeline for automatic manga translation. It detects text areas in manga images, extracts the text using OCR, translates it to English, and reinserts the translated text into the image.

## Features
- Detects manga text bubbles and text areas using a deep learning model and EasyOCR
- Extracts Japanese text with OCR (MangaOCR)
- Translates text to English using an LLM API
- Inpaints original text areas and reinserts translated text
- Works with images in the `inputs/` folder and saves results to `outputs/`

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Usage
1. Place manga images in the `inputs/` folder (e.g., `inputs/p (1).jpg`).
2. Set up your configuration in `config.json` (see below).
3. Run the notebook `translation_pipline_prototype.ipynb` step by step.
4. Translated images will be saved in the `outputs/` folder.

## Configuration
Create a file named `config.json` in the project root with the following structure:
```json
{
  "API_KEYS": {
    "INFERENCE_API_KEY": "your_inference_api_key",
    "TOGETHER_API_KEY": "your_together_api_key"
  },
  "IMPORT_PARAMS": {
    "input_folder": "inputs",
    "output_folder": "outputs",
    "default_image_extension": "jpg"
  }
}
```
**Do not commit your `config.json` to version control.**

## Folder Structure
- `inputs/` — Place your input manga images here
- `outputs/` — Translated images will be saved here
- `translation_pipline_prototype.ipynb` — Main notebook with the full pipeline
- `requirements.txt` — Python dependencies

## Notes
- The pipeline is a prototype and may require GPU and internet access for model inference and translation.
- For best results, use high-quality manga scans.

## License
This project is for research and educational purposes only.
