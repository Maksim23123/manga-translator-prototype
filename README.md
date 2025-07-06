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
2. Copy `config.template.json` to `config.json` in the project root. This will be your personal configuration file.
3. Edit `config.json` to add your API keys and adjust parameters as needed (see below for details).
4. Run the notebook `translation_pipline_prototype.ipynb` step by step.
5. Translated images will be saved in the `outputs/` folder.

## Configuration

1. **Copy and rename:**
   - Duplicate `config.template.json` and rename it to `config.json` in the project root.

2. **Edit your API keys:**
   - Open `config.json` and fill in your API keys:
     - `INFERENCE_API_KEY`: Required for model inference. [Get your key here](https://roboflow.com/) (Roboflow API).
     - `TOGETHER_API_KEY`: Required for translation. [Get your key here](https://www.together.ai/docs/inference/getting-started) (Together API).

3. **Adjust import parameters if needed:**
   - `input_folder`: Folder with input images (default: `inputs`)
   - `output_folder`: Folder for translated images (default: `outputs`)
   - `default_image_extension`: Image file extension (default: `jpg`)
   - `input_file_name`: Name of the input file without extension (e.g., `p (1)`)

Example `config.json` structure:
```json
{
  "API_KEYS": {
    "INFERENCE_API_KEY": "your_inference_api_key",
    "TOGETHER_API_KEY": "your_together_api_key"
  },
  "IMPORT_PARAMS": {
    "input_folder": "inputs",
    "output_folder": "outputs",
    "default_image_extension": "jpg",
    "input_file_name": "p (1)"
  }
}
```

**Important:**
- Never commit your `config.json` to version control, as it contains sensitive information.

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

## Project Status
This repository is an early prototype of the Manga Translator project.
The full version is now in active development here:
 [Manga Translator – Full Version](https://github.com/Maksim23123/manga-translator)