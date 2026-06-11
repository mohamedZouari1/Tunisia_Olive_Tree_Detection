# Tunisia Olive Tree Detection 🫒🛰️

This repository contains an end-to-end deep learning pipeline for automatically detecting and segmenting olive groves from Sentinel-2 satellite imagery in Tunisia.

The model distinguishes between two different types of olive agriculture:
1. **Extensif (Traditional):** Widely spaced olive trees typical of central/southern Tunisia.
2. **Intensif (Modern):** High-density, closely planted olive orchards.
3. **Background:** Non-olive areas including cities, mountains, desert, and bare soil.

## Project Structure

### 1. Data Collection (Google Earth Engine)
We use Sentinel-2 imagery (10m/20m resolution) collected during the summer (June–August) to maximize the contrast between evergreen olive trees and dry summer soil.

- **`gee_sentinel2_olive_tunisia.js`**: Core script to export large mosaic regions from GEE with 14 bands (10 raw bands + 4 indices: NDVI, NDWI, EVI, SAVI).
- **`gee_auto_training_data.js`**: Script to generate and export massive amounts of training polygons automatically.
- **`gee_export_test.js`**: Quick script to export a small test region of 10-band Sentinel-2 data for running through the web app.

### 2. Training Pipeline (PyTorch)
The core model is a **U-Net** architecture with a **ResNet-34** encoder (often referred to as a U-Net ResNet-34), powered by `segmentation_models_pytorch`. It was trained on over 5,300 custom image chips (128x128 pixels, 14 channels) to ensure robust generalisation across deserts, cities, and agricultural zones.

- **`train_unet_olive.py`**: The main script for training the U-Net model and calculating dataset statistics.
- **`chip_training_v2.py`**: Script to slice large Sentinel-2 TIFFs and vector polygons into `128x128` training chips and masks.
- **`chip_backgrounds.py`**: Script to add negative samples (e.g. cities, deserts) to the dataset with zero-masks to reduce false positives.
- **`band_stats.json`**: Pre-calculated normalisation statistics (mean/std) for the 14 Sentinel-2 bands across the entire dataset.
- **`best_unet_olive.pth`**: The final trained AI model file (PyTorch weights). Due to its large size (280MB), it is hosted externally: **[Download the model here](https://drive.google.com/drive/folders/1Lt8dijXG7dPuiPQVNIwxL2SpKkWvIOPI?usp=sharing)**. This is the actual "brain" of the project that has already learned to detect olive trees, meaning you can download it and run the web app immediately without needing to retrain anything.

### 3. Inference Web App (Flask)
A local web application that allows you to upload a Sentinel-2 GeoTIFF, run the AI model over it, and visualise the detections on an interactive map.

- **`app.py`**: The Flask backend server. Handles sliding-window inference and auto-computes NDVI/NDWI/EVI/SAVI if you upload raw 10-band Copernicus Browser images.
- **`web/index.html`**: The frontend UI for the web app.

## How to Run the Web App

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the trained model:** Download `best_unet_olive.pth` from **[this Google Drive link](https://drive.google.com/drive/folders/1Lt8dijXG7dPuiPQVNIwxL2SpKkWvIOPI?usp=sharing)** and place it in the root folder of this project.
3. Start the web server:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your web browser.
5. Upload a Sentinel-2 GeoTIFF. You can use `gee_export_test.js` in Google Earth Engine to easily export an area, or download a 10-band TIFF (B02-B8A, B11, B12) directly from the Copernicus Browser.

## Technical Details
- **Input Channels (14):** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, EVI, SAVI
- **Architecture:** U-Net (ResNet34 backbone)
- **Patch Size:** 128x128
- **Loss Function:** Combined CrossEntropyLoss and DiceLoss

## Acknowledgements
Data sourced from Copernicus Sentinel-2 missions via Google Earth Engine.
