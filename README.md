# Document Perspective Correction

This project implements document perspective correction using Hough Transform and RANSAC methods. It can detect and correct perspective distortion in document images.

## Features

- Custom implementation of Hough Transform for line detection
- RANSAC-based line fitting to handle outliers
- Quadrilateral detection for document boundaries
- Perspective correction transformation
- Evaluation using Structural Similarity Index (SSIM)
- Visualization of each processing step

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- Matplotlib
- scikit-image

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Generate a Sample Distorted Image

Run the sample creation script to generate a test image:

```
python create_sample.py
```

This will create a `sample_image.jpg` file with perspective distortion, and the original document in the `ground_truth` folder.

### Run the Perspective Correction

To process a single image:

```
python main.py
```

By default, it will process the `sample_image.jpg` file. Edit `main.py` to process your own images.

### Process a Dataset

To process a dataset with multiple images:

1. Organize your dataset in folders by class (e.g., `curved`, `fold`, `perspective`, etc.)
2. Update the path in `main.py` to point to your dataset
3. Run `python main.py`

## Method Details

1. **Preprocessing**: Convert to grayscale, apply Gaussian blur, detect edges using Canny
2. **Hough Transform**: Custom implementation to detect lines in the edge image
3. **RANSAC**: Refine the detected lines to handle outliers
4. **Quadrilateral Detection**: Find intersection points of lines and determine the document boundaries
5. **Perspective Correction**: Apply geometric transformation to restore the document to its frontal view

## Evaluation

The implementation evaluates corrected images against ground truth using the Structural Similarity Index (SSIM) measure from scikit-image. 