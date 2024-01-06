IMAGE EXTRACTION - Simple CNN (*2021*)
=============================

Overview
--------

This project is dedicated to the challenging task of predicting missing parts of images, specifically focusing on image borders. It employs a Convolutional Neural Network (CNN) approach to intelligently extrapolate and fill in missing image data.

Dataset
-------

The dataset, a private collection of approximately 40,000 images, plays a crucial role in training our model. Each image has been converted to grayscale using specific channel weightings:

*   Red: 0.2989
*   Green: 0.5870
*   Blue: 0.1140

Files and Usage
---------------

### `img_preprocessing.py`

This script includes various functions necessary for image preprocessing:

*   **Resizing:** Converts each image to 90x90 pixels.
*   **Border Removal:** Randomly removes a specified border from each image, ensuring a minimum of 5 pixels are removed from each axis. The removed portion is saved as a target array.
*   **Array Creation:** Generates input arrays of 75x75 pixels (excluding borders) and corresponding mask arrays.
*   **Examples:** Demonstrations of these functions are available at the end of the script.

### `best_model3.py`

Contains the weights of our most efficient model, which is integral to the image prediction process.

### `image_extrapolation.ipynb`

A comprehensive Jupyter notebook that encompasses the entire workflow:

*   **Preprocessing:** Detailed explanation and execution of image preprocessing steps.
*   **Training:** Utilizing Google Colab's GPU for efficient model training.
*   **Evaluation:** Assessing the model's performance.
*   **Architecture:** The model consists of 3 CNN hidden layers, each followed by a ReLU activation layer.
*   **Hyperparameters:**
    *   Kernel size: (7,7)
    *   Stride: (1,1)
    *   Padding: (3,3)

Output Examples
---------------

Visual demonstrations of the model's output can be found here: <br> ![Model Output](https://github.com/Giuliasdfghjk/Giulia-portfolio/assets/80102658/80bc8dc0-9c6c-4518-9b4e-f5f6b446283c)


￼

