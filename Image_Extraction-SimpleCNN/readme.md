MAGE EXTRACTION - Simple CNN - Giulia Bianchi - 2021.

This project aims to predict missing parts of an image (borders in this case).
The dataset used is privately collected and consist of ~40,000 images, converted to grayscale with with the following weighting of the colour channels: r=0.2989, g=0.5870, b=0.1140.

`img_preprocessing.py`:<br>
contains some functions to:<br>
-resize every datapoint to a 90x90 pixel image.<br>
-remove (set to zero) specified border of an image randomly (at least 5 pixel for each axis). The removed part is stored as target array.<br>
-create input arrays (75x75 pixel) without borders and mask arrays.<br>
Examples can be found at the end of the main file.<br>

`best_model3.py`:<br>
Contains the weights of the best model.

`image_extrapolation.ipynb`: <br>
The notebook complete with preprocessing, training and evaluation. The training has been done with Google Colab GPU, the architecture is pretty simple and consist of 3 CNN hidden layers (+1 output), each followed by ReLU activation layer. Hyperparameters size are:<br>
kernel = (7,7)<br>
stride =(1,1)<br>
padding =(3,3)<br>
<br>
Output examples:<br>

![download](https://github.com/Giuliasdfghjk/Giulia-portfolio/assets/80102658/80bc8dc0-9c6c-4518-9b4e-f5f6b446283c)

￼

