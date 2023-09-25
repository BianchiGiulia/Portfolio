IMAGE EXTRAPOLATION - 2021 - GIULIA BIANCHI - first project
<br> 
----------------------------------------------------------------------------------------------------------------------------------------
<br> 
A neural network (CNN) to extrapolate unknown border pixels of an image.

• The test set images have a shape of (90, 90) pixels. <br>
• The area of known pixels in the test set images is at least (75, 75) <br>
• The borders containing unknown pixels in the test set images are at least 5 pixels
  wide on each side. 
  
----------------------------------------------------------------------------------------------------------------------------------------
 The Dataset is private and has been collected by JKU students 

----------------------------------------------------------------------------------------------------------------------------------------

 <br> The Architecture is very simple, namely: <br>
SimpleCNN(<br>
  (hidden_layers): Sequential(<br>
    (0): Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))<br>
    (1): ReLU()<br>
    (2): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))<br>
    (3): ReLU()<br>
    (4): Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))<br>
    (5): ReLU()<br>
  )<br>
  (output_layer): Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))<br>
)<br>


----------------------------------------------------------------------------------------------------------------------------------------
<br> <br>  The project consist of:<br>
<b>Image_extrapolation_CNN.ipynb :</b> main file<br>
<b>ime_preprocess.py:</b> help functions to preprocess images.<br>
<b>best_model3.py:</b> best model weights.
