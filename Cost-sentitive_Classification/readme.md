Identifying Emotion in Music: Cost-sentive classification
===============================

Overview
--------

This folder contains a series of Jupyter notebooks, each demonstrating a different ensemble approach for the same object: predicting emotion suscitateD by music pieces considering a specific cost for misclassification.

The emotion class are 4: happy, angry, sad and relaxed.

The cost matrix (the values are the revenue per month) is the following:

| True Emotion Class | Predicted Class 1 | Predicted Class 2 | Predicted Class 3 | Predicted Class 4 |
|--------------------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| 1 Happy      | 5                 | -5                | -5                | 2                 |
| 2 Angry      | -5                | 10                | 2                 | -5                |
| 3 Sad        | -5                | 2                 | 10                | -5                |
| 4 Relaxed    | 2                 | -5                | -2                | 5                 |


Evaluation Scales
--------

This model quantifies emotion it in terms of two independent dimensions called Valence and Arousal:
*  	Valence refers to "the intrinsic attractiveness/'good'ness (positive valence) or averseness/'bad'ness (negative valence) of an event, object, or situation. [...] For example, emotions popularly referred to as 'negative', such as anger and fear, have negative valence. Joy has positive valence." (Wikipedia)
*  	Arousal relates to the perceived intensity of an emotion, and to the level of (bodily and cognitive) excitation that goes with it – "how much it would raise your heartbeat" (for good or bad). For instance, "relaxed" or "bored" are affective states with low arousal, "excited" or "angry" states with high arousal (but different positive or negative valence).
For further illustration, Russel (1980) has a figure showing where different emotional states could be placed in these two dimensions:


![russel_graph1](https://github.com/BianchiGiulia/Portfolio/assets/80102658/430890fb-b588-4610-900e-68f9859b8e4a)

Note that the two dimensions are independent: All combinations of positive/negative valence and high/low arousal are possible

### Notebooks:
--------
All the notebooks follow the same pipeline, only the ensemble methods implementation is different. In the end, the total profit is calculated according to the cost matrix here above descripted.
The implementations include:

1.  `ensemble_adaboost_svr_valence_arousal.ipynb`: KernelPCA+ Adaboost + SVC = estimated profit of €1290.0

2.   `ensemble_svr_adaboost_valence_arousal.ipynb`: SVR+ KernelPCA + Adaboost = estimated profit of €1381.0

3.  `ensemble_svr_gmm_valence_arousal.ipynb`: SVR + KernelPCA + Gaussian Mixture = estimated profit of €1381.0

4. `neural_net.ipynb`: 3 layer Neural Network = estimated profit of €1294.0

Experimental Setup
--------------------------
After removing less informative features a train/test split with 20% test size was done.
For regression algorithms and networks like the Support Vector Regression the data was normalized by using the StandardScaler function on the dataset.Otherwise, the data was not augmented. Every time a random seed of 42 was used for the train/test split. The hyperparameter search for all was done using 3-fold cross validation and a 20% stratified train test split was used for evaluation of effectiveness.

Methodology
--------------------------

1.  Feature Selection: Both Sequential Feature Selector and Correlation matrix were used to reduce the feature space, only the most representative feature were kept.
2.  Preprocessing: as described in Experimental Setup hereabove.
Otherwise, the data was not augmented. Every time a random seed of 42 was used for the train/test split.
3.  Prediction Strategy: As a first try, I tried to predict the quadrant of a datapoint directly using Random Forest and Multi-layer Perceptron, but the results were the same of a Random Baseline. The target was then shifted, I came to the conclusion that valence and arousal should be the target to predict, since it is the combination of values from these two features that defines where the datapoint lies on the quadrants. To do so, the classify function was implemented.
   <img width="1097" alt="Screenshot 2024-01-31 at 16 08 50" src="https://github.com/BianchiGiulia/Portfolio/assets/80102658/336570ea-c6ea-498a-92be-235ae45016da">
   Multiple models were implemented, in the folder I only upload some ensemble method which I find most interesting. Here you can find a graph map of the process
<img width="1271" alt="Screenshot 2024-01-31 at 16 09 47" src="https://github.com/BianchiGiulia/Portfolio/assets/80102658/216c63d4-df02-46d8-a815-d57d28f40223">


Note
--------------------------
The best model was a SVR+SVR ensamble, which reached an estimated profit value of 1516.0€

Contributions and Feedback
--------------------------

Your contributions and feedback are welcome! Feel free to fork this repository, experiment with the code, and provide suggestions or improvements.

* * *



