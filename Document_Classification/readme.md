Document Classification Project
===============================

Overview
--------

This folder contains a series of Jupyter notebooks, each demonstrating a unique approach to the same document classification task. These notebooks explore various methods ranging from standard machine learning techniques to advanced deep learning models, showcasing the strengths and differences of each approach.

### Notebooks:

1.  **Standard Machine Learning Approach** (`document_classification_standardML.ipynb`): This notebook implements two traditional machine learning algorithms for document classification: K-Nearest Neighbors (KNN) and Random Forest (RF) based on Term Count (tc) and Term Frequency - Inverse Document Frequency(tf-idf) inputs.
2.  **Word Embeddings** (`doc_class_wembedd.ipynb`): Here, the focus is on using word embeddings, specifically Word2Vec, for enhancing the classification model's performance of four algorithms: K-Nearest Neighbors (KNN), Random Forest (RF), Multi Layer Perceptron (MLP), Singular Value Decomposition (SVD).
3.  **BERT with PyTorch** (`doc_class_pytorch_BERT.ipynb`): This notebook utilizes the [BERT miniature model](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) implemented in PyTorch, demonstrating the power of transformers in NLP tasks.
4.  **LSTM and Transformers** (`Document_Classification_LSTM+Transformers.ipynb`): A combination of LSTM and Transformer models is explored in this notebook to leverage the strengths of both architectures and study the effect of different architecture choices and (hyper)parameter settings.

Dataset
-------
In all four notebooks the dataset is a subset of the [`HumSet` dataset](https://blog.thedeep.io/humset/) produced by the DEEP (https://www.thedeep.io) platform, an open-source platform.  
The input are unprocessed text excerpts, extracted from news and reports into a set of domain specific classes. The provided dataset has 12 classes (labels) like agriculture, health, and protection.



Results and Analysis
--------------------

Please consult notebooks. Each of them has formulas, observations, results and plots to describe the maths of the methods used and reports of the results.

How to Use
----------

To use these notebooks:

1.  Clone the repository.
2.  Ensure you have the necessary dependencies installed (listed in `requirements.txt`).
3.  Run the desired Jupyter notebook to see the methodology and results.

Contributions and Feedback
--------------------------

Your contributions and feedback are welcome! Feel free to fork this repository, experiment with the code, and provide suggestions or improvements.

* * *


