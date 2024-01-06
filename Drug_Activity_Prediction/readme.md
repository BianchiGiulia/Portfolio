Drug Activity Prediction Project
==========================================================

Overview
--------

This project aims to predict the activity of <ins>**unseen compounds** </ins> based on their chemical properties. Utilizing advanced machine learning techniques, the project identifies potential correlations between compound characteristics and their biological activities, facilitating the discovery of effective drugs.

Focus on Advanced Resampling Techniques
--------------------------------
In the course of developing this project, a significant emphasis was placed on experimenting with resampling techniques, particularly exploring variations of the Synthetic Minority Over-sampling Technique ([SMOTE](https://arxiv.org/abs/1106.1813)). This initiative was driven by the need to address the challenges posed by imbalanced datasets, which is a common occurrence in drug activity prediction. The result table, with comparison of 4 variation of SMOTE algorithm (SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN)  can be found in the `report.pdf`.

Dataset
--------------

The dataset comprises 12,000 samples, where each sample includes a compound represented by its SMILES (Simplified Molecular Input Line Entry System) string. Accompanying each compound is its activity value, categorized into three compact, integer-based classes: 1 for 'active,' 0 for 'unknown,' and -1 for 'inactive.' This classification is specific to each compound-assay pair. The test dataset, in contrast, contains only the SMILES strings of compounds. The objective is to predict the activity status of these compounds.

Methodology
-------------------

*  **Preprocessing**: While SMILES strings effectively represent molecular compounds, their lack of uniqueness poses a challenge. To address this, we converted them into Morgan fingerprints using the RDKit library, following the parameters outlined in a relevant [ScienceDirect paper](https://www.sciencedirect.com/science/article/pii/S2666386422004155). This conversion ensures the uniqueness of each compound representation. Additionally, due to the data suffering from unbalanced and sparse label distribution, resampling was a crucial step in the preprocessing phase.

*  **Model and Training**: The study employed two distinct algorithms: Random Forest and Gaussian Naive Bayes. The Random Forest model was developed in two variations, each with different hyperparameters, as detailed in an MDPI study (refer to ["Molecules," Volume 28, Issue 4, 2023](https://www.mdpi.com/1420-3049/28/4/1663)). For the Gaussian Naive Bayes model, the `var_smoothing` parameter was optimized to 0.1 through grid search. Notably, the project involved training 11 separate models, one for each bioassay task.

Results
-----------
To see the table results with the AUC of both validation and test set please consult the report. Overall different resampling algorithm yield very different results, with an AUC ranging from 0.566 to 0.704. Moving forward, the next phase of this project will focus on developing task-specific models. This approach will involve experimenting with various feature combinations for each task, aiming to further improve the AUC metrics.

Usage
-----

To use this model:

1.  Clone the repository.
2.  Install required dependencies: `pip install -r requirements.txt`.
3.  Run the Jupyter Notebook: `Drug_activity_prediction.ipynb`.

Contributions and Feedback
--------------------------

Contributions to this project are welcome! Please feel free to fork the repository, make changes, and submit a pull request. For feedback and queries, open an issue in the repository or reach out to me directly.
