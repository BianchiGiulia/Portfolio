Comprehensive Data Analysis of Airbnb Amsterdam Listings
========================================================================

Overview
------------

This project is a thorough application of data analysis techniques on the Airbnb Amsterdam dataset, acquired from [Inside Airbnb](http://insideairbnb.com/get-the-data.html). The core objective is to uncover patterns, trends, outliers, and unusual distributions within the data, providing valuable insights into the Amsterdam Airbnb market.

**N.B.**: since the size of the notebook and html file was too big, it has been uploaded without outputs. This latter can be seen in the relevant .pdf file.

Data Source
-----------

The dataset used in this analysis is the Airbnb Amsterdam listing data, which can be downloaded from the Inside Airbnb website. This dataset includes detailed information about Airbnb listings in Amsterdam, such as location, price, reviews, and more.

Methods and Structure of the Notebook:
-------------------------------------

1.  **Data Formatting and Preprocessing**: Cleaning the data by handling NaN values and other irregularities to ensure quality and consistency.
    
2.  **Descriptive Statistics Analysis**: Employing descriptive statistics to analyze individual attributes. This includes investigating and plotting of feature distributions and computing summary statistics.
    
3.  **Correlation Analysis**: Investigating correlations between different attributes in the dataset to uncover relationships and dependencies (e.g. price/rating based on neighbourhood, response rate/review scores, prive/amenities and such).
    
4.  **Clustering Analysis**: Using PCA for dimensionality reduction to allow deplotment of various clustering algorithms to group similar listings. This will help identify patterns and segments within the Airbnb market in Amsterdam. The clustering has been run with: K-Means, BIRCH, Mini Batch K-Means, Agglomerative Clustering and Gaussian Mixture.
    
5.  **Visualization**: Utilizing a range of libraries for effective data visualization, including common ones like Matplotlib and Seaborn, as well as dynamic libraries like Altair for more advanced graphical representations (specifically, clustering analysis).
    


Potential further analysis
------------------------

*   Expanding the analysis to compare with data from other cities.
*   Performing time-series analysis to observe trends over different time frames.
*   Integrating external datasets for more comprehensive insights (e.g., tourism data).

Acknowledgements
----------------

Special thanks to Inside Airbnb for providing the dataset, and to the open-source community for the tools and libraries used in this project.

