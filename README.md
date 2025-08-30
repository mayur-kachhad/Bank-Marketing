### Bank Term Deposit Subscription Prediction

#### Description
In the highly competitive banking sector, it's crucial for banks to optimize their marketing efforts. This project addresses the challenge by building a predictive model that identifies potential customers most likely to subscribe to a term deposit. By focusing on these high-probability customers, the bank can significantly reduce operational costs and boost the success rate of its campaigns.

The project is a binary classification problem that uses a predictive model to determine if a client will subscribe to a term deposit based on historical marketing data. The solution includes a full machine learning pipeline and a simple web application for making predictions.

Data: The dataset used in this project is the Bank Marketing Dataset, originally made publicly available by [Moro et al., 2014]. It consists of 41,188 records and 20 input features along with a binary target variable (y), which indicates whether a customer subscribed to a term deposit (yes or no). The features include client demographic details, previous campaign information, and economic indicators.
    
    Dataset Link : https://archive.ics.uci.edu/dataset/222/bank+marketing

##### Impact: Using the predictive model, the bank can potentially increase net profit by over 101% (from $10,407 to $20,953) and reduce marketing costs by 82% (from $29,555 to $5,205) compared to a non-targeted campaign


#### Installation & Usage
Prerequisites
    To run this project, you need to have 

    Python 3.12  or later installed.
    virtual environment(recommended).

Installation

    Clone the repository:
    
    git clone https://github.com/mayur-kachhad/Bank-Marketing.git

    cd Bank-Marketing

Install dependencies:

    pip install -r requirements.txt

Run the notebooks starting with eda and then whatever model you want. 


This project also includes a web application for easy prediction.

    Train the model: Run the training pipeline to prepare the model for predictions.

        python train_pipeline.py

    Run the web application:

        python app.py

    Access the application: Open your web browser and go to http://127.0.0.1:5000/ to use the prediction interface.

#### Project Structure

data/: Directory to store the eda_data.csv dataset.

Jupyter Notebooks:

    00_eda.ipynb: Exploratory Data Analysis to understand the dataset.


    01_model.ipynb: The main modeling approach, including standard scaling, one-hot encoding, model selection, hyperparameter tuning, and custom business matrix evaluation.


    02_smote_model.ipynb: An approach that includes resampling with SMOTETomek in the cross-validation loop to address data imbalance.


    03_fs_model.ipynb: An approach that includes feature selection in the cross-validation loop to avoid feature selection bias.


    04_model.ipynb: Combines both resampling (SMOTETomek) and feature selection within the cross-validation loop.


    binning_model.ipynb: Explores the effect of binning certain columns (like age) on the model's performance.


    cluster_model.ipynb: Uses clustering to segment the data and trains a separate predictive model for each segment.


    nn_model.ipynb: An approach that trains a predictive model using neural networks.

saved_model/: Directory where the trained model pipeline (lgbm_pipeline.pkl) is saved.

templates/ : user interface for making predictions.

app.py: The main Flask application that runs the web interface.

train_pipeline.py: The script for building and training the complete machine learning pipeline.

predict.py: Contains the function for making predictions using the trained model.

config.py: Stores all project configurations, including file paths, column names, and model hyperparameters.

requirements.txt: Lists all the necessary Python dependencies for the project.


#### approaches used to build predictive model and solve this problem

Approach 1

    Name: O1_model
    
        a. In this model we used standard scalar and one hot encoder in preprocessing pipeline.
        b. Find best model using ROC AUC score from all model dictionary.
        c. Done hyper parameter tuning on best model.
        d. Done model training on best parameters.
        e. Defined custom matrix to find optimal threshold.
        f. Model evaluation (classification report).
        g. SHAP values to explain features importance.


Approach 2

    Name: O2_smote_model
    
    Done all step as approach 1 but also included resampling using SMOTETomek in CV loop to avoid data leakage.


Approach 3

    Name: O3_fs_model
    
    Done all step as approach 1 but also included feature selection in cv loop to avoid feature selection bias.

Approach 4

    Name: O4_model
    
    Done all step as approach 1 but also included resampling using SMOTETomek and feature selection in CV loop.

Approach 5

    Name: binning_model
    
    Done best approach from above four with Binnig of some columns (e.g. age)

Approach 6

    Name : cluster_model
    
    Done segmentation using clustering .then used best approach from first four approaches and makes predictions models for each segment.

Approach 7

    Name : nn_model
    
    Done model training using neural networks .

Best performing model ( after hyperparameter tuning) : LightGBM

Best performing approach : approach 1(3,5,7 give nearly same results)