
# Student Performance Prediction

## Overview

This project aims to predict student performance based on various features such as study time, past grades, and other relevant factors. The project leverages machine learning techniques to create a model that can provide accurate predictions, helping educators and stakeholders make informed decisions.

## Project Structure

```bash

├── .ebextensions         # Deployment configuration
├── artifacts             # Artifacts generated during training
│   ├── data.csv          # Raw data file
│   ├── model.pkl         # Trained model file
│   ├── preprocessor.pkl  # Preprocessor object file
│   ├── test.csv          # Test data file
│   └── train.csv         # Training data file
├── catboost_info         # Model training information
│   ├── learn             # Learning information
│   ├── catboost_training.json # CatBoost training configuration
│   ├── learn_error.tsv   # Learning error log
│   └── time_left.tsv     # Time left for training
├── notebook              # Jupyter Notebooks for model training
│   ├── catboost_info     # CatBoost training logs
│   ├── data              # Data directory for notebooks
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb # Exploratory Data Analysis
│   └── 2. MODEL TRAINING.ipynb # Model training notebook
├── src                   # Application source code
│   ├── components        # Core components of the application
│   │   ├── __init__.py
│   │   ├── data_ingestion.py # Data ingestion module
│   │   ├── data_transformation.py # Data transformation module
│   │   └── model_trainer.py # Model training module
│   ├── pipeline          # Pipelines for data processing and model training
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py # Prediction pipeline
│   │   ├── train_pipeline.py # Training pipeline
│   │   ├── exception.py # Custom exceptions
│   │   ├── logger.py # Logging utilities
│   │   └── utils.py # Utility functions
├── templates             # HTML templates for the web application
│   ├── home.html
│   └── index.html
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
├── application.py        # Main application file
├── requirements.txt      # Python dependencies
└── setup.py              # Setup and requirements

```



## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:

```bash
cd student-performance-prediction
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```


### Running the Application

To run the application, execute the following command:

```bash
python application.py
```


This will start the application and you can access it via your web browser.

## Project Details

### Data

The dataset used for this project includes various features that can influence student performance, such as:

* `gender`: sex of students -> (Male/Female)
* `race/ethnicity`: ethnicity of students -> (Group A, B, C, D, E)
* `parental level of education`: parents' final education -> (bachelor's degree, some college, master's degree, associate's degree, high school)
* `lunch`: having lunch before test -> (standard or free/reduced)
* `test preparation course`: completion status before test -> (complete or not complete)
* `math score`: score in math
* `reading score`: score in reading
* `writing score`: score in writing

### Models

The project utilizes multiple machine learning models to predict student performance. The models used include:

* Linear Regression
* Lasso
* Ridge
* K-Neighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor
* XGBRegressor
* CatBoost Regressor
* AdaBoost Regressor

After evaluating all the models, Linear Regression performed the best and was selected for making the final and proper model.

### Training

The model training process is documented in the `notebook` directory, which contains Jupyter Notebooks detailing the steps and results.

## Deployment

The project is configured for deployment using AWS Elastic Beanstalk. The deployment configuration files are located in the `.ebextensions` directory.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.
