Student Performance Prediction

## Overview

This project aims to predict student performance based on various features such as study time, past grades, and other relevant factors. The project leverages machine learning techniques to create a model that can provide accurate predictions, helping educators and stakeholders make informed decisions.

## Project Structure

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">.
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
</code></div></div></pre>

## Getting Started

### Prerequisites

Ensure you have the following installed:

* Python 3.x
* pip

### Installation

1. Clone the repository:
   <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">git clone <repository_url>
   </code></div></div></pre>
2. Navigate to the project directory:
   <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">cd student-performance-prediction
   </code></div></div></pre>
3. Install the required packages:
   <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">pip install -r requirements.txt
   </code></div></div></pre>

### Running the Application

To run the application, execute the following command:

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">python application.py
</code></div></div></pre>

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

## License

This project is licensed under the MIT License - see the [LICENSE]() file for details.

## Acknowledgements

* Special thanks to my mentors and peers for their support.
* The data for this project was sourced from [source].

---

Here's the code for the updated README file:

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>markdown</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-markdown"># Student Performance Prediction

## Overview

This project aims to predict student performance based on various features such as study time, past grades, and other relevant factors. The project leverages machine learning techniques to create a model that can provide accurate predictions, helping educators and stakeholders make informed decisions.

## Project Structure

</code></div></div></pre>

.
├── .ebextensions # Deployment configuration
├── artifacts # Artifacts generated during training
│ ├── data.csv # Raw data file
│ ├── model.pkl # Trained model file
│ ├── preprocessor.pkl # Preprocessor object file
│ ├── test.csv # Test data file
│ └── train.csv # Training data file
├── catboost_info # Model training information
│ ├── learn # Learning information
│ ├── catboost_training.json # CatBoost training configuration
│ ├── learn_error.tsv # Learning error log
│ └── time_left.tsv # Time left for training
├── notebook # Jupyter Notebooks for model training
│ ├── catboost_info # CatBoost training logs
│ ├── data # Data directory for notebooks
│ ├── 1. EDA STUDENT PERFORMANCE.ipynb # Exploratory Data Analysis
│ └── 2. MODEL TRAINING.ipynb # Model training notebook
├── src # Application source code
│ ├── components # Core components of the application
│ │ ├──  **init** .py
│ │ ├── data_ingestion.py # Data ingestion module
│ │ ├── data_transformation.py # Data transformation module
│ │ └── model_trainer.py # Model training module
│ ├── pipeline # Pipelines for data processing and model training
│ │ ├──  **init** .py
│ │ ├── predict_pipeline.py # Prediction pipeline
│ │ ├── train_pipeline.py # Training pipeline
│ │ ├── exception.py # Custom exceptions
│ │ ├── logger.py # Logging utilities
│ │ └── utils.py # Utility functions
├── templates # HTML templates for the web application
│ ├── home.html
│ └── index.html
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── application.py # Main application file
├── requirements.txt # Python dependencies
└── setup.py # Setup and requirements

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>markdown</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-markdown">
## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
</code></div></div></pre>

2. Navigate to the project directory:
   <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">cd student-performance-prediction
   </code></div></div></pre>
3. Install the required packages:
   <pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">pip install -r requirements.txt
   </code></div></div></pre>

### Running the Application

To run the application, execute the following command:

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-bash">python application.py
</code></div></div></pre>

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

## License

This project is licensed under the MIT License - see the [LICENSE]() file for details.

## Acknowledgements

* Special thanks to my mentors and peers for their support.
* The data for this project was sourced from [source].
