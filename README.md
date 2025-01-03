# Train_Resnet

## Overview
This project is a Python application that interacts with the Chouette Vision API to fetch images and manage machine learning experiments. It uses various services and tools such as MLflow for experiment tracking and Azure Blob Storage for data storage.

## Features
- Build dataset from the Chouette Vision API with respect to labels (i.e. vine, ground, grass), resnet model versions, and starting and end dates.
- Compile trainable Resnet model dynamically ()
- Track machine learning experiments using MLflow.
- Store experiments locally (Sqlite and Local Folder) or with Azure (Posgresql and Azure Blob Storage).
- Recovery of model weights and evaluation of the trained model on a test dataset.

## Installation
The following steps describe installation setup using bash command line, and pyenv. 
1. **Clone the repository**:
    ```sh
    git clone https://github.com/J-Pouzoulet/training_resnet.git
    cd training_resnet
    ```

2. **Create a virtual environment**:
    ```sh
    pyenv install 3.9.17  # Install a Python version
    pyenv virtualenv 3.9.17 myenv  # Create a virtualenv
    pyenv activate myenv   # Activate the environment
    pyenv local myenv  # Set the environment to be used locally
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt   # Install packages
    ```

4. **Set up environment variables**:
    Create a .env file in the root directory of your project and add the following variables:

    ```properties
    API_KEY=your_api_key
    API_URL=https://api.staging.chouette.vision/api/jobs/get-images/
    # if using local sqlite database
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    # if using postgresql database in azure
    PGHOST=your_postgresql_host
    PGUSER=your_postgresql_user
    PGPORT=5432
    PGDATABASE=your_postgresql_database
    PGPASSWORD=your_postgresql_password
    # if using azure blob storage
    AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_connection_string
    AZURE_STORAGE_ACCESS_KEY=your_azure_storage_access_key
    AZURE_STORAGE_ACCOUNT_NAME=your_azure_storage_account_name
    AZURE_STORAGE_CONTAINER_NAME=your_azure_storage_container_name
    AZURE_STORAGE_SAS_TOKEN=your_azure_storage_sas_token
    ```

## Usage

### Lauching MLflow UI
    The mlflow server can be started using the start_mlflow_server.py script. Launching the server will allow you to access the mlflow UI using your web browser at http://127.0.0.1:5000 
    ```sh
    python start_mlflow_server.py   # Lauch mlflow server
    ```
    By default the server will be configured to connect to local storage solution (Sqlite and Local Folder).
    To connect to Azure storage solution (Posgresql and Azure Blob Storage) use the following command:
    '''sh
    python start_mlflow_server.py --storage Azure
    ''' 

### Using the Notebooks for Model Training and Evaluation
    From the root directory of the project launch Jupyter server.
    ```sh
    jupyter notebook .    # Lauch Jupyter server
    ```
    Access Jupyter notebook by copying the url displayed in the adress bar of our web browser.  

1. **Train the ResNet model locally**:
    Open the train_resnet_model_local.ipynb notebook in Jupyter, follow the instructions and run the cells to train and log the ResNet model using local resources.

2. **Evaluate the ResNet model locally**:
    Open the evaluate_resnet_model_local.ipynb notebook in Jupyter, follow the instructions and run the cells to evaluate the ResNet model using local resources.

3. **Train the ResNet model using Azure**:
    Open the train_resnet_model_azure.ipynb notebook in Jupyter, follow the instructions and run the cells to train the ResNet model using Azure resources.

4. **Evaluate the ResNet model using Azure**:
    Open the evaluate_resnet_model_azure.ipynb notebook in Jupyter, follow the instructions and run the cells to evaluate the ResNet model using Azure resources.

### Using the MLflow CLI
    Training of the model can be performed using mlflow CLI (Command Line Interface) in the terminal. 

    The code was only tested for local backend solution (Sqlite and Local Folder).

    !!!The way MLflow CLI performs experiments and log paramaters and artifacts in the backend storage does not allowed environment variables to be set dynamically (like it is achieved in the notebook for instance)!!! 
    
    Before running experiments through the mlflow CLI, set the following environment variables in your shell session where experiments are run.

    For local storage solution (Sqlite and Local Folder):
    '''sh
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    '''
    Run mlflow experiment using the mlflow CLI 
    '''sh
    mlflow run . --experiment-name your-experiment
    '''
    MLflow will read the MLproject file and launch the execution of the train.py file. Additional parameters can be passed such as seen in the example below
    '''sh
    mlflow run . --model_name ResNet50 --labels vine --start_date 2021-05-27 --end_date 2021-06-01 --experiment-name your-experiment
    '''
    To get addition information about accepted parameters and their formats
    '''sh
    python train.py --help
    ''' 

## Project structure

    ### Description of Key Files and Directories

    - `__pycache__/`: Directory containing compiled Python files.
    - `.env`: Environment variables file containing configuration settings such as API keys and database credentials.
    - `.gitignore`: Specifies files and directories to be ignored by Git.
    - `.python-version`: Specifies the Python version used for the project.
    - `evaluate_resnet_model_azure.ipynb`: Jupyter notebook for evaluating the ResNet model using Azure resources.
    - `evaluate_resnet_model_local.ipynb`: Jupyter notebook for evaluating the ResNet model locally.
    - `label_mapping.json`: JSON file containing label mappings for the dataset.
    - `media/`: Directory for storing media files.
    - `MLproject`: MLflow project file defining the structure and dependencies of the machine learning project.
    - `mlruns/`: Directory containing MLflow experiment runs.
    - `python_env.yaml`: YAML file specifying the Python environment and dependencies.
    - `README.md`: Project's README file containing an overview, installation instructions, usage, and more.
    - `requirements.txt`: File listing the Python dependencies required for the project.
    - `start_mlflow_server.py`: Script to start the MLflow server.
    - `train_resnet_model_azure.ipynb`: Jupyter notebook for training the ResNet model using Azure resources.
    - `train_resnet_model_local.ipynb`: Jupyter notebook for training the ResNet model locally.
    - `train.py`: Python script for training the ResNet model.
    - `utils/`: Directory containing utility scripts.
    - `build_dataset.py`: Script for building the dataset.
    - `build_model.py`: Script for building the machine learning model.

