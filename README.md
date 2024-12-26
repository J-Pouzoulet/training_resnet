# training_resnet
resnet training automation

# Clone MLflow repot for project structure and example : git clone https://github.com/mlflow/mlflow.git     
# Install MLflow: pip3 install mlflow

# Run the following command in the terminal to mlflow server while specifying database location, the server will open and be listening locally to the port 5000, thus to the following url : "http://127.0.0.1:5000"
# Copy and paste the url in the browser adress to access the mlflow dashboard 
# Here the database will be stored in a local folder but it could a cloud file storage system (i.e. Azure Blob Storage, AWS S3)
mlflow server --backend-store-uri file:///home/jerome/code/J-Pouzoulet/training_resnet/mlflow-database
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

# Now we need to connect mlflow to our experiment so that it can export model's artifacts after training
# To do so, in the terminal go to the folder where the notebook or the .py (ex. exercice1.ipynb) running the training experiment is located
# Then run the command : 
export MLFLOW_TRACKING_URI=http://localhost:5000
