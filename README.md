# training_resnet
resnet training automation

# Clone MLflow repot for project structure and example : git clone https://github.com/mlflow/mlflow.git     
# Install MLflow: pip3 install mlflow

# Run the following command in the terminal to mlflow server while specifying database location, the server will open and be listening locally to the port 5000, thus to the following url : "http://127.0.0.1:5000"
# Copy and paste the url in the browser adress to access the mlflow dashboard 
# Here the database will be stored in a local folder but it could a cloud file storage system (i.e. Azure Blob Storage, AWS S3)
mlflow server --backend-store-uri file:///home/jerome/code/J-Pouzoulet/training_resnet/mlflow-database
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root wasbs://mlflow@mlflowstoredjerome.blob.core.windows.net -h 0.0.0.0 -p 8080


# Here the database will be stored in a postgresql db on azure and artifacts on a blob storage

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root "$MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT" --host 127.0.0.1 --port 5000


# Now we need to connect mlflow to our experiment so that it can export model's artifacts after training
# To do so, in the terminal go to the folder where the notebook or the .py (ex. exercice1.ipynb) running the training experiment is located
# Then run the command : 
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_URI=http://localhost:8080

# Installation du SSL certificat pour la connection à la base de données posgresql sur azure 
openssl s_client -starttls postgres -showcerts -connect <your-postgresql-server-name>:5432
openssl s_client -starttls postgres -showcerts -connect posgresqlmlflow.postgres.database.azure.com:5432


export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=mlflowstoredjerome;AccountKey=RkLatKBzi0KcC+/VMWrhJuDtHqgtvQefLgaMCFyOKVLiB817Zv8R9bfOFMDCEuZqve7rXxoxiZOg+ASt37rCaA==;EndpointSuffix=core.windows.net"

export AZURE_STORAGE_ACCESS_KEY="RkLatKBzi0KcC+/VMWrhJuDtHqgtvQefLgaMCFyOKVLiB817Zv8R9bfOFMDCEuZqve7rXxoxiZOg+ASt37rCaA=="

export MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT="wasbs://mlflow@mlflowstoredjerome.blob.core.windows.net?sp=racwdli&st=2024-12-30T14:39:02Z&se=2025-01-30T22:39:02Z&sip=91.164.251.67&sv=2022-11-02&sr=c&sig=tR8%2BGUOViYmCdVp56TNehZWX%2Fm1wrsVYBtaiSLffdl0%3D"  


export MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT="https://mlflowstoredjerome.blob.core.windows.net/mlflow?sp=racwdli&st=2024-12-30T14:52:01Z&se=2025-01-30T22:52:01Z&sip=91.164.251.67&sv=2022-11-02&sr=c&sig=aLEynukClFcAw2dM9PmqCFgzu%2FqM2aDwHX4A5EZbniY%3D"
