import os
from dotenv import load_dotenv
from utils.build_dataset import *
from utils.build_model import *
import argparse
import mlflow
from azure.storage.blob import BlobServiceClient
import tempfile
import mlflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd

# We set up argument parser so that they can be passed from the command line
parser = argparse.ArgumentParser(description="Train a ResNet model on a dataset of images given the classes, the start_date and the end_date.")
parser.add_argument("--model_name", type=str, default="ResNet50", help="Accept ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2.")
parser.add_argument("--labels", type=str, nargs="+", default="vine", help="Accept 'vine', 'grass', 'ground' or combination of them. ex: --labels vine grass")
parser.add_argument("--start_date", type=str, default="2021-05-27", help="The starting date of the images to be used for training in YYYY-mm-DD format.")
parser.add_argument("--end_date", type=str, default="2021-06-01", help="The ending date of the images to be used for training in YYYY-mm-DD format.")
parser.add_argument("--experiment_name", type=str, default="my_experiment", help="The name of the experiement to log the results to in MLflow.")
parser.add_argument("--storage", type=str, default="Local", help="Whether storage solution is Sqlite + local folder or Postgresql + Azure Blob Storage")
parser.add_argument("--number_of_epoch", type=int, default=5, help="The number of epochs to train the model for.")

# We parse the arguments
args = parser.parse_args()

# We assign the arguments to variables
model_name = args.model_name
labels = args.labels
start_date = args.start_date
end_date = args.end_date
experiment_name = args.experiment_name
storage = args.storage

print(f"model: {model_name}")
print(f"labels: {labels}")
print(f"start_date: {start_date}")
print(f"end_date: {end_date}")
print(f"experiment_name: {experiment_name}")
print(f"storage: {storage}")

# We load the environment variables from the secret.env file
load_dotenv()

# We access environment variables using os.getenv()
api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")

print(f'---------------{os.environ["MLFLOW_TRACKING_URI"]}----------------')
#print(f'----------------{os.environ["ARTIFACT_LOCATION"]}----------------')

if storage == "Azure":
    # We need the follow variables to connect to the Azure Blob Storage
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if connection_string:
        # Set it in the shell environment
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string
        print("Connection string set in the shell environment.")
    
    # We need the follow variables to connect to the Azure Posgresql Database
    pghost = os.getenv("PGHOST")
    pguser = os.getenv("PGUSER")
    pgport = os.getenv("PGPORT")
    pgdatabase = os.getenv("PGDATABASE")
    pgpassword = os.getenv("PGPASSWORD")
        
    # Construct the path the for the artifact location on Azure Blob Storage
    artifact_location = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net?"

    # Construct the tracking URI for the postgresql database
    tracking_uri=f"postgresql://{pguser}:{pgpassword}@{pghost}:{pgport}/{pgdatabase}"

    # We instantiate the MLflow client for Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
else:
    # We constrauct the tracking URI for the Sqlite database
    tracking_uri = f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}"
    #tracking_uri = f"sqlite:///mlflow.db"
    artifact_location = os.path.join(os.getcwd(), "mlruns")

# We constrauct the tracking URI for the Sqlite database
tracking_uri = f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}"
#tracking_uri = "sqlite:///mlflow.db
#artifact_location = os.path.join(os.getcwd(), "mlruns")

# We collect the image urls for the labels and the dates
image_urls = get_image_urls_with_multiple_labels(labels, start_date, end_date, api_key, api_url)
# We create a dataframe with the image urls and the labels
df_sample_map = create_sample_map(image_urls)
# We download the images and save them in the media folder
image_dir = 'media'
df_sample_map = download_images(df_sample_map, image_dir)
# we save the dataset as a .csv file
df_sample_map.to_csv(os.path.join(os.getcwd(), "dataset_csv.csv"))
# We create the train and validation datasets for the given model
train_dataset, val_dataset = create_train_val_datasets(df_sample_map,
                              image_dir = image_dir,
                              model_name = model_name,
                              )

# We set the MLflow tracking URI
#mlflow.set_tracking_uri(tracking_uri)
print("Tracking URI:", mlflow.get_tracking_uri())

'''# Attempt to get the experiment by name
#existing_experiment = mlflow.get_experiment_by_name(experiment_name)'''

'''# We check if the experiment exists and create it if it doesn't
if existing_experiment is None:
    # If the experiment doesn't exist, create it
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_location,
        tags={"version": "v1", "priority": "P1"},
    )
    print(f"Experiment '{experiment_name}' created.")
else:
    # If the experiment exists, use the existing experiment
    experiment_id = existing_experiment.experiment_id
    print(f"Experiment '{experiment_name}' already exists. Using the existing experiment.")'''
    
temp_dir = os.path.join(os.getcwd(),'temporary_model_dir')

# We use a temporary directory for ModelCheckpoint
with tempfile.TemporaryDirectory() as temp_dir:
    checkpoint_filepath = f"{temp_dir}/best_model.keras"
   
# We define the ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,  # Temporary location
    monitor='val_loss',             # Metric to monitor
    save_best_only=True,            # Save only the best model
    save_weights_only=False,        # Save the entire model (architecture + weights)
    mode='min',                     # 'min' for loss
    verbose=1                       # Print saving information
)
   
# We set the number of epochs
number_of_epochs = 5

# Start a new MLflow run
#with mlflow.start_run(experiment_id=experiment_id) as run:
with mlflow.start_run() as run:
    print(f"Started run with ID: {run.info.run_id}")
    
    # Unable autologging for the model using the keras autolog to save the model using the .keras file format
    mlflow.keras.autolog()
    
    # We generate the trainable model
    model = compile_new_model(model_name)
    
    # Log other parameters    
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("labels", labels)
    mlflow.log_param("start_date", start_date)
    mlflow.log_param("end_date", end_date)
    # Log the dataset as artifact
    mlflow.log_artifact("dataset_csv.csv")
    
    # We train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=number_of_epochs,
        callbacks=[model_checkpoint])
    
# We end the MLflow run
mlflow.end_run()