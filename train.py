#python train.py --model_name ResNet101 --labels vine grass --start_date 2021-03-01 --end_date 2021-08-01 --experiment_name experiment_test 

#import requests
#from keras.applications import ResNet50
#from keras.applications.resnet import decode_predictions, preprocess_input
#from keras.preprocessing.image import load_img, img_to_array
#import numpy as np
import os
from dotenv import load_dotenv
from utils.build_dataset import *
from utils.build_model import *
import argparse
from datetime import datetime
import mlflow

# We set up argument parser so that they can be passed from the command line
parser = argparse.ArgumentParser(description="Train a ResNet model on a dataset of images given the classes, the start_date and the end_date.")
parser.add_argument("--model_name", type=str, default="ResNet50", help="Accept ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2.")
parser.add_argument("--labels", type=str, nargs="+", default="vine", help="Accept 'vine', 'grass', 'ground' or combination of them. ex: --labels vine grass")
parser.add_argument("--start_date", type=str, default="2021-05-27", help="The start date of the images to be used for training in YYYY-mm-DD format.")
parser.add_argument("--end_date", type=str, default="2021-06-01", help="The start date of the images to be used for training in YYYY-mm-DD format.")
parser.add_argument("--experiment_name", type=str, default=None, help="The name of the experiement to log the results to in MLflow.")

# We parse the arguments
args = parser.parse_args()

# We assign the arguments to variables
model_name = args.model_name
labels = args.labels
start_date = args.start_date
end_date = args.end_date
experiment_name = args.experiment_name

print(f"model: {model_name}")
print(f"labels: {labels}")
print(f"start_date: {start_date}")
print(f"end_date: {end_date}")
print(f"experiment_name: {experiment_name}")

# We load the environment variables from the secret.env file
load_dotenv()

# We access environment variables using os.getenv()
api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

# We compile the model
model = compile_new_model(model_name)

# We make the train and validation datasets
image_urls = get_image_urls_with_multiple_labels(labels, start_date, end_date, api_key, api_url)
# Download images and create a sample map
df_sample_map = create_sample_map(image_urls)
download_images(df_sample_map: pd.DatFrame, image_dir: str = "media")

# Convert DataFrame to JSON
json_data = df_sample_map.to_json(orient="records")

# We check if an experiment name is provided, if so we log the data to MLflow
if experiment_name is not None:
    
    # We set the tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # We set the experiment name
    mlflow.set_experiment(experiment_name)
    
    # Save JSON data to a file
    with open("dataset.json", "w") as f:
        f.write(json_data)
    
    # We set the run name
    run_name = f"{model_name}_"+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    mlflow.tensorflow.autolog()
    
    # Start a new MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters    
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("labels", labels)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        # Log the dataset as artifact
        mlflow.log_artifact("dataset.json")
        # For the other parameters and artifact we use the mlflow.tensorflow.autolog (but of course this could be customized if needed)

train_dataset, val_dataset = create_train_val_datasets(df_sample_map,
                              image_dir = 'media',
                              model_name = model_name,
                              )

# We train the model
number_of_epochs = 3
history = model.fit(train_dataset, validation_data=val_dataset, epochs=number_of_epochs)

# We end the MLflow run
mlflow.end_run()