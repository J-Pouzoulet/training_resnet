import requests
import tensorflow as tf
import pandas as pd
import os
import json

 
# Function to load the ResNet depencies dynamically
# The function takes in the model_name
# The function returns preprocess_input and decode_predictions functions 
def load_resnet_dependencies(model_name: str) -> tuple[callable]:
    # Check if the model_name is available in tf.keras.applications
    try:
        # Fetch the preprocess_input and decode_predictions functions
        if 'V2' not in model_name:
            preprocess_input = getattr(tf.keras.applications, 'resnet').preprocess_input
            decode_predictions = getattr(tf.keras.applications, 'resnet').decode_predictions
            print(f"Preprocess_input function for '{model_name}' loaded successfully.")
        else:
            preprocess_input = getattr(tf.keras.applications, 'resnet_v2').preprocess_input
            decode_predictions = getattr(tf.keras.applications, 'resnet_v2').decode_predictions
            print(f"Preprocess_input and decode_predictions function for '{model_name}' loaded successfully.")
        return preprocess_input, decode_predictions
    except AttributeError:
        raise ValueError(f"Model '{model_name}' is not available in tf.keras.applications.")
    

# Function to get the image urls from the API
# The function takes in the label, start_date, end_date, api_key and api_url as input
# The function returns a list of dictionaries containing the image urls, timestamp and label
def get_image_urls(label: str,
                   start_date: str,
                   end_date: str,
                   api_key: str,
                   api_url: str
                   ) -> list:
    
    # Define the URL for the GET request using formatted string
    url = f"{api_url}?tag={label}&start_date={start_date}&end_date={end_date}"
    
    # Chech if the label is either 'vine', 'grass' or 'ground'
    if label not in ['vine', 'grass', 'ground']:
        print("Unvalid Label used : {label} >>> Label should be either 'vine', 'grass' or 'ground")
        return []
    else:
        #check if the api_key is not None
        if api_key is not None:
            # Set the Authorization header with the API key
            headers = {
                "Authorization": f"Token {api_key}"
                }
                
            # Make the GET request with the token in the header
            api_response = requests.get(url, headers=headers)
                
            # Check if the GET request was successful
            if api_response.status_code == 200:
            # Process and print the API response
                response = api_response.json()
                print(f'Number of urls collected for {label}: {len(response)}')
            else:
                print(f"GET request failed with status code {api_response.status_code}")
        else:
            print("API key not found in the environment variables")
        
        # Return the response if successful
        if response is not None:
            # Add the label to the response
            for item in response:
                item['label'] = label 
            return response
        # If response is not successful then return None
        else:
            return []
        
        
# Function to get the image urls from the API with multiple labels
# The function takes in the labels (str or list), start_date, end_date, api_key and api_url as input
# The function returns a list of dictionaries containing the image urls, timestamp and label
def get_image_urls_with_multiple_labels(labels: tuple[str, list],
                                      start_date: str,
                                      end_date: str,
                                      api_key: str,
                                      api_url: str
                                      ) -> list:
    
    # Initialize the responses list
    responses = []
    
    # Check if the labels is a list
    if type(labels) == list:
        # If so iterate over the labels and get the image urls for each label and append to the responses list  
        for label in labels:
            responses += get_image_urls(label, start_date, end_date, api_key, api_url)
        return responses
    # If the labels is a string then get the image urls for the label and append to the responses list
    elif type(labels) == str:
        responses += get_image_urls(labels, start_date, end_date, api_key, api_url)
        return responses
    # If the labels is neither a list nor a string then return None
    else:
        print("Labels should be either a list of valid labels or a single label's string")
        return []
   
    
# Function that create a pd.Dataframe from the list of image urls (output from get_image_urls_with_multiple_labels)
# The function downloads the images in the media folder, creates and returns a pd.Dataframe for the mapping of filenames and labels 
def create_sample_map(image_urls: list) -> tuple[pd.DataFrame, None]:
    
    # We initiate a pd.DataFrame to store the image urls, image name, and their respective labels and timesstamp
    df_sample_map = pd.DataFrame(data = {'image_urls' : [], 'filename' : [], 'label' : [], 'timestamp' : []})
    
    # Iterate over the image urls
    for image_url in image_urls:
        # Get the image name from the url, we just keep the signature of the image otherwise the name will be too long
        image_name = image_url['media'].split('-')[-1]
        
        # We create a new record for the df_sample_map    
        df_new_record = pd.DataFrame(data = {'image_urls': image_url['media'], 
                                                'filename': f"{image_name}.jpg",
                                                'label': image_url['label'],
                                                'timestamp': image_url['timestamp']}, index=[0])
        
        # We add the new record to the df_sample_map
        df_sample_map = pd.concat([df_sample_map, df_new_record], ignore_index=True)    
            
    # We check if the sample_map is not empty else we return None       
    if df_sample_map.shape[0] > 0:
        print(f"Dataframe created successfully with shape : {df_sample_map.shape}")
        return df_sample_map
    else: 
        print("Something went wrong, the dataset in empty")
        return None 
    
    
# Function to upload the image to a local folder using the df_sample_map
# The function takes in the df_sample_map dataframe and the image directory as input
# The function returns the df_sample_map if the images are downloaded successfully else it returns None
def download_images(df_sample_map: pd.DataFrame, image_dir: str = "media") -> tuple[pd.DataFrame, None]:
    
    df_download_report = pd.DataFrame(data = {'image_urls' : [], 'status' : []})
    
    # Remove images from the media folder
    for file in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, file))
    
    # Iterate over the image urls and download the images
    for index, row in df_sample_map.iterrows():
        # We make a GET request to the image url
        response = requests.get(row['image_urls'])
        # Get the image name from the url, we just keep the signature of the image otherwise the name will be too long
        image_name = row['filename']
        # We check if the response is successful and save the image
        if response.status_code == 200:
            with open(os.path.join(image_dir, image_name), "wb") as file:
                file.write(response.content)
            df_new_record = pd.DataFrame(data = {'image_urls': row['image_urls'], 'status': 'Success'}, index=[0])
            df_download_report = pd.concat([df_download_report, df_new_record], ignore_index=True)
            #print(f"Image {image_name[:50]} downloaded successfully")
        else:
            df_new_record = pd.DataFrame(data = {'image_urls': row['image_urls'], 'status': 'Failed'}, index=[0])
            df_download_report = pd.concat([df_download_report, df_new_record], ignore_index=True)
            print(f"Failed to retrieve image {image_name[:50]}. Status code: {response.status_code}")
    
    df_sample_map['download_image_status'] = df_download_report['status'] 
    
    return df_sample_map


# The function create the path to the image in the media folder and preprocess the image
# At the same time it returns the encoded_labels
def create_path_list_and_encoded_label(df_sample_map: pd.DataFrame,
                                       image_dir: str
                                       ) -> tuple[list[str], list]:
    
    # We create a list of file paths
    image_paths = [os.path.join(image_dir, filename) for filename in df_sample_map['filename']]

    # We encode labels as integers using a mapping
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    df_sample_map['encoded_label'] = df_sample_map['label'].map(label_mapping)

    # We convert labels to one-hot encoding using TensorFlow's to_categorical function
    encoded_labels = tf.keras.utils.to_categorical(df_sample_map['encoded_label'], num_classes=3)

    return image_paths, encoded_labels


# The function is mean to be used on a TensorSliceDataset object using the .map() method
# The function takes in the image path and encoded label and returns the preprocessed image and encoded label
# The preprocessing depend upon the model used
def create_preprossesor(model_name: str) -> callable:
    preprocess_input, decode_predictions = load_resnet_dependencies(model_name)
    
    def load_and_preprocess_image(image_path: str, encoded_label: int) -> tuple[tf.Tensor, int]:
        # Read the image file
        image = tf.io.read_file(image_path)
        # Decode the image
        image = tf.image.decode_jpeg(image, channels=3) 
        # Resize to the target size
        image = tf.image.resize(image, (224,224)) # All six ResNet models use 224x224 input size
        # Preprocess for the ResNet model
        image = preprocess_input(image)
        #image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, encoded_label
      
    return load_and_preprocess_image


# Function to create the train and validation datasets
# The function takes in the df_sample_map, image_dir, model_name, val_split, batch_size and random_seed as input
def create_train_val_datasets(df_sample_map: pd.DataFrame,
                              image_dir: str,
                              model_name: str,
                              val_split: float = 0.2,
                              random_seed: int = 42,
                              batch_size: int = 5
                              ) -> tuple:
    
    image_paths, encoded_labels = create_path_list_and_encoded_label(df_sample_map, image_dir)
    
    # We create a dataset of image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, encoded_labels))
    
    # We shuffle the dataset to randomize the order of the data
    dataset = dataset.shuffle(buffer_size=len(df_sample_map), seed=random_seed)

    # We calculate the split index for training and validation datasets
    val_size = int(len(df_sample_map) * val_split)
    
    # We create train and validation datasets
    val_dataset = dataset.take(val_size)  # First `val_size` samples are used for validation
    train_dataset = dataset.skip(val_size)  # Remaining samples are used for training

    # We the load_and_preprocess_image function using the model_name so that proper dependencies are loaded for the preprocessing
    load_and_preprocess_image = create_preprossesor(model_name)
    
    # We use Batch and prefetch on both datasets for efficient training
    train_dataset = train_dataset.map(load_and_preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(load_and_preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

# Function to create the test datasets
# The function takes in the df_sample_map, image_dir, and the model_name (i.e the name of the resnet model) as input
# The function returns the test dataset that can be used for evaluation
def create_test_dataset(df_sample_map: pd.DataFrame,
                              image_dir: str,
                              model_name: str,
                              batch_size: int = 5
                              ) -> tuple:
    
    image_paths, encoded_labels = create_path_list_and_encoded_label(df_sample_map, image_dir)
    
    # We create a dataset of image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, encoded_labels))

    # We the load_and_preprocess_image function using the model_name so that proper dependencies are loaded for the preprocessing
    load_and_preprocess_image = create_preprossesor(model_name)
    
    # We use Batch and prefetch on both datasets for efficient training
    test_dataset = dataset.map(load_and_preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return test_dataset
