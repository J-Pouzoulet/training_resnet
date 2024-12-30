import requests
# from keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
#from keras.applications.resnet import preprocess_input, decode_predictions
#from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
import numpy as np
#from dotenv import load_dotenv
import pandas as pd
import os
import sys
import json


# Function to load the ResNet model and the preprocess_input function dynamically
# The function takes in the model_name, input_shape, include_top and weights as input
# The function returns the model and the preprocess_input function
def load_resnet_basemodel(model_name : str, 
                          input_shape : tuple[int, int, int] = (224, 224, 3), 
                          include_top : bool = False,
                          weights : str = 'imagenet'
                          ) -> tuple[tf.keras.Model, callable]:
    
    # Check if the model_name is available in tf.keras.applications
    try:
        # Dynamically fetch the model class
        model_class = getattr(tf.keras.applications, model_name)
        print(f"Model '{model_name}' found in tf.keras.applications.")
        # Instantiate the base_model
        base_model = model_class(weights=weights, include_top=include_top, input_shape=input_shape)
        print(f"Base_model '{model_name}' loaded successfully.")
        # Fetch the preprocess_input function
        return base_model
    except AttributeError:
        raise ValueError(f"Model '{model_name}' is not available in tf.keras.applications.")
    
    
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
    

# Function to get the image url from the API
# The function takes in the tag, start_date, end_date, api_key and api_url as input
# The function returns a list of dictionaries containing the image urls, timestamp and tag
def get_image_urls(tag: str,
                   start_date: str,
                   end_date: str,
                   api_key: str,
                   api_url: str
                   ) -> list:
    
    # Define the URL for the GET request using formatted string
    url = f"{api_url}?tag={tag}&start_date={start_date}&end_date={end_date}"
    
    # Chech if the tag is either 'vine', 'grass' or 'ground'
    if tag not in ['vine', 'grass', 'ground']:
        print("Unvalid Tag used : {tag} >>> Tag should be either 'vine', 'grass' or 'ground")
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
                print(f'Number of urls collected for {tag}: {len(response)}')
            else:
                print(f"GET request failed with status code {api_response.status_code}")
        else:
            print("API key not found in the environment variables")
        
        # Return the response if successful
        if response is not None:
            #add the tag to the response
            for i in response:
                i['tag'] = tag 
            return response
        # If response is not successful then return None
        else:
            return []
        
        
# Function to get the image urls from the API with multiple tags
# The function takes in the tags (str or list), start_date, end_date, api_key and api_url as input
# The function returns a list of dictionaries containing the image urls, timestamp and tag
def get_image_urls_with_multiple_tags(tags: tuple[str, list],
                                      start_date: str,
                                      end_date: str,
                                      api_key: str,
                                      api_url: str
                                      ) -> list:
    
    # Initialize the responses list
    responses = []
    
    # Check if the tags is a list
    if type(tags) == list:
        # If so iterate over the tags and get the image urls for each tag and append to the responses list  
        for tag in tags:
            responses += get_image_urls(tag, start_date, end_date, api_key, api_url)
        return responses
    # If the tags is a string then get the image urls for the tag and append to the responses list
    elif type(tags) == str:
        responses += get_image_urls(tags, start_date, end_date, api_key, api_url)
        return responses
    # If the tags is neither a list nor a string then return None
    else:
        print("Tags should be either a list of valid tags or a single tag's string")
        return []
    
    
# Function the upload the image from the list of image urls (output from get_image_urls_with_multiple_tags)
# The function downloads the images in the media folder, creates and returns a pd.Dataframe for the mapping of filenames and tags 
def download_images_and_create_sample_map(image_urls: list, image_dir: str = "media") -> tuple[pd.DataFrame, None]:
    
    # We initiate a pd.DataFrame to store the image urls, image name, and their respective tags and timesstamp
    df_sample_map = pd.DataFrame(data = {'image_urls' : [], 'filename' : [], 'tag' : [], 'timestamp' : []})
    
    # Remove images from the media folder
    for file in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, file))
    
    # Initiate a dictionary to store the image urls and their respective tags
    sample_map = {}
    
    # Iterate over the image urls and download the images
    for i, image_url in enumerate(image_urls):
        response = requests.get(image_url['media'])
        # Get the image name from the url, we just keep the signature of the image otherwise the name will be too long
        image_name = image_url['media'].split('-')[-1]
        # We check if the response is successful and save the image in the media folder and add the image name and tag to the dictionary
        if response.status_code == 200:
            with open(f"{image_dir}/{image_name}.jpg", "wb") as file:
                file.write(response.content)
            # We create a new record for the df_sample_map    
            df_new_record = pd.DataFrame(data = {'image_urls': image_url['media'], 
                                                 'filename': f"{image_name}.jpg",
                                                 'tag': image_url['tag'],
                                                 'timestamp': image_url['timestamp']}, index=[0])
            # We add the new record to the df_sample_map
            df_sample_map = pd.concat([df_sample_map, df_new_record], ignore_index=True)    
            print(f"Image {image_name[:50]} downloaded successfully")
        else:
            print(f"Failed to retrieve image {image_name[:50]}. Status code: {response.status_code}")
            
    # We check if the sample_map is not empty else we return None       
    if df_sample_map.shape[0] > 0:
        print(f"Dataframe created successfully with shape : {df_sample_map.shape}")
        return df_sample_map
    else: 
        print("No images downloaded")
        return None








# Function that create the path list and encoded labels for the images
# The function takes in the df_sample_map dataframe and the image directory as input
# The function returns a tuple of the image paths and encoded labels
def create_path_list_and_encoded_label(df_sample_map: tuple[pd.DataFrame],
                                       image_dir: str
                                       ) -> tuple[list[str], list[int]]:
    
    # Create a list of file paths and labels
    image_paths = [os.path.join(image_dir, filename) for filename in df_sample_map['filename']]
    # Encode labels as integers
    label_mapping = {'vine':0, 'grass':1, 'ground':2}
    #label_mapping = {label: idx for idx, label in enumerate(labels.unique())}
    df_sample_map['encoded_label'] = df_sample_map['label'].map(label_mapping)
    encoded_labels = df_sample_map['encoded_label'].values
    
    return image_paths, encoded_labels


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
                              random_seed: int = 42
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
    train_dataset = train_dataset.map(load_and_preprocess_image).batch(5).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(load_and_preprocess_image).batch(5).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


# The function is mean to compile the new model using the base_model and the custom layers respective to the model_name
# Here the number of neurons in the last Dense layer is 3 because we have 3 categories and through is hard coded so that the model can be trained on 1, 2 or 3 categories   
def compile_new_model(model_name: str,
                      allowed_all_layers_to_be_trained: bool = True
                      ) -> tf.keras.Model:
    
    # We load the base_model from keras repective to the model_name
    base_model = load_resnet_basemodel(model_name, input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    
    # We set the base_model to be trainable or not
    base_model.trainable = allowed_all_layers_to_be_trained
    
    # We add custom layers on top
    # First, we create a variable which is output of the base_model
    x = base_model.output
    # According to restnet architecture we perform a GlobalAveragePooling2D
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Then we add a Dense layer (2048 neurons, ReLu activation function)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    # Then we add a final Dense layer (3 neurones because we have 3 categories, and a Softmax activation function to compute overall class probabilities)
    # The output is the predictions 
    predictions = tf.keras.layers.Dense(3, activation='softmax')(x)  
    
    # We define the new model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # We specify the learning rate, we could use a learning rate scheduler instead
    learning_rate = 0.001
    # We create Adam optimizer with the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    # We compile our new model with the Adam optimizer, sparse_categorical_crossentropy as loss function (because we use integer encoded classes) and accuracy as metrics
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"New {model_name} compiled successfully and is ready to be trained!")
    
    return model