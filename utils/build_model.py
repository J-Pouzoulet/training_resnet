
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


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
    
    # We compile our new model with the Adam optimizer, categorical_crossentropy as loss function (because we use integer encoded classes) and accuracy as metrics
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"New {model_name} compiled successfully and is ready to be trained!")
    
    return model