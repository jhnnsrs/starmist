import tensorflow as tf


# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs are available.")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("No GPUs found.")
    # Optionally, you can set the environment variable to force TensorFlow to use CPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Forcing TensorFlow to use CPU.")