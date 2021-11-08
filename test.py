import tensorflow as tf
import os
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(trained_model/model)# path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)