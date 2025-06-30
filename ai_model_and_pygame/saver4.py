import keras2onnx
import onnx
import tensorflow as tf

model = tf.keras.models.load_model("moonboard_model.keras")
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, "model.onnx")

