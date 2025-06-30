import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("moonboard_model.keras")

spec = (tf.TensorSpec(model.input.shape, tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

