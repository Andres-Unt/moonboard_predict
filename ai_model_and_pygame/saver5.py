import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("moonboard_model.keras")

# Replace this with your model's actual input shape, e.g. (None, 10) for sequence length 10, batch None
input_shape = model.inputs[0].shape  # safer way to get input shape if defined

# Make sure input_shape is concrete (no None in batch dim), e.g. (1, 10)
input_signature = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

