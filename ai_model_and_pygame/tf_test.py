import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build

print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("CUDA version from build info:", tf_build.build_info.get("cuda_version", "N/A"))
print("cuDNN version from build info:", tf_build.build_info.get("cudnn_version", "N/A"))
