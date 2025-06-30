from keras.models import load_model

model = load_model("moonboard_model.keras")  # Assumes model is pure Sequential or Functional API
model.save("clean_model.h5")

