# save_model_h5.py
from keras.models import load_model

model = load_model("moonboard_model.keras")
model.save("moonboard_model.h5")

