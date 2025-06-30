import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keras_tuner as kt

# 1. Load data
with open('moonboard_data.json', 'r') as f:
    raw = json.load(f)['data']

gfont = [
    '6A+','6B','6B+','6C','6C+',
    '7A','7A+','7B','7B+','7C',
    '7C+','8A','8A+','8B','8B+',
]
grade_to_int = {g: i for i, g in enumerate(gfont)}
int_to_grade = {i: g for g, i in grade_to_int.items()}

# 3. Clean & filter
filtered = []
discard = {'method':0, 'ug_null':0, 'ug_invalid':0, 'reps_low':0, 'grade_mismatch':0, 'start_invalid':0, 'no_start_or_end':0}
for e in raw:
    if e.get('method') != 'Feet follow hands':
        discard['method'] += 1; continue
    ug = e.get('userGrade')
    if ug is None:
        discard['ug_null'] += 1; continue
    ug = ug.upper()
    if ug not in grade_to_int:
        discard['ug_invalid'] += 1; continue
    g = e.get('grade')
    if g is None or g.upper() != ug:
        discard['grade_mismatch'] += 1; continue
    if e.get('repeats', 0) < 10:
        discard['reps_low'] += 1; continue
    has_start, has_end, valid_start = False, False, True
    for m in e['moves']:
        if m.get('isStart'):
            has_start = True
            desc = m.get('description')
            if desc and not (desc[0] in 'ABCDEFGHIJK' and 1 <= int(desc[1:]) <= 6):
                valid_start = False
        if m.get('isEnd'):
            has_end = True
    if not has_start or not has_end:
        discard['no_start_or_end'] += 1; continue
    if not valid_start:
        discard['start_invalid'] += 1; continue
    filtered.append(e)

print(f"Total entries: {len(raw)}")
print(f"After filtering: {len(filtered)}")
print("Discard counts:")
for k, v in discard.items():
    print(f"  {k}: {v}")

# 4. Encode holds
main_dims = [f"{chr(ord('A')+c)}{r}" for c in range(11) for r in range(1,19)]
start_dims = [f"{chr(ord('A')+c)}{r}" for c in range(11) for r in range(1,7)]
end_dims = [f"{chr(ord('A')+c)}18" for c in range(13)]
pos2i_main = {p: i for i, p in enumerate(main_dims)}
pos2i_start = {p: i for i, p in enumerate(start_dims)}
pos2i_end = {p: i for i, p in enumerate(end_dims)}

X_main = np.zeros((len(filtered), len(main_dims)), dtype=np.float32)
X_start = np.zeros((len(filtered), len(start_dims)), dtype=np.float32)
X_end = np.zeros((len(filtered), len(end_dims)), dtype=np.float32)
y = np.zeros(len(filtered), dtype=np.float32)

for i, e in enumerate(filtered):
    for m in e['moves']:
        desc = m.get('description')
        if desc in pos2i_main: X_main[i, pos2i_main[desc]] = 1
        if m.get('isStart') and desc in pos2i_start:
            X_start[i, pos2i_start[desc]] = 1
        if m.get('isEnd') and desc in pos2i_end:
            X_end[i, pos2i_end[desc]] = 1
    y[i] = grade_to_int[e['userGrade'].upper()]

X = np.concatenate([X_main, X_start, X_end], axis=1)

# 5. Shuffle & split
idx = np.arange(len(filtered))
np.random.shuffle(idx)
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
input_dim = X.shape[1]

# 6. Hyperparameter tuning setup
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    noise_level = hp.Choice('gaussian_noise', [0.01, 0.05, 0.1])
    if noise_level > 0:
        model.add(layers.GaussianNoise(noise_level))
    for _ in range(hp.Choice('num_layers', [4])):
        model.add(layers.Dense(
            hp.Choice('units', [64, 96]),
            activation='relu',
            kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-6, 1e-5, sampling='log'))
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='mse'
    )
    return model

max_epochs = 200
# 7. Run tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=max_epochs,
    factor=3,
    directory='moonboard_tuning',
    project_name='grade_predictor'
)
batch_size = 256
cb = [callbacks.EarlyStopping(monitor='val_loss', patience=5)]
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=cb, batch_size=batch_size)

# 8. Train best model
best_hp = tuner.get_best_hyperparameters(1)[0]
model = build_model(best_hp)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=max_epochs,
    batch_size=batch_size,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=1e-6)
    ],
    verbose=2
)
model.save("moonboard_model.keras")

# 9. Plot
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

# 10. Evaluate
preds = model.predict(X_val).flatten()
mse = mean_squared_error(y_val, preds)
p_int = np.clip(np.rint(preds), 0, len(gfont)-1).astype(int)
print(f"Validation MSE: {mse:.4f}")
print(f"Exact match: {np.mean(p_int==y_val)*100:.2f}%")
print(f"Off-by-1: {np.mean(np.abs(p_int-y_val)<=1)*100:.2f}%")
print(f"Off-by-2: {np.mean(np.abs(p_int-y_val)<=2)*100:.2f}%")
print(f"Off-by-3: {np.mean(np.abs(p_int-y_val)<=3)*100:.2f}%")

# 11. Prediction helper
def predict_route(moves):
    vec_main = np.zeros((1, len(main_dims)), dtype=np.float32)
    vec_start = np.zeros((1, len(start_dims)), dtype=np.float32)
    vec_end = np.zeros((1, len(end_dims)), dtype=np.float32)
    for m in moves:
        if m in pos2i_main:
            vec_main[0, pos2i_main[m]] = 1
        if m in pos2i_start:
            vec_start[0, pos2i_start[m]] = 1
        if m in pos2i_end:
            vec_end[0, pos2i_end[m]] = 1
    vec = np.concatenate([vec_main, vec_start, vec_end], axis=1)
    p = model.predict(vec)[0,0]
    idx = int(np.clip(round(p), 0, len(gfont)-1))
    return p, int_to_grade[idx]
