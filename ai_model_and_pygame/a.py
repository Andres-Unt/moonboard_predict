import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keras_tuner as kt
from itertools import combinations

# 1. Load data
with open('moonboard_data.json','r') as f:
    raw = json.load(f)['data']

# 2. Grade mapping
gfont = ['6A+','6B','6B+','6C','6C+',
         '7A','7A+','7B','7B+','7C',
         '7C+', '8A','8A+','8B','8B+']
grade_to_int = {g:i for i,g in enumerate(gfont)}
int_to_grade = {i:g for g,i in grade_to_int.items()}

# 3. Clean & filter
filtered=[]; discard={'method':0,'ug_null':0,'ug_invalid':0,'reps_low':0,'grade_mismatch':0,'start_invalid':0,'no_start_or_end':0}
for e in raw:
    if e.get('method')!='Feet follow hands': discard['method']+=1; continue
    ug=e.get('userGrade');
    if ug is None: discard['ug_null']+=1; continue
    ug=ug.upper()
    if ug not in grade_to_int: discard['ug_invalid']+=1; continue
    if e.get('grade') is None or e['grade'].upper()!=ug: discard['grade_mismatch']+=1; continue
    if e.get('repeats',0)<10: discard['reps_low']+=1; continue
    has_s,has_e,valid_s=False,False,True
    for m in e['moves']:
        if m.get('isStart'):
            has_s=True;
            d=m['description']
            if not (d[0] in 'ABCDEFGHIJK' and 1<=int(d[1:])<=6): valid_s=False
        if m.get('isEnd'): has_e=True
    if not has_s or not has_e: discard['no_start_or_end']+=1; continue
    if not valid_s: discard['start_invalid']+=1; continue
    filtered.append(e)
print(f"Kept {len(filtered)}/{len(raw)} routes. Discards: {discard}")

# 4. Encode holds
dims_main=[f"{chr(65+c)}{r}" for c in range(11) for r in range(1,19)]
dims_start=[f"{chr(65+c)}{r}" for c in range(11) for r in range(1,7)]
dims_end=[f"{chr(65+c)}18" for c in range(11)]
pos2i_main={p:i for i,p in enumerate(dims_main)}
pos2i_start={p:i for i,p in enumerate(dims_start)}
pos2i_end={p:i for i,p in enumerate(dims_end)}
N=len(filtered)
X_main=np.zeros((N,len(dims_main)),np.float32)
X_start=np.zeros((N,len(dims_start)),np.float32)
X_end=np.zeros((N,len(dims_end)),np.float32)
y=np.zeros((N,),np.float32)
for i,e in enumerate(filtered):
    for m in e['moves']:
        d=m['description']
        if d in pos2i_main: X_main[i,pos2i_main[d]]=1
        if m.get('isStart') and d in pos2i_start: X_start[i,pos2i_start[d]]=1
        if m.get('isEnd') and d in pos2i_end: X_end[i,pos2i_end[d]]=1
    y[i]=grade_to_int[e['userGrade'].upper()]
X_all=np.concatenate([X_main,X_start,X_end],axis=1)

# 5. Split
idx=np.random.permutation(N)
s=int(0.8*N)
train_idx,val_idx=idx[:s],idx[s:]
X_tr,X_val=X_all[train_idx],X_all[val_idx]
y_tr,y_val=y[train_idx],y[val_idx]

# 5b. Augment training data with flags
# flag: 0=orig, -1=removal, +1=addition
aug_X=[]; aug_y=[]; aug_f=[]
cnt_prev = len(X_tr)
cnt_rem1 = cnt_rem2 = cnt_rem3 = cnt_add = 0
for xi,yi in zip(X_tr,y_tr):
    main=xi[:len(dims_main)]
    st=xi[len(dims_main):len(dims_main)+len(dims_start)]
    en=xi[-len(dims_end):]
    start_idxs=[pos2i_main[p] for p,i in pos2i_start.items() if st[i]]
    end_idxs=[pos2i_main[p] for p,i in pos2i_end.items() if en[i]]
    nonse=[i for i,v in enumerate(main) if v and i not in start_idxs+end_idxs]
    for k in nonse:
        x2=xi.copy(); x2[k]=0; aug_X.append(x2); aug_y.append(yi); aug_f.append(-1)
        cnt_rem1 += 1
    for a,b in combinations(nonse,2):
        x2=xi.copy(); x2[a]=x2[b]=0; aug_X.append(x2); aug_y.append(yi); aug_f.append(-1)
        cnt_rem2 += 1
    for a,b,c in combinations(nonse,3):
        x2=xi.copy(); x2[a]=x2[b]=x2[c]=0; aug_X.append(x2); aug_y.append(yi); aug_f.append(-1)
        cnt_rem3 += 1
    absent=[i for i,v in enumerate(main) if not v]
    for k in absent:
        if np.random.random() < 0.216:# random chance to skip addition
            x2=xi.copy(); x2[k]=1; aug_X.append(x2); aug_y.append(yi); aug_f.append(1)
            cnt_add += 1
# original samples
orig_X=list(X_tr); orig_y=list(y_tr); orig_f=[0]*len(y_tr)
# combine
X_tr_aug=np.array(orig_X+aug_X,dtype=np.float32)
y_tr_aug=np.vstack([orig_y+aug_y, orig_f+aug_f]).T
# print augmentation stats
cnt_rem = cnt_rem1 + cnt_rem2 + cnt_rem3
print(f"Augmentation stats: removals of 1 hold: {cnt_rem1}, 2 holds: {cnt_rem2}, 3 holds: {cnt_rem3}, additions: {cnt_add}")
print(f"Total removals: {cnt_rem}, additions: {cnt_add}")
print(f"Augmented: {len(aug_X)} samples, train now {X_tr_aug.shape[0]}")

# build train/val tensors
X_train=X_tr_aug; y_train=y_tr_aug
# for validation, flag=0
y_val_aug=np.vstack([y_val, np.zeros_like(y_val)]).T

# 6. Custom one-sided loss
def one_sided_loss(y_true,y_pred):
    grade=y_true[:,0]; flag=y_true[:,1]
    err=y_pred[:,0]-grade
    loss_rem=tf.square(tf.maximum(err,0.))
    loss_add=tf.square(tf.maximum(-err,0.))
    loss_orig=tf.square(err)
    loss=tf.where(flag<0, loss_rem, tf.where(flag>0, loss_add, loss_orig))
    return tf.reduce_mean(loss)

# 7. Hyperparameter tuning
def build_model():
    m=models.Sequential()
    m.add(layers.Input(shape=(X_train.shape[1],)))
    m.add(layers.GaussianNoise(0.05))
    for i in range(5):
        m.add(layers.Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(3e-6)))
        m.add(layers.Dropout(0.3))
    m.add(layers.Dense(15))
    m.add(layers.Dense(1))
    m.compile('adam',loss=one_sided_loss)
    return m

stop=callbacks.EarlyStopping(monitor='val_loss',patience=5)

# 8. Train final
# best=tuner.get_best_hyperparameters(1)[0]
model=build_model()
hist=model.fit(X_train,y_train,validation_data=(X_val,y_val_aug),epochs=10000,batch_size=128,callbacks=[callbacks.EarlyStopping('val_loss',patience=100,restore_best_weights=True)],verbose=2)
model.save('moonboard_model.keras')

# 9. Evaluate
preds=model.predict(X_val).flatten()
# use linear error for reporting
idx=np.clip(np.rint(preds),0,len(gfont)-1).astype(int)
mse=mean_squared_error(y_val, preds)
print(f"MSE: {mse:.3f}")
print(f"Exact: {np.mean(idx==y_val)*100:.1f}% Off1: {np.mean(np.abs(idx-y_val)<=1)*100:.1f}%")
print(f"off2: {np.mean(np.abs(idx-y_val)<=2)*100:.1f}% Off3: {np.mean(np.abs(idx-y_val)<=3)*100:.1f}%")

# 10. Prediction helper
def predict_route(moves):
    vec=np.zeros((1,X_train.shape[1]),dtype=np.float32)
    for m in moves:
        if m in pos2i_main: vec[0,pos2i_main[m]]=1
        if m in pos2i_start: vec[0,len(dims_main)+pos2i_start[m]]=1
        if m in pos2i_end: vec[0,len(dims_main)+len(dims_start)+pos2i_end[m]]=1
    p=model.predict(vec)[0,0]
    return p,int_to_grade[int(np.clip(round(p),0,len(gfont)-1))]
