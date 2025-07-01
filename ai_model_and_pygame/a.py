import json
import numpy as np
import random
from itertools import combinations
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# --- board dims & mappings ---
dims_main  = [f"{chr(65+c)}{r}" for c in range(11) for r in range(1,19)]
dims_start = [f"{chr(65+c)}{r}" for c in range(11) for r in range(1,7)]
dims_end   = [f"{chr(65+c)}18"     for c in range(11)]
pos2i_main  = {p:i for i,p in enumerate(dims_main)}
pos2i_start = {p:i for i,p in enumerate(dims_start)}
pos2i_end   = {p:i for i,p in enumerate(dims_end)}

# --- augmentation caps & weights ---
caps = {
    'rem1': 50_000,
    'rem2': 50_000,
    'rem3': 30_000,
    'rem4': 20_000,
    'add' : 100_000
}
weights_map = {
    'orig': 1.0,
    'rem1': 0.7,
    'rem2': 0.5,
    'rem3': 0.3,
    'rem4': 0.1,
    'add' : 0.4
}

def load_and_preprocess(path='moonboard_data.json'):
    # load & filter
    with open(path,'r') as f:
        raw = json.load(f)['data']
    gfont = ['6A+','6B','6B+','6C','6C+',
             '7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+']
    grade_to_int = {g:i for i,g in enumerate(gfont)}
    int_to_grade = {i:g for g,i in grade_to_int.items()}
    filtered = []
    for e in raw:
        if e.get('method')!='Feet follow hands': continue
        ug = e.get('userGrade'); 
        if not ug: continue
        ug = ug.upper()
        if ug not in grade_to_int: continue
        if e.get('grade') is None or e['grade'].upper()!=ug: continue
        if e.get('repeats',0)<10: continue
        has_s = has_e = False; valid_s=True
        for m in e['moves']:
            if m.get('isStart'):
                has_s=True
                d=m['description']
                if not(d[0] in 'ABCDEFGHIJK' and 1<=int(d[1:])<=6):
                    valid_s=False
            if m.get('isEnd'): has_e=True
        if has_s and has_e and valid_s:
            filtered.append(e)

    N = len(filtered)
    Xm = np.zeros((N,len(dims_main)),  np.float32)
    Xs = np.zeros((N,len(dims_start)), np.float32)
    Xe = np.zeros((N,len(dims_end)),   np.float32)
    y  = np.zeros((N,),               np.float32)
    for i,e in enumerate(filtered):
        for m in e['moves']:
            d = m['description']
            if d in pos2i_main:  Xm[i,pos2i_main[d]]=1
            if m.get('isStart') and d in pos2i_start:
                Xs[i,pos2i_start[d]] = 1
            if m.get('isEnd')   and d in pos2i_end:
                Xe[i,pos2i_end[d]]     = 1
        y[i] = grade_to_int[e['userGrade'].upper()]

    X_all = np.concatenate([Xm,Xs,Xe],axis=1)
    # train/val split
    idx = np.random.permutation(N)
    s   = int(0.8*N)
    tr, va = idx[:s], idx[s:]
    X_tr, y_tr = X_all[tr], y[tr]
    X_va, y_va = X_all[va], y[va]

    # augmentation with caps
    aug_X, aug_y, aug_type = [], [], []
    counters = {k:0 for k in caps}
    Lm, Ls, Le = len(dims_main), len(dims_start), len(dims_end)

    print("Building capped augmentations...")
    for xi, yi in zip(X_tr, y_tr):
        main = xi[:Lm]
        st   = xi[Lm:Lm+Ls]
        en   = xi[-Le:]
        start_idxs = [pos2i_main[p] for p,i in pos2i_start.items() if st[i]]
        end_idxs   = [pos2i_main[p] for p,i in pos2i_end.items()   if en[i]]

        nonse = [i for i,v in enumerate(main) if v and i not in start_idxs+end_idxs]
        # removals k=1..4
        for k,cap_key in zip([1,2,3,4],['rem1','rem2','rem3','rem4']):
            if counters[cap_key] >= caps[cap_key]: 
                continue
            for combo in combinations(nonse, k):
                if counters[cap_key] >= caps[cap_key]:
                    break
                x2 = xi.copy()
                for j in combo: x2[j]=0
                aug_X.append(x2); aug_y.append(yi); aug_type.append(cap_key)
                counters[cap_key] += 1
            # move on to next sample

        # additions
        if counters['add'] < caps['add']:
            absent = [i for i,v in enumerate(main) if not v]
            random.shuffle(absent)
            for j in absent:
                if counters['add'] >= caps['add']: break
                if random.random() < 0.34:
                    x2 = xi.copy(); x2[j]=1
                    aug_X.append(x2); aug_y.append(yi); aug_type.append('add')
                    counters['add'] += 1

    print("Augmentation counts:", counters)
    print(f"Train before: {len(X_tr)}, after aug: {len(X_tr)+len(aug_X)}")

    # assemble final train + weights
    X_train = np.vstack([X_tr, np.array(aug_X, dtype=np.float32)])
    y_train = np.hstack([y_tr, np.array(aug_y, dtype=np.float32)])
    weights = np.hstack([
        np.full(len(y_tr), weights_map['orig'], dtype=np.float32),
        np.array([weights_map[t] for t in aug_type], dtype=np.float32)
    ])

    return X_train, y_train, weights, X_va, y_va, int_to_grade

def train():
    X_train, y_train, weights, X_val, y_val, int_to_grade = load_and_preprocess()

    print(f"\nTraining on {X_train.shape[0]} samples ({len(weights)} weights) "
          f"with {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")

    # DMatrix with weights
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    params = {
        'objective':       'reg:squarederror',
        'max_depth':       6,
        'eta':             0.03,
        'subsample':       0.8,
        'colsample_bytree':0.8,
        'reg_alpha':       1.0,
        'reg_lambda':      1.0,
        'verbosity':       1
    }

    print("\nStarting training with early stopping...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=800,
        evals=[(dtrain,'train'), (dval,'eval')],
        early_stopping_rounds=30,
        verbose_eval=10
    )

    bst.save_model('moonboard_model.xgb')

    # final evaluation
    print("\nEvaluating on validation set...")
    preds = bst.predict(dval)
    mse   = mean_squared_error(y_val, preds)
    idx   = np.clip(np.rint(preds).astype(int), 0, len(int_to_grade)-1)

    print(f"MSE: {mse:.3f}")
    print(f"Exact: {np.mean(idx==y_val)*100:.1f}%")
    for d in [1,2,3]:
        print(f"Off{d}: {np.mean(np.abs(idx-y_val)<=d)*100:.1f}%")

def predict_route(moves, bst, input_dim, int_to_grade):
    vec = np.zeros((1, input_dim), dtype=np.float32)
    for m in moves:
        if m in pos2i_main:
            vec[0,pos2i_main[m]] = 1
        if m in pos2i_start:
            vec[0,len(dims_main)+pos2i_start[m]] = 1
        if m in pos2i_end:
            vec[0,len(dims_main)+len(dims_start)+pos2i_end[m]] = 1
    dm = xgb.DMatrix(vec)
    p = bst.predict(dm)[0]
    g = int(np.clip(round(p), 0, len(int_to_grade)-1))
    return p, int_to_grade[g]

if __name__ == '__main__':
    train()
