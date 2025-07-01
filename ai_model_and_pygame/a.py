import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import random
from itertools import combinations
print(torch.cuda.is_available(), torch.__version__)
device = torch.device('cuda')
dims_main = [f"{chr(65+c)}{r}" for c in range(11) for r in range(1,19)]
dims_start = [f"{chr(65+c)}{r}" for c in range(11) for r in range(1,7)]
dims_end = [f"{chr(65+c)}18" for c in range(11)]
pos2i_main = {p:i for i,p in enumerate(dims_main)}
pos2i_start = {p:i for i,p in enumerate(dims_start)}
pos2i_end = {p:i for i,p in enumerate(dims_end)}

def load_and_preprocess(path='moonboard_data.json'):
    # Load raw JSON
    with open(path, 'r') as f:
        raw = json.load(f)['data']

    # Grade mappings
    gfont = ['6A+','6B','6B+','6C','6C+',
             '7A','7A+','7B','7B+','7C',
             '7C+', '8A','8A+','8B','8B+']
    grade_to_int = {g:i for i,g in enumerate(gfont)}
    int_to_grade = {i:g for g,i in grade_to_int.items()}

    # Filter
    filtered = []
    for e in raw:
        if e.get('method')!='Feet follow hands': continue
        ug = e.get('userGrade')
        if ug is None: continue
        ug = ug.upper()
        if ug not in grade_to_int: continue
        if e.get('grade') is None or e['grade'].upper()!=ug: continue
        if e.get('repeats',0) < 10: continue
        has_s = has_e = False
        valid_s = True
        for m in e['moves']:
            if m.get('isStart'):
                has_s = True
                d = m['description']
                if not (d[0] in 'ABCDEFGHIJK' and 1<=int(d[1:])<=6): valid_s = False
            if m.get('isEnd'): has_e=True
        if not (has_s and has_e and valid_s): continue
        filtered.append(e)

    # Encode holds

    N = len(filtered)
    X_main = np.zeros((N,len(dims_main)),np.float32)
    X_start = np.zeros((N,len(dims_start)),np.float32)
    X_end = np.zeros((N,len(dims_end)),np.float32)
    y = np.zeros((N,),np.float32)

    for i,e in enumerate(filtered):
        for m in e['moves']:
            d = m['description']
            if d in pos2i_main: X_main[i,pos2i_main[d]] = 1
            if m.get('isStart') and d in pos2i_start: X_start[i,pos2i_start[d]] = 1
            if m.get('isEnd') and d in pos2i_end: X_end[i,pos2i_end[d]] = 1
        y[i] = grade_to_int[e['userGrade'].upper()]

    X_all = np.concatenate([X_main, X_start, X_end], axis=1)

    # Split train/val
    idx = np.random.permutation(N)
    s = int(0.8 * N)
    train_idx, val_idx = idx[:s], idx[s:]
    # X_tr, X_val = X_all[train_idx], X_all[val_idx]
    X_tr, X_val = X_main[train_idx], X_main[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Augmentation with counters
    aug_X, aug_y, aug_f = [], [], []
    cnt_rem1 = cnt_rem2 = cnt_rem3 = cnt_add = 0
    L_main = len(dims_main)
    L_start = len(dims_start)
    L_end = len(dims_end)
    # for xi, yi in zip(X_tr, y_tr):
    #     main = xi[:L_main]
    #     st = xi[L_main:L_main+L_start]
    #     en = xi[-L_end:]
    #     start_idxs = [pos2i_main[p] for p,i in pos2i_start.items() if st[i]]
    #     end_idxs = [pos2i_main[p] for p,i in pos2i_end.items() if en[i]]
    #     nonse = [i for i,v in enumerate(main) if v and i not in start_idxs+end_idxs]
    #     for k in nonse:
    #         x2 = xi.copy(); x2[k] = 0
    #         aug_X.append(x2); aug_y.append(yi); aug_f.append(1)
    #         cnt_rem1 += 1
    #     for a,b in combinations(nonse,2):
    #         x2 = xi.copy(); x2[a] = x2[b] = 0
    #         aug_X.append(x2); aug_y.append(yi); aug_f.append(1)
    #         cnt_rem2 += 1
    #     for a,b,c in combinations(nonse,3):
    #         x2 = xi.copy(); x2[a] = x2[b] = x2[c] = 0
    #         aug_X.append(x2); aug_y.append(yi); aug_f.append(1)
    #         cnt_rem3 += 1
    #     absent = [i for i,v in enumerate(main) if not v]
    #     for k in absent:
    #         if random.random() < 0.216:
    #             x2 = xi.copy(); x2[k] = 1
    #             aug_X.append(x2); aug_y.append(yi); aug_f.append(-1)
    #             cnt_add += 1
    cnt_rem = cnt_rem1 + cnt_rem2 + cnt_rem3
    print(f"Augmentation stats: removals of 1 hold: {cnt_rem1}, 2 holds: {cnt_rem2}, 3 holds: {cnt_rem3}, additions: {cnt_add}")
    print(f"Total removals: {cnt_rem}, additions: {cnt_add}")
    print(f"Augmented: {len(aug_X)} samples, train before {len(X_tr)}, after {len(X_tr)+len(aug_X)}")

    # original
    orig_X = list(X_tr)
    orig_y = list(y_tr)
    orig_f = [0]*len(y_tr)
    # combine
    X_train = np.array(orig_X + aug_X, dtype=np.float32)
    y_train = np.vstack([orig_y + aug_y, orig_f + aug_f]).T.astype(np.float32)
    y_val_aug = np.vstack([y_val, np.zeros_like(y_val)]).T.astype(np.float32)

    return X_train, y_train, X_val, y_val_aug, y_val, int_to_grade

class MoonboardDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = sigma
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class MoonModel(nn.Module):
    """
    2D-CNN according to the paper:
    Input: (batch, features) reshaped to (batch, 1, 18, 11)
    Conv2D(32, kernel=3) -> ReLU -> BN
    Conv2D(32, kernel=3) -> ReLU -> BN
    Conv2D(64, kernel=3) -> ReLU -> BN
    Conv2D(64, kernel=3) -> ReLU -> BN
    Flatten -> Dense(32, relu) -> Dense(1, linear)
    """
    def __init__(self, height=18, width=11):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        flat_dim = 64 * height * width
        self.fc1 = nn.Linear(flat_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, features)
        b = x.size(0)
        x = x.view(b, 1, 18, 11)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)


def one_sided_loss(y_true, y_pred):
    grade = y_true[:,0]
    flag = y_true[:,1]
    err = y_pred - grade
    loss_rem = torch.square(torch.clamp(err, min=0.0))
    loss_add = torch.square(torch.clamp(-err, min=0.0))
    loss_orig = torch.square(err)
    loss = torch.where(flag<0, loss_rem, torch.where(flag>0, loss_add, loss_orig))
    return loss.mean()

# Training loop
def train():
    X_train, y_train, X_val, y_val_aug, y_val, int_to_grade = load_and_preprocess()
    train_ds = MoonboardDataset(X_train, y_train)
    val_ds = MoonboardDataset(X_val, y_val_aug)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MoonModel(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters())

    best_loss = float('inf')
    patience, wait = 10, 0
    for epoch in range(1, 1001):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = one_sided_loss(yb, preds)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_losses.append(one_sided_loss(yb, preds).item())
        avg_val = np.mean(val_losses)
        if avg_val < best_loss:
            best_loss = avg_val; best_model = model.state_dict(); wait = 0
            torch.save(model.state_dict(), 'moonboard_model.pth')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        print(f"Epoch {epoch}, Val Loss: {avg_val:.4f}")

    # load best
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), 'moonboard_model.pth')

    # evaluate
    model.eval()
    Xv = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        preds = model(Xv).cpu().numpy()
    mse = mean_squared_error(y_val, preds)
    idx = np.clip(np.rint(preds).astype(int), 0, len(int_to_grade)-1)
    print(f"MSE: {mse:.3f}")
    print(f"Exact: {np.mean(idx==y_val)*100:.1f}%")
    for d in [1,2,3]:
        print(f"Off{d}: {np.mean(np.abs(idx-y_val)<=d)*100:.1f}%")

# prediction helper
def predict_route(moves, model, input_dim, int_to_grade):
    vec = np.zeros((1, input_dim), dtype=np.float32)
    # you need pos2i mappings available globally or pass them
    for m in moves:
        if m in pos2i_main:
            vec[0, pos2i_main[m]] = 1
        if m in pos2i_start:
            vec[0, len(dims_main) + pos2i_start[m]] = 1
        if m in pos2i_end:
            vec[0, len(dims_main) + len(dims_start) + pos2i_end[m]] = 1
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(vec).to(device)).item()
    return p, int_to_grade[int(np.clip(round(p), 0, len(int_to_grade)-1))]

if __name__ == '__main__':
    train()
