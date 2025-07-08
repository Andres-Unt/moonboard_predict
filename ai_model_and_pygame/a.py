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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update dims to include foothold rows -1 and 0
cols = list(chr(65 + c) for c in range(11))  # A-K
rows = [-1, 0] + list(range(1, 19))           # foothold rows then 1-18
# Main grid positions
dims_main = [f"{c}{r}" for r in rows for c in cols]
dims_start = [f"{c}{r}" for r in rows for c in cols]
dims_end = [f"{c}{r}" for r in rows for c in cols]

# Index mappings
pos2i_main  = {p: i for i,p in enumerate(dims_main)}
pos2i_start = {p: i for i,p in enumerate(dims_start)}
pos2i_end   = {p: i for i,p in enumerate(dims_end)}

# Load & preprocess
def load_and_preprocess(path='moonboard_data.json'):
    with open(path, 'r') as f:
        raw = json.load(f)['data']

    # Grade mappings
    grades = ['6A+','6B','6B+','6C','6C+',
              '7A','7A+','7B','7B+','7C',
              '7C+','8A','8A+','8B','8B+']
    grade_to_int = {g:i for i,g in enumerate(grades)}
    int_to_grade = {i:g for g,i in grade_to_int.items()}

    # Filter entries
    filtered = []
    for e in raw:
        if e.get('method')!='Feet follow hands': continue
        ug = e.get('userGrade')
        if not ug: continue
        ug = ug.upper()
        if ug not in grade_to_int: continue
        if e.get('grade') is None or e['grade'].upper()!=ug: continue
        if e.get('repeats',0) < 10: continue
        has_s = has_e = False
        valid_start = True
        for m in e['moves']:
            d = m['description']
            if m.get('isStart'):
                has_s = True
                if d not in dims_start: valid_start=False
            if m.get('isEnd'): has_e=True
        if not (has_s and has_e and valid_start): continue
        filtered.append(e)

    N = len(filtered)
    # Prepare arrays
    X_main = np.zeros((N, len(dims_main)), np.float32)
    X_start = np.zeros((N, len(dims_start)), np.float32)
    X_end   = np.zeros((N, len(dims_end)), np.float32)
    y       = np.zeros((N,), np.float32)

    for i,e in enumerate(filtered):
        for m in e['moves']:
            d = m['description']
            if d in pos2i_main:  X_main[i, pos2i_main[d]] = 1
            if m.get('isStart') and d in pos2i_start:
                X_start[i, pos2i_start[d]] = 1
            if m.get('isEnd')   and d in pos2i_end:
                X_end[i, pos2i_end[d]]   = 1
        y[i] = grade_to_int[e['userGrade'].upper()]
        # after you compute X_main from the moves, add:
    footholds = ['B-1','D-1','F-1','H-1','J-1','B0','D0','F0','H0','J0']
    fh_idxs = [ pos2i_main[p] for p in footholds ]
    # ensure those positions are allowed in every example:
    X_main[:, fh_idxs] = 1.0


    # Stack channels: allowed, start, end
    X_all = np.stack([X_main, X_start, X_end], axis=1)
    # Note: for start/end channels we upsample to full grid by repeating values across foothold rows

    # Train/val split
    idx = np.random.permutation(N)
    s = int(0.8 * N)
    tr_idx, vl_idx = idx[:s], idx[s:]
    X_tr, X_val = X_all[tr_idx], X_all[vl_idx]
    y_tr, y_val = y[tr_idx], y[vl_idx]

    return X_tr, y_tr, X_val, y_val, int_to_grade

class MoonboardConvDataset(Dataset):
    def __init__(self, X, y, H=20, W=11):
        # X: [N, C, M] where M = H*W
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.H, self.W = H, W
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        # reshape each channel
        x = x.view(-1, self.H, self.W)
        return x, self.y[idx]

class ConvMoonModel(nn.Module):
    def __init__(self, in_ch=3, num_grades=15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.d1   = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.d2   = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.d3   = nn.Dropout2d(0.3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.d4   = nn.Dropout2d(0.4)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(64)
        self.d5   = nn.Dropout2d(0.5)
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(64, 1)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.d1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.d2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.d3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.d4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.d5(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x).squeeze(1)

# Example training loop
if __name__ == '__main__':
    X_tr, y_tr, X_val, y_val, int_to_grade = load_and_preprocess()
    train_ds = MoonboardConvDataset(X_tr, y_tr)
    val_ds   = MoonboardConvDataset(X_val, y_val)
    tr_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    vl_loader = DataLoader(val_ds, batch_size=128)

    model = ConvMoonModel().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 510):
        model.train()
        total_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_train = total_loss / len(train_ds)

        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.float().to(device)
                total_val += loss_fn(model(xb), yb).item() * xb.size(0)
        avg_val = total_val / len(val_ds)
        print(f"Epoch {epoch}: Train Loss {avg_train:.4f}, Val Loss {avg_val:.4f}")

    # Save
    torch.save(model.state_dict(), 'moonboard_conv.pth')
