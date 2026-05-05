import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
import pandas as pd
from model import SASRecModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load('preprocessed_data.pt', map_location=device)
cnt_item = data['cnt_item']

model = SASRecModel(cnt_item=cnt_item, max_seq_len=50)
model.load_state_dict(torch.load('checkpoints/sasrec_best.pth', map_location=device))
model = model.to(device)
model.eval()

url = "https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/listens.parquet"
df = pd.read_parquet(url)
user_data = {}
for _, row in df.iterrows():
    user_data[row['uid']] = row['item_id']


print('успешно')
print(cnt_item)
print(device)
print(df)