import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from model import SASRecModel

print("Загрузка данных и модели", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load('preprocessed_data.pt', map_location=device)
cnt_item = data['cnt_item']
track_to_index = data['track_to_index']

model = SASRecModel(cnt_item=cnt_item, max_seq_len=50)
model.load_state_dict(torch.load('checkpoints/sasrec_best.pth', map_location=device))
model = model.to(device)
model.eval()

print("Загрузка пользователей", flush=True)
url = "https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/listens.parquet"
df = pd.read_parquet(url, columns=['uid', 'item_id', 'played_ratio_pct'])
user_data = {}
for _, row in tqdm(df.iterrows(), total=len(df), desc="Пользователи"):
    user_data[row['uid']] = row['item_id']

root = tk.Tk()
root.title("SASRec модель - Музыкальные рекомендации")
root.geometry("700x500")
tk.Label(root, text="ID пользователя:").pack()
user_entry = tk.Entry(root)
user_entry.pack()
tk.Label(root, text="Количество рекомендаций:").pack()
k_var = tk.IntVar(value=10)
tk.Spinbox(root, from_=5, to=50, textvariable=k_var, width=5).pack()


def random_user():
    uid = random.choice(list(user_data.keys()))
    user_entry.delete(0, tk.END)
    user_entry.insert(0, str(uid))


def get_recommendations():
    try:
        uid = int(user_entry.get())
        history_raw = user_data[uid][-50:]
        history = [track_to_index.get(item, 0) for item in history_raw]
        indices, scores = model.predict_next(history, top_k=k_var.get())

        for item in tree.get_children():
            tree.delete(item)

        for i, (item, score) in enumerate(zip(indices, scores), 1):
            prob = 1 / (1 + np.exp(-score))
            tree.insert('', tk.END, values=(i, item, f"{score:.4f}", f"{prob:.2%}"))
    except:
        messagebox.showwarning("Ошибка", "Пользователь не найден")


tk.Button(root, text="Случайный пользователь", command=random_user).pack(pady=5)
tk.Button(root, text="Получить рекомендации", command=get_recommendations).pack(pady=10)

columns = ('Позиция', 'ID трека', 'Скор', 'Вероятность')
tree = ttk.Treeview(root, columns=columns, show='headings', height=15)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=120, anchor='center')
tree.pack(pady=10)

root.mainloop()