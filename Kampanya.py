import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# DOSYA YOLU
# Bu sayede program hangi klasörden çalıştırılırsa çalıştırılsın
# CSV'yi Python dosyasının bulunduğu klasörde arar.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Kampanya_Yanit_Tahmini_Sentetik_Veri.csv"


# =========================================================
# VERİYİ OKU
# =========================================================
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nCSV bulunamadı:\n{CSV_PATH}\n\n"
        "Kampanya.py ile Kampanya_Yanit_Tahmini_Sentetik_Veri.csv "
        "aynı klasörde olmalı."
    )

df = pd.read_csv(CSV_PATH)

required_cols = {
    "Musteri_ID", "Kampanyaya_Yanit",
    "Yas", "Musteri_Kidem_Ay", "Aylik_Ortalama_Gelir",
    "Aylik_Ortalama_Harcama", "Urun_Sayisi",
    "Mobil_Aktiflik", "Son_3Ay_Islem_Sayisi",
    "Son_Kampanyaya_Katilim", "Son_12Ay_Kampanya_Katilim_Sayisi",
    "Kredi_Karti_Var", "Mevduat_Urunu_Var",
    "Dijital_Kanal_Kullanim", "Kampanya_Kanali",
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV'de eksik sütunlar var: {sorted(missing)}")


# =========================================================
# MODEL İÇİN X VE y
# =========================================================
X = df.drop(columns=["Musteri_ID", "Kampanyaya_Yanit"])
y = df["Kampanyaya_Yanit"]

num_cols = [
    "Yas",
    "Musteri_Kidem_Ay",
    "Aylik_Ortalama_Gelir",
    "Aylik_Ortalama_Harcama",
    "Urun_Sayisi",
    "Son_3Ay_Islem_Sayisi",
    "Son_Kampanyaya_Katilim",
    "Son_12Ay_Kampanya_Katilim_Sayisi",
    "Kredi_Karti_Var",
    "Mevduat_Urunu_Var",
]

cat_cols = [
    "Mobil_Aktiflik",
    "Dijital_Kanal_Kullanim",
    "Kampanya_Kanali",
]

prep = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]),
        num_cols,
    ),
    (
        "cat",
        Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]),
        cat_cols,
    ),
])

# class_weight="balanced" KULLANMIYORUZ.
# Çünkü tahmin olasılıklarını beklenen dönüş hesabında kullanıyoruz.
model = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000)),
])


# =========================================================
# TRAIN / TEST
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)

test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.50).astype(int)

auc = roc_auc_score(y_test, test_proba)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
base_rate_test = y_test.mean()

eval_df = pd.DataFrame({
    "Gercek_Yanit": y_test.to_numpy(),
    "Tahmin_Olasiligi": test_proba
}).sort_values("Tahmin_Olasiligi", ascending=False)

top_n = max(1, int(len(eval_df) * 0.20))
lift20 = eval_df.head(top_n)["Gercek_Yanit"].mean() / base_rate_test


# =========================================================
# FİNAL MODEL
# Test tamamlandıktan sonra tüm geçmiş veriyle eğitiyoruz.
# =========================================================
model.fit(X, y)


# =========================================================
# YENİ KAMPANYA MÜŞTERİ HAVUZU
# Ayrı CSV gerektirmez.
# Geçmiş sentetik verinin özellik dağılımından yeni 10.000
# örnek müşteri havuzu üretiyoruz.
# =========================================================
rng = np.random.default_rng(42)
n_new = 10000

feature_cols = [c for c in df.columns if c not in ["Musteri_ID", "Kampanyaya_Yanit"]]

new_pool = (
    df[feature_cols]
    .sample(n=n_new, replace=True, random_state=42)
    .reset_index(drop=True)
    .copy()
)

# Sürekli sayısal alanlara küçük değişiklikler ekliyoruz.
noise_scales = {
    "Yas": 1.5,
    "Musteri_Kidem_Ay": 6.0,
    "Aylik_Ortalama_Gelir": max(float(df["Aylik_Ortalama_Gelir"].std()) * 0.08, 1.0),
    "Aylik_Ortalama_Harcama": max(float(df["Aylik_Ortalama_Harcama"].std()) * 0.08, 1.0),
    "Son_3Ay_Islem_Sayisi": 3.0,
}

for col, scale in noise_scales.items():
    new_pool[col] = pd.to_numeric(new_pool[col], errors="coerce")
    new_pool[col] = new_pool[col] + rng.normal(0, scale, size=n_new)

new_pool["Yas"] = new_pool["Yas"].round().clip(18, 80).astype(int)
new_pool["Musteri_Kidem_Ay"] = new_pool["Musteri_Kidem_Ay"].round().clip(lower=1).astype(int)
new_pool["Son_3Ay_Islem_Sayisi"] = new_pool["Son_3Ay_Islem_Sayisi"].round().clip(lower=0).astype(int)

for col in [
    "Urun_Sayisi",
    "Son_Kampanyaya_Katilim",
    "Son_12Ay_Kampanya_Katilim_Sayisi",
    "Kredi_Karti_Var",
    "Mevduat_Urunu_Var",
]:
    new_pool[col] = pd.to_numeric(new_pool[col], errors="coerce").round().clip(lower=0).astype(int)

new_pool.insert(
    0,
    "Musteri_ID",
    [f"YKM{str(i).zfill(5)}" for i in range(1, n_new + 1)]
)

new_X = new_pool.drop(columns=["Musteri_ID"])
new_proba = model.predict_proba(new_X)[:, 1]

scored = new_pool.copy()
scored["Tahmin_Olasiligi"] = new_proba
scored = scored.sort_values("Tahmin_Olasiligi", ascending=False).reset_index(drop=True)

historical_rate = float(y.mean())
CONTACT_COST = 20


# =========================================================
# ARAYÜZ — MODERN FİNAL TASARIM
# Model ve hesaplama mantığı yukarıda aynen korunmuştur.
# =========================================================
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

root = tk.Tk()
root.title("Kampanya Hedefleme & Optimizasyon")
root.geometry("1280x820")
root.minsize(1120, 720)
root.configure(bg="#F7F5FB")

# Renkler
PURPLE = "#5B3FA8"
PURPLE_DARK = "#39276E"
TEAL = "#008C8C"
PINK = "#D94F8A"
TEXT = "#29233A"
MUTED = "#746D82"
BORDER = "#E5E0EE"
BG = "#F7F5FB"
WHITE = "#FFFFFF"
PALE_PURPLE = "#F1ECFA"
PALE_TEAL = "#EAF7F6"
PALE_PINK = "#FCECF3"
PALE_BLUE = "#EDF4FC"

style = ttk.Style()
try:
    style.theme_use("clam")
except Exception:
    pass

style.configure(
    "Treeview",
    background=WHITE,
    fieldbackground=WHITE,
    foreground=TEXT,
    rowheight=29,
    borderwidth=0,
    font=("Segoe UI", 9),
)
style.configure(
    "Treeview.Heading",
    background="#F3EFF8",
    foreground=PURPLE_DARK,
    relief="flat",
    font=("Segoe UI", 9, "bold"),
)
style.map("Treeview.Heading", background=[("active", "#ECE6F5")])
style.configure("Horizontal.TScale", background=WHITE)


def card(parent, bg=WHITE, padx=0, pady=0):
    f = tk.Frame(
        parent,
        bg=bg,
        highlightthickness=1,
        highlightbackground=BORDER,
        bd=0,
    )
    return f


# ---------- HEADER ----------
header = tk.Frame(root, bg=PURPLE, height=104)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(
    header,
    text="KAMPANYA HEDEFLEME & OPTİMİZASYON",
    font=("Segoe UI", 24, "bold"),
    fg=WHITE,
    bg=PURPLE,
).pack(anchor="w", padx=34, pady=(18, 1))

tk.Label(
    header,
    text="Doğru müşteriye, doğru bütçeyle, daha yüksek beklenen dönüş",
    font=("Segoe UI", 10),
    fg="#EEE9FA",
    bg=PURPLE,
).pack(anchor="w", padx=36)

# ---------- SCROLLABLE MAIN ----------
outer = tk.Frame(root, bg=BG)
outer.pack(fill="both", expand=True)

canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
content = tk.Frame(canvas, bg=BG)

content.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
window_id = canvas.create_window((0, 0), window=content, anchor="nw")
canvas.configure(yscrollcommand=vscroll.set)

def resize_content(event):
    canvas.itemconfigure(window_id, width=event.width)

canvas.bind("<Configure>", resize_content)
canvas.pack(side="left", fill="both", expand=True)
vscroll.pack(side="right", fill="y")

# ---------- QUOTE ----------
quote = tk.Frame(content, bg="#EEE9FA", height=48)
quote.pack(fill="x", padx=26, pady=(18, 12))
quote.pack_propagate(False)
tk.Label(
    quote,
    text="“Veri, fırsatları görünür kılar.”",
    font=("Segoe UI", 11, "italic"),
    fg=PURPLE_DARK,
    bg="#EEE9FA",
).pack(side="left", padx=18)

# ---------- BUDGET + MODEL ----------
top = tk.Frame(content, bg=BG)
top.pack(fill="x", padx=26)

budget_card = card(top)
budget_card.pack(side="left", fill="both", expand=True, padx=(0, 7))

tk.Label(
    budget_card, text="Kampanya Bütçesi",
    font=("Segoe UI", 12, "bold"), bg=WHITE, fg=TEXT
).pack(anchor="w", padx=18, pady=(14, 2))

budget_text = tk.StringVar(value="50.000 TL")
tk.Label(
    budget_card, textvariable=budget_text,
    font=("Segoe UI", 22, "bold"), bg=WHITE, fg=PURPLE
).pack(anchor="w", padx=18)

slider = ttk.Scale(budget_card, from_=10000, to=200000, orient="horizontal")
slider.set(50000)
slider.pack(fill="x", padx=18, pady=(9, 5))

tk.Label(
    budget_card,
    text=f"10.000 TL                                    200.000 TL",
    font=("Segoe UI", 8), bg=WHITE, fg=MUTED
).pack(fill="x", padx=18)

tk.Label(
    budget_card,
    text=f"İletişim maliyeti varsayımı: {CONTACT_COST} TL / müşteri",
    font=("Segoe UI", 9), bg=WHITE, fg=MUTED
).pack(anchor="w", padx=18, pady=(5, 13))

model_card = card(top)
model_card.pack(side="left", fill="both", expand=True, padx=(7, 0))

tk.Label(
    model_card, text="Model Performansı",
    font=("Segoe UI", 12, "bold"), bg=WHITE, fg=TEXT
).pack(anchor="w", padx=18, pady=(14, 8))

metrics_row = tk.Frame(model_card, bg=WHITE)
metrics_row.pack(fill="x", padx=12, pady=(0, 12))

for title, value, color in [
    ("ROC-AUC", f"{auc:.3f}", PURPLE),
    ("Precision", f"{precision:.3f}", TEAL),
    ("Recall", f"{recall:.3f}", PINK),
    ("Lift@20%", f"{lift20:.2f}x", PURPLE),
]:
    box = tk.Frame(metrics_row, bg="#FAF9FC")
    box.pack(side="left", fill="both", expand=True, padx=4)
    tk.Label(box, text=title, font=("Segoe UI", 8), bg="#FAF9FC", fg=MUTED).pack(pady=(9, 1))
    tk.Label(box, text=value, font=("Segoe UI", 14, "bold"), bg="#FAF9FC", fg=color).pack(pady=(0, 9))

# ---------- KPI CARDS ----------
kpi_frame = tk.Frame(content, bg=BG)
kpi_frame.pack(fill="x", padx=21, pady=(13, 12))

kpi_vars = {
    "target": tk.StringVar(),
    "model": tk.StringVar(),
    "random": tk.StringVar(),
    "extra": tk.StringVar(),
}

def make_kpi(parent, title, var, bg_color, value_color):
    f = tk.Frame(
        parent, bg=bg_color,
        highlightthickness=1, highlightbackground=BORDER
    )
    f.pack(side="left", fill="both", expand=True, padx=5)
    tk.Label(f, text=title, font=("Segoe UI", 9), bg=bg_color, fg=MUTED).pack(pady=(12, 3))
    tk.Label(f, textvariable=var, font=("Segoe UI", 20, "bold"), bg=bg_color, fg=value_color).pack(pady=(0, 12))

make_kpi(kpi_frame, "Hedef Müşteri", kpi_vars["target"], PALE_BLUE, PURPLE)
make_kpi(kpi_frame, "Model Bazlı Beklenen", kpi_vars["model"], PALE_TEAL, TEAL)
make_kpi(kpi_frame, "Rastgele Beklenen", kpi_vars["random"], PALE_PINK, PINK)
make_kpi(kpi_frame, "Tahmini Ek Dönüş", kpi_vars["extra"], PALE_PURPLE, PURPLE)

# ---------- COMPARISON GRAPH + SCENARIOS ----------
middle = tk.Frame(content, bg=BG)
middle.pack(fill="x", padx=26)

graph_card = card(middle)
graph_card.pack(side="left", fill="both", expand=True, padx=(0, 7))

tk.Label(
    graph_card, text="Model vs. Rastgele Hedefleme",
    font=("Segoe UI", 12, "bold"), bg=WHITE, fg=TEXT
).pack(anchor="w", padx=18, pady=(13, 5))

graph_canvas = tk.Canvas(graph_card, height=145, bg=WHITE, highlightthickness=0)
graph_canvas.pack(fill="x", padx=18, pady=(0, 10))

scenario_card = card(middle)
scenario_card.pack(side="left", fill="both", expand=True, padx=(7, 0))

tk.Label(
    scenario_card, text="Bütçe Senaryoları",
    font=("Segoe UI", 12, "bold"), bg=WHITE, fg=TEXT
).pack(anchor="w", padx=18, pady=(13, 6))

scenario_text = tk.Text(
    scenario_card, height=7, bd=0, bg=WHITE, fg=TEXT,
    font=("Consolas", 9), state="disabled"
)
scenario_text.pack(fill="both", expand=True, padx=18, pady=(0, 10))

# ---------- TABLE ----------
table_card = card(content)
table_card.pack(fill="both", expand=True, padx=26, pady=(13, 10))

table_head = tk.Frame(table_card, bg=WHITE)
table_head.pack(fill="x", padx=16, pady=(11, 7))

tk.Label(
    table_head, text="Öncelikli Hedef Müşteriler",
    font=("Segoe UI", 12, "bold"), bg=WHITE, fg=TEXT
).pack(side="left")

tk.Label(
    table_head, text="İlk 50 müşteri gösteriliyor",
    font=("Segoe UI", 8), bg=WHITE, fg=MUTED
).pack(side="right")

table_wrap = tk.Frame(table_card, bg=WHITE)
table_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 12))

columns = ("Musteri_ID", "Olasilik", "Mobil", "Kanal", "Urun")
tree = ttk.Treeview(table_wrap, columns=columns, show="headings", height=10)

for key, text, width in [
    ("Musteri_ID", "Müşteri ID", 130),
    ("Olasilik", "Yanıt Olasılığı", 150),
    ("Mobil", "Mobil Aktiflik", 150),
    ("Kanal", "Kampanya Kanalı", 190),
    ("Urun", "Ürün Sayısı", 110),
]:
    tree.heading(key, text=text)
    tree.column(key, width=width, anchor="center")

table_scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=table_scroll.set)
tree.pack(side="left", fill="both", expand=True)
table_scroll.pack(side="right", fill="y")

# ---------- FOOTER ----------
footer = tk.Frame(content, bg=BG)
footer.pack(fill="x", padx=26, pady=(0, 18))

status_var = tk.StringVar()
tk.Label(
    footer, textvariable=status_var,
    font=("Segoe UI", 8), bg=BG, fg=MUTED
).pack(side="left")

def export_target():
    budget = int(round(float(slider.get()) / 5000) * 5000)
    n = min(budget // CONTACT_COST, len(scored))
    out = scored.head(n).copy()
    path = filedialog.asksaveasfilename(
        title="Hedef listeyi kaydet",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        initialfile="Kampanya_Hedef_Listesi.csv",
    )
    if path:
        out.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Kaydedildi", "Hedef liste başarıyla kaydedildi.")

tk.Button(
    footer,
    text="Hedef Listeyi CSV Olarak Kaydet",
    command=export_target,
    font=("Segoe UI", 9, "bold"),
    bg=PURPLE, fg=WHITE,
    activebackground=PURPLE_DARK, activeforeground=WHITE,
    relief="flat", cursor="hand2",
    padx=17, pady=8
).pack(side="right")


def expected_for_budget(budget):
    n = min(int(budget) // CONTACT_COST, len(scored))
    selected = scored.head(n)
    model_expected = int(round(selected["Tahmin_Olasiligi"].sum()))
    random_expected = int(round(n * historical_rate))
    return n, model_expected, random_expected


def draw_graph(model_expected, random_expected):
    graph_canvas.delete("all")
    graph_canvas.update_idletasks()
    w = max(graph_canvas.winfo_width(), 400)
    h = 145

    max_val = max(model_expected, random_expected, 1)
    left = 58
    usable = w - 95

    # model bar
    m_width = usable * model_expected / max_val
    graph_canvas.create_text(5, 38, text="Model", anchor="w", fill=TEXT, font=("Segoe UI", 9, "bold"))
    graph_canvas.create_rectangle(left, 25, left + m_width, 53, fill=TEAL, outline="")
    graph_canvas.create_text(left + m_width + 7, 39, text=f"{model_expected:,}".replace(",", "."), anchor="w", fill=TEAL, font=("Segoe UI", 9, "bold"))

    # random bar
    r_width = usable * random_expected / max_val
    graph_canvas.create_text(5, 83, text="Rastgele", anchor="w", fill=TEXT, font=("Segoe UI", 9, "bold"))
    graph_canvas.create_rectangle(left, 70, left + r_width, 98, fill=PINK, outline="")
    graph_canvas.create_text(left + r_width + 7, 84, text=f"{random_expected:,}".replace(",", "."), anchor="w", fill=PINK, font=("Segoe UI", 9, "bold"))

    if random_expected > 0:
        increase = ((model_expected / random_expected) - 1) * 100
        graph_canvas.create_text(
            left, 124,
            text=f"Model ile yaklaşık %{increase:.0f} daha yüksek beklenen dönüş",
            anchor="w", fill=PURPLE_DARK, font=("Segoe UI", 9, "bold")
        )


def update_scenarios():
    rows = []
    for b in [20000, 50000, 100000, 200000]:
        n, m, r = expected_for_budget(b)
        rows.append(
            f"{b:>7,} TL   {n:>5,} müşteri   Model {m:>5,}   Rastgele {r:>5,}"
            .replace(",", ".")
        )
    scenario_text.configure(state="normal")
    scenario_text.delete("1.0", "end")
    scenario_text.insert("1.0", "\n".join(rows))
    scenario_text.configure(state="disabled")


def update_dashboard(_=None):
    budget = int(round(float(slider.get()) / 5000) * 5000)
    budget = max(10000, min(200000, budget))
    budget_text.set(f"{budget:,.0f} TL".replace(",", "."))

    n, model_expected, random_expected = expected_for_budget(budget)
    extra = model_expected - random_expected
    selected = scored.head(n)

    kpi_vars["target"].set(f"{n:,}".replace(",", "."))
    kpi_vars["model"].set(f"{model_expected:,}".replace(",", "."))
    kpi_vars["random"].set(f"{random_expected:,}".replace(",", "."))
    kpi_vars["extra"].set(("+" if extra >= 0 else "") + f"{extra:,}".replace(",", "."))

    status_var.set(
        f"Demo amaçlıdır  •  Geçmiş veri: {len(df):,} kayıt  •  "
        f"Yeni havuz: {len(scored):,} müşteri  •  "
        "Veri tamamen sentetiktir  •  Gerçek müşteri veya banka verisi içermez"
        .replace(",", ".")
    )

    for item in tree.get_children():
        tree.delete(item)

    for _, row in selected.head(50).iterrows():
        tree.insert(
            "", "end",
            values=(
                row["Musteri_ID"],
                f"%{row['Tahmin_Olasiligi'] * 100:.1f}",
                row["Mobil_Aktiflik"],
                row["Kampanya_Kanali"],
                int(row["Urun_Sayisi"]),
            )
        )

    draw_graph(model_expected, random_expected)


slider.configure(command=update_dashboard)
update_scenarios()
root.after(100, update_dashboard)

root.mainloop()
