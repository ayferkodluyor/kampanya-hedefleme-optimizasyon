import pandas as pd
import numpy as np
from pathlib import Path

src = Path("/mnt/data/a8a0c62e-a9a4-4b6f-a862-3cd065152622.csv")
df = pd.read_csv(src)

# Build a separate, unlabeled "new campaign population" from the same synthetic feature space.
rng = np.random.default_rng(42)
n_new = 10000

feature_cols = [c for c in df.columns if c not in ["Musteri_ID", "Kampanyaya_Yanit"]]
new_pool = df[feature_cols].sample(n=n_new, replace=True, random_state=42).reset_index(drop=True).copy()

# Add small noise to continuous numeric columns so this is not just a row-for-row duplicate.
continuous_cols = ["Yas", "Musteri_Kidem_Ay", "Aylik_Ortalama_Gelir", "Aylik_Ortalama_Harcama", "Son_3Ay_Islem_Sayisi"]
for col in continuous_cols:
    if col in new_pool.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        std = float(s.std()) if float(s.std()) > 0 else 1.0
        noise_scale = {
            "Yas": 1.5,
            "Musteri_Kidem_Ay": 6.0,
            "Aylik_Ortalama_Gelir": std * 0.08,
            "Aylik_Ortalama_Harcama": std * 0.08,
            "Son_3Ay_Islem_Sayisi": 3.0,
        }[col]
        new_pool[col] = pd.to_numeric(new_pool[col], errors="coerce") + rng.normal(0, noise_scale, size=n_new)

# Keep sensible ranges and integer-like columns
if "Yas" in new_pool:
    new_pool["Yas"] = new_pool["Yas"].round().clip(18, 80).astype(int)
if "Musteri_Kidem_Ay" in new_pool:
    new_pool["Musteri_Kidem_Ay"] = new_pool["Musteri_Kidem_Ay"].round().clip(lower=1).astype(int)
if "Son_3Ay_Islem_Sayisi" in new_pool:
    new_pool["Son_3Ay_Islem_Sayisi"] = new_pool["Son_3Ay_Islem_Sayisi"].round().clip(lower=0).astype(int)
for col in ["Urun_Sayisi", "Son_Kampanyaya_Katilim", "Son_12Ay_Kampanya_Katilim_Sayisi", "Kredi_Karti_Var", "Mevduat_Urunu_Var"]:
    if col in new_pool:
        new_pool[col] = pd.to_numeric(new_pool[col], errors="coerce").round().clip(lower=0).astype(int)

new_pool.insert(0, "Musteri_ID", [f"YKM{str(i).zfill(5)}" for i in range(1, n_new + 1)])

pool_path = Path("/mnt/data/Yeni_Kampanya_Musteri_Havuzu.csv")
new_pool.to_csv(pool_path, index=False, encoding="utf-8-sig")

script = r'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# ---------------------------------------------------------
# 1) DOSYALARI OKU
# ---------------------------------------------------------
# Geçmiş kampanya verisi: sonucu biliyoruz (0/1)
gecmis = pd.read_csv("Kampanya_Yanit_Tahmini_Sentetik_Veri.csv")

# Yeni kampanya havuzu: sonucu henüz bilmiyoruz, hedef sütunu yok
yeni = pd.read_csv("Yeni_Kampanya_Musteri_Havuzu.csv")

# ---------------------------------------------------------
# 2) GİRDİLER (X) VE HEDEF (y)
# ---------------------------------------------------------
X = gecmis.drop(columns=["Musteri_ID", "Kampanyaya_Yanit"])
y = gecmis["Kampanyaya_Yanit"]

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

# ---------------------------------------------------------
# 3) VERİ HAZIRLAMA
# ---------------------------------------------------------
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

# class_weight="balanced" kaldırıldı.
# Çünkü burada üretilen olasılıkları "beklenen dönüş" hesabında kullanıyoruz.
model = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000)),
])

# ---------------------------------------------------------
# 4) TRAIN / TEST AYIR
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# ---------------------------------------------------------
# 5) MODELİ EĞİT VE TEST ET
# ---------------------------------------------------------
model.fit(X_train, y_train)

test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.50).astype(int)

auc = roc_auc_score(y_test, test_proba)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)

test_base_rate = y_test.mean()

eval_df = pd.DataFrame({
    "Gercek_Yanit": y_test.to_numpy(),
    "Tahmin_Olasiligi": test_proba,
}).sort_values("Tahmin_Olasiligi", ascending=False)

top_n = max(1, int(len(eval_df) * 0.20))
top20 = eval_df.head(top_n)
lift20 = top20["Gercek_Yanit"].mean() / test_base_rate

# ---------------------------------------------------------
# 6) DOĞRULAMA BİTTİKTEN SONRA:
#    TÜM GEÇMİŞ VERİYLE FİNAL MODELİ EĞİT
# ---------------------------------------------------------
model.fit(X, y)

# ---------------------------------------------------------
# 7) YENİ KAMPANYA MÜŞTERİ HAVUZUNU SKORLA
# ---------------------------------------------------------
yeni_X = yeni.drop(columns=["Musteri_ID"])
yeni_proba = model.predict_proba(yeni_X)[:, 1]

skorlanan = yeni[["Musteri_ID"]].copy()
skorlanan["Tahmin_Olasiligi"] = yeni_proba
skorlanan = skorlanan.sort_values("Tahmin_Olasiligi", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------
# 8) BÜTÇEYE GÖRE HEDEFLEME
# ---------------------------------------------------------
iletisim_maliyeti = 20
butce = 50000

ulasim_adedi = min(butce // iletisim_maliyeti, len(skorlanan))
hedef_grup = skorlanan.head(ulasim_adedi).copy()

# Beklenen dönüş: seçilen müşterilerin tahmin olasılıklarının toplamı
model_beklenen = int(round(hedef_grup["Tahmin_Olasiligi"].sum()))

# Rastgele baz çizgisi: geçmiş kampanyadaki genel yanıt oranı
genel_yanit_orani = y.mean()
rastgele_beklenen = int(round(ulasim_adedi * genel_yanit_orani))

ek_donus = model_beklenen - rastgele_beklenen

# ---------------------------------------------------------
# 9) SONUÇLARI YAZDIR
# ---------------------------------------------------------
print("\n=== KAMPANYA HEDEFLEME & OPTİMİZASYON ===")
print(f"Geçmiş kampanya kaydı: {len(gecmis):,}")
print(f"Yeni kampanya müşteri havuzu: {len(yeni):,}")

print("\n=== MODEL PERFORMANSI (TEST VERİSİ) ===")
print(f"ROC-AUC: {auc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Lift@20%: {lift20:.2f}x")
print(f"Test yanıt oranı: %{test_base_rate * 100:.1f}")

print("\n=== 50.000 TL BÜTÇE SENARYOSU ===")
print(f"İletişim maliyeti: {iletisim_maliyeti} TL / müşteri")
print(f"Hedeflenebilecek müşteri: {ulasim_adedi:,}")
print(f"Model bazlı beklenen dönüş: {model_beklenen:,}")
print(f"Rastgele seçim beklenen dönüş: {rastgele_beklenen:,}")
print(f"Tahmini ek dönüş: {ek_donus:+,}")

print("\nİlk 10 hedef müşteri:")
print(
    hedef_grup.head(10)
    .assign(Tahmin_Olasiligi=lambda d: (d["Tahmin_Olasiligi"] * 100).round(1).astype(str) + "%")
    .to_string(index=False)
)

# İstersen hedef listeyi ayrıca kaydedelim
hedef_grup.to_csv("Kampanya_Hedef_Listesi.csv", index=False, encoding="utf-8-sig")

print("\nNot: Tüm veriler sentetiktir. Gerçek müşteri veya banka verisi içermez.")
print("Not: 'Beklenen dönüş' tahmindir; garanti edilen gerçek kampanya sonucu değildir.")
'''

script_path = Path("/mnt/data/Kampanya_Yanit_DUZELTILMIS_FINAL.py")
script_path.write_text(script, encoding="utf-8")

# Run the corrected logic once here to verify it works with the generated pool.
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

hist = df.copy()
X = hist.drop(columns=["Musteri_ID", "Kampanyaya_Yanit"])
y = hist["Kampanyaya_Yanit"]

num_cols = ["Yas","Musteri_Kidem_Ay","Aylik_Ortalama_Gelir","Aylik_Ortalama_Harcama",
            "Urun_Sayisi","Son_3Ay_Islem_Sayisi","Son_Kampanyaya_Katilim",
            "Son_12Ay_Kampanya_Katilim_Sayisi","Kredi_Karti_Var","Mevduat_Urunu_Var"]
cat_cols = ["Mobil_Aktiflik","Dijital_Kanal_Kullanim","Kampanya_Kanali"]

prep = ColumnTransformer([
    ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
mdl = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
mdl.fit(X_train, y_train)
proba = mdl.predict_proba(X_test)[:,1]
pred = (proba >= 0.5).astype(int)
auc = roc_auc_score(y_test, proba)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
base_rate = y_test.mean()
eval_df = pd.DataFrame({"Gercek_Yanit": y_test.to_numpy(), "Tahmin_Olasiligi": proba}).sort_values("Tahmin_Olasiligi", ascending=False)
lift20 = eval_df.head(max(1,int(len(eval_df)*0.20)))["Gercek_Yanit"].mean()/base_rate

mdl.fit(X,y)
newX = new_pool.drop(columns=["Musteri_ID"])
newp = mdl.predict_proba(newX)[:,1]
sc = pd.DataFrame({"Musteri_ID": new_pool["Musteri_ID"], "Tahmin_Olasiligi": newp}).sort_values("Tahmin_Olasiligi", ascending=False)
n = 50000//20
model_exp = int(round(sc.head(n)["Tahmin_Olasiligi"].sum()))
rand_exp = int(round(n*y.mean()))

print("Dosyalar oluşturuldu ve kod test edildi.")
print(f"ROC-AUC={auc:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | Lift@20%={lift20:.2f}x")
print(f"50.000 TL -> {n:,} müşteri | model beklenen={model_exp:,} | rastgele={rand_exp:,} | fark={model_exp-rand_exp:+,}")
print(pool_path)
print(script_path)
