import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# =========================================================
# 1) VERIYI OKU
# =========================================================
cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols,
    engine="python"
)

# =========================================================
# 2) PROJE KAPSAMINDA GEREKLI TEMSILI SENSORLERI SEC
# =========================================================
# Bu seçim birebir çimento sensörü değil, temsilidir:
# s2  -> sıcaklık benzeri
# s3  -> sıcaklık/proses yükü benzeri
# s4  -> sıcaklık/proses davranışı benzeri
# s7  -> akış/basınç benzeri
# s11 -> motor/fan davranışı benzeri
# s15 -> motor/fan yükü benzeri

selected_sensors = ["s2", "s3", "s4", "s7", "s11", "s15"]

# İstersen raporda daha anlaşılır dursun diye yeniden adlandıralım
sensor_map = {
    "s2": "Temp_1",
    "s3": "Temp_2",
    "s4": "Process_Load",
    "s7": "Flow_Pressure",
    "s11": "Motor_Fan_1",
    "s15": "Motor_Fan_2"
}

work_df = df[["unit", "cycle"] + selected_sensors].copy()
work_df = work_df.rename(columns=sensor_map)

# Yeni sensör isimleri
model_features = list(sensor_map.values())

# =========================================================
# 3) TURETILMIS ENERJI DEGISKENI URET
# =========================================================
# Ham sensörleri önce 0-1 aralığına alıyoruz
mm_scaler = MinMaxScaler()
scaled_energy_inputs = mm_scaler.fit_transform(work_df[model_features])

scaled_energy_df = pd.DataFrame(
    scaled_energy_inputs,
    columns=model_features,
    index=work_df.index
)

# Ağırlıklı toplam ile temsilî enerji üret
# Motor/fan ve proses yüküne biraz daha fazla ağırlık verildi
work_df["Estimated_Energy"] = (
    0.20 * scaled_energy_df["Temp_1"] +
    0.20 * scaled_energy_df["Temp_2"] +
    0.20 * scaled_energy_df["Process_Load"] +
    0.15 * scaled_energy_df["Flow_Pressure"] +
    0.125 * scaled_energy_df["Motor_Fan_1"] +
    0.125 * scaled_energy_df["Motor_Fan_2"]
) * 100

# =========================================================
# 4) LAG FEATURE EKLE
# =========================================================
work_df["Energy_lag_1"] = work_df.groupby("unit")["Estimated_Energy"].shift(1)
work_df["Energy_lag_2"] = work_df.groupby("unit")["Estimated_Energy"].shift(2)

# Eksikleri doldur
work_df[["Energy_lag_1", "Energy_lag_2"]] = (
    work_df.groupby("unit")[["Energy_lag_1", "Energy_lag_2"]]
    .transform(lambda x: x.bfill().ffill())
)

# =========================================================
# 5) ISOLATION FOREST ICIN OZELLIKLERI HAZIRLA
# =========================================================
if_features = model_features + ["Estimated_Energy", "Energy_lag_1", "Energy_lag_2"]

X = work_df[if_features].copy()

std_scaler = StandardScaler()
X_scaled = std_scaler.fit_transform(X)

# =========================================================
# 6) ISOLATION FOREST
# =========================================================
model = IsolationForest(
    n_estimators=300,
    contamination=0.02,
    random_state=42
)

pred = model.fit_predict(X_scaled)
scores = model.decision_function(X_scaled)

work_df["anomaly"] = pred
work_df["anomaly_score"] = scores

# =========================================================
# 7) SONUCLAR
# =========================================================
total_anomalies = (work_df["anomaly"] == -1).sum()
print("Toplam anomali sayısı:", total_anomalies)

anomaly_by_unit = (
    work_df.groupby("unit")["anomaly"]
    .apply(lambda x: (x == -1).sum())
    .sort_values(ascending=False)
)

print("\nEn çok anomali içeren ilk 10 unit:")
print(anomaly_by_unit.head(10))

# Görselleştirme için en çok anomalili unit seç
plot_unit = anomaly_by_unit.index[0]
plot_df = work_df[work_df["unit"] == plot_unit].copy()

print(f"\nGrafikler için seçilen unit: {plot_unit}")
print(f"Bu unit içindeki anomali sayısı: {(plot_df['anomaly'] == -1).sum()}")

# =========================================================
# 8) GRAFIK 1 - SECILEN SENSORLERIN ZAMAN GRAFIGI
# =========================================================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_plot = scaler.fit_transform(plot_df[model_features])

scaled_plot_df = pd.DataFrame(
    scaled_plot,
    columns=model_features,
    index=plot_df.index
)

plt.figure(figsize=(14, 6))

for col in model_features:
    plt.plot(plot_df["cycle"], scaled_plot_df[col], linewidth=2, label=col)

plt.title(f"Grafik 1 - Normalize Sensör Davranışı (Unit {plot_unit})")
plt.xlabel("Cycle")
plt.ylabel("Normalized Value (0-1)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# 9) GRAFIK 2 - TURETILMIS ENERJI GRAFIGI
# =========================================================
plt.figure(figsize=(14, 6))
plt.plot(
    plot_df["cycle"],
    plot_df["Estimated_Energy"],
    linewidth=2,
    label="Estimated_Energy"
)

plt.title(f"Grafik 2 - Türetilmiş Enerji Göstergesi (Unit {plot_unit})")
plt.xlabel("Cycle")
plt.ylabel("Estimated Energy Index")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 10) GRAFIK 3 - ANOMALILI ENERJI GRAFIGI
# =========================================================
anom = plot_df[plot_df["anomaly"] == -1]

plt.figure(figsize=(14, 6))
plt.plot(
    plot_df["cycle"],
    plot_df["Estimated_Energy"],
    linewidth=2,
    label="Estimated_Energy"
)

plt.scatter(
    anom["cycle"],
    anom["Estimated_Energy"],
    color="red",
    s=50,
    label="Anomali"
)

plt.title(f"Grafik 3 - Isolation Forest ile Tespit Edilen Anomaliler (Unit {plot_unit})")
plt.xlabel("Cycle")
plt.ylabel("Estimated Energy Index")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()