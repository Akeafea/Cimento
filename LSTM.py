import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from keras.callbacks import EarlyStopping


log_file = open("log.txt", "w", encoding="utf-8")
original_stdout = sys.stdout


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


sys.stdout = Tee(original_stdout, log_file)

cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols,
    engine="python"
)

print("Veri okundu:", df.shape)


sensor_map = {
    "s2": "Temp_1",
    "s3": "Temp_2",
    "s4": "Process_Load",
    "s7": "Flow_Pressure",
    "s11": "Motor_Fan_1",
    "s15": "Motor_Fan_2"
}

selected = list(sensor_map.keys())

work_df = df[["unit", "cycle"] + selected].copy()
work_df.rename(columns=sensor_map, inplace=True)

feature_cols = list(sensor_map.values())

print("Seçilen sensörler:", feature_cols)


#  ENERGY FEATURE
energy_scaler = MinMaxScaler()
scaled = energy_scaler.fit_transform(work_df[feature_cols])

scaled_df = pd.DataFrame(scaled, columns=feature_cols)

work_df["Estimated_Energy"] = (
    0.20 * scaled_df["Temp_1"] +
    0.20 * scaled_df["Temp_2"] +
    0.20 * scaled_df["Process_Load"] +
    0.15 * scaled_df["Flow_Pressure"] +
    0.125 * scaled_df["Motor_Fan_1"] +
    0.125 * scaled_df["Motor_Fan_2"]
) * 100

lstm_features = feature_cols + ["Estimated_Energy"]


unit_id = work_df.groupby("unit").size().idxmax()
unit_df = work_df[work_df["unit"] == unit_id].reset_index(drop=True)

print("Seçilen unit:", unit_id)


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(unit_df[lstm_features])


SEQ_LEN = 30
def create_seq(data, seq_len):
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])


X = create_seq(scaled_data, SEQ_LEN)

print("Sequence:", X.shape)


inputs = Input(shape=(SEQ_LEN, X.shape[2]))

encoded = LSTM(64, activation="relu", return_sequences=True, dropout=0.2)(inputs)
encoded = LSTM(32, activation="relu", return_sequences=False, dropout=0.2)(encoded)

decoded = RepeatVector(SEQ_LEN)(encoded)
decoded = LSTM(32, activation="relu", return_sequences=True, dropout=0.2)(decoded)
decoded = LSTM(64, activation="relu", return_sequences=True, dropout=0.2)(decoded)

outputs = TimeDistributed(Dense(X.shape[2]))(decoded)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X, X,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)


pd.DataFrame(history.history).to_csv("training_loss.csv", index=False)


#  RECONSTRUCTION ERROR

X_pred = model.predict(X, verbose=0)
error = np.mean(np.square(X - X_pred), axis=(1, 2))


threshold = np.mean(error) + 2 * np.std(error)

anomaly_flags = error > threshold

print("Threshold:", threshold)
print("Anomali sayısı:", anomaly_flags.sum())

#sonuçlar
result_df = unit_df.iloc[SEQ_LEN:].copy().reset_index(drop=True)
result_df["reconstruction_error"] = error
result_df["anomaly"] = anomaly_flags.astype(int)


result_df.to_csv("reconstruction_error.csv", index=False)
result_df[result_df["anomaly"] == 1].to_csv("anomalies.csv", index=False)


#  GRAFIK 1

plt.figure(figsize=(12, 5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.legend()
plt.title("Training Loss")
plt.grid()
plt.savefig("grafik_1.png")
plt.show()


# GRAFIK 2

plt.figure(figsize=(12, 5))
plt.plot(result_df["cycle"], result_df["reconstruction_error"])
plt.axhline(threshold, color="red", linestyle="--")
plt.title("Reconstruction Error")
plt.grid()
plt.savefig("grafik_2.png")
plt.show()


#  GRAFIK 3

anom = result_df[result_df["anomaly"] == 1]

plt.figure(figsize=(12, 5))
plt.plot(result_df["cycle"], result_df["Estimated_Energy"])
plt.scatter(anom["cycle"], anom["Estimated_Energy"], color="red")
plt.title("Anomaliler")
plt.grid()
plt.savefig("grafik_3.png")
plt.show()


log_file.close()