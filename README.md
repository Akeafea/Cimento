# Cimento MPC TEYDEP Prototip Demo

Bu depo, `Data/TEYDEP1507.pdf` raporundaki prototip mimarisine gore hazirlanmis calisir demo paketidir. Mevcut `train_FD001.txt` verisi SCADA benzeri anomali demosu icin, `CAX data` altindaki Kaggle/CAX cimento kalite verisi ise Free Lime kalite tahmini icin kullanilir.

## Veri Kurulumu

Ham veri dosyalari repo disinda tutulur. Projeyi calistirmak icin su dosyalari yerelde ayni klasor yapisiyla bulundur:

- `train_FD001.txt`
- `Data/TEYDEP1507.pdf`
- `CAX data/CAX_Train_Quality (1)/CAX_Train_Quality.csv`
- `CAX data/CAX_Test_Quality/CAX_Test_Quality.csv`
- `CAX data/CAX_Freelime_Submission_File.csv`

## Calistirma

```powershell
C:\Users\mhmtd\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe run_demo.py
```

Alternatif olarak kendi ortaminda:

```powershell
pip install -r requirements.txt
python run_demo.py
```

## Uretilen Ciktilar

- `outputs/model_predictions.csv`: Sensorler, enerji/kalite gostergeleri, anomali skorlari, guven skorlari ve pseudo label.
- `outputs/model_metrics.csv`: Precision, recall, F1, accuracy ve confusion matrix degerleri.
- `outputs/metric_reliability_notes.csv`: Metriklerin neden demo seviyesinde oldugunu, leakage/overfitting riskini ve alinacak aksiyonlari aciklar.
- `outputs/real_validation_plan.csv`: Gercek saha etiketiyle yapilacak backtest ve pilot dogrulama plani.
- `outputs/cax_quality_metrics.csv`: CAX cimento kalite verisi uzerinde Free Lime regresyon metrikleri.
- `outputs/cax_freelime_predictions.csv`: Submission timestampleri icin Free Lime tahminleri.
- `outputs/cax_quality_holdout_predictions.csv`: Zaman bazli holdout icin gercek/tahmin karsilastirmasi.
- `outputs/mpc_recommendations.csv`: Operator onayli acik cevrim MPC set degeri onerileri.
- `outputs/economic_scenarios.csv`: Muhafazakar, beklenen ve iyimser tasarruf/geri odeme senaryolari.
- `outputs/prototype.sqlite`: `sensor_readings`, `model_predictions`, `mpc_recommendations`, `economic_scenarios` tablolari.
- `outputs/dashboard.html`: TEYDEP demo dashboard arayuzu.
- `outputs/api_payload.json`: API tarafinin servis ettigi ozet payload.

## API

Demo ciktisini urettikten sonra:

```powershell
python api/main.py
```

Endpointler:

- `http://127.0.0.1:8765/summary`
- `http://127.0.0.1:8765/metrics`
- `http://127.0.0.1:8765/quality`
- `http://127.0.0.1:8765/recommendations`
- `http://127.0.0.1:8765/economics`

## Not

Anomali F1 ve guven skorlari, gercek saha etiketi olmadigi icin deterministik pseudo-ground-truth uzerinden hesaplanir. Bu nedenle bu metrikler sunum/prototip seviyesindedir; gercek performans iddiasi degildir.

CAX Free Lime metrikleri ise gercek hedef degiskeni olan `Output Parameter` uzerinde zaman bazli holdout ile hesaplanir. Bu kisim, NASA tabanli anomali demosuna gore daha guvenilir bir kalite tahmini gosterir; yine de fabrika sahasinda kullanmadan once proses uzmaniyla degisken anlamlari ve zaman kaymalari dogrulanmalidir.
