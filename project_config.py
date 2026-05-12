from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "train_FD001.txt"
OUTPUT_DIR = ROOT / "outputs"

RAW_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

SENSOR_MAP = {
    "s2": "Temp_1",
    "s3": "Temp_2",
    "s4": "Process_Load",
    "s7": "Flow_Pressure",
    "s11": "Motor_Fan_1",
    "s15": "Motor_Fan_2",
}

PROCESS_FEATURES = list(SENSOR_MAP.values())

ENERGY_WEIGHTS = {
    "Temp_1": 0.20,
    "Temp_2": 0.20,
    "Process_Load": 0.20,
    "Flow_Pressure": 0.15,
    "Motor_Fan_1": 0.125,
    "Motor_Fan_2": 0.125,
}

PROJECT_BUDGET_TL = 1_015_190
MONTHLY_ENERGY_COST_TL = 3_500_000

WORK_PACKAGES = [
    ("IP1", "Gereksinim Analizi ve Sistem Mimarisi", "1. Ay", 180),
    ("IP2", "SCADA Veri Toplama ve ETL Altyapisi", "2-3. Ay", 260),
    ("IP3", "Veri On Isleme ve Ozellik Muhendisligi", "4-5. Ay", 280),
    ("IP4", "Yapay Zeka Tahmin Modelleri", "6-8. Ay", 420),
    ("IP5", "Dijital Ikiz ve MPC Kontrol Motoru", "8-10. Ay", 360),
    ("IP6", "Arayuz, Entegrasyon ve Saha Testi", "10-11. Ay", 300),
    ("IP7", "Ekonomik Analiz, Raporlama ve Ticarilesme", "12. Ay", 180),
]

BUDGET_LINES = [
    ("M011 - Personel Giderleri", 794_900),
    ("M013 - Alet/Techizat/Yazilim", 53_000),
    ("M015 - Hizmet Alimi", 45_000),
    ("M012 - Seyahat", 20_000),
    ("Dokumantasyon/Diger", 10_000),
    ("Beklenmeyen Gider Payi (%10)", 92_290),
]
