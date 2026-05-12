CREATE TABLE IF NOT EXISTS sensor_readings (
    unit INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    temp_1 REAL,
    temp_2 REAL,
    process_load REAL,
    flow_pressure REAL,
    motor_fan_1 REAL,
    motor_fan_2 REAL,
    estimated_energy REAL,
    sensor_reliability_score REAL,
    PRIMARY KEY (unit, cycle)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    unit INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    anomaly_score REAL,
    anomaly_confidence REAL,
    reconstruction_error REAL,
    quality_risk_60m REAL,
    ensemble_risk_score REAL,
    ensemble_anomaly INTEGER,
    risk_level TEXT,
    PRIMARY KEY (unit, cycle)
);

CREATE TABLE IF NOT EXISTS mpc_recommendations (
    unit INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    mode TEXT NOT NULL,
    fuel_rate_delta_pct REAL,
    fan_speed_delta_pct REAL,
    kiln_feed_delta_pct REAL,
    predicted_energy_saving_pct REAL,
    safety_score REAL,
    recommendation_confidence REAL,
    operator_message TEXT,
    PRIMARY KEY (unit, cycle)
);

CREATE TABLE IF NOT EXISTS economic_scenarios (
    scenario TEXT PRIMARY KEY,
    saving_rate_pct REAL,
    monthly_saving_tl REAL,
    annual_saving_tl REAL,
    payback_months REAL
);

CREATE TABLE IF NOT EXISTS quality_predictions (
    timestamp TEXT PRIMARY KEY,
    prediction REAL,
    prediction_confidence REAL
);

CREATE TABLE IF NOT EXISTS quality_regression_metrics (
    metric TEXT PRIMARY KEY,
    value REAL
);
