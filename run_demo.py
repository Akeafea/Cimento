from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.app import build_dashboard
from data_ingestion.cax_connector import (
    cax_files_available,
    read_cax_quality,
    read_submission_timestamps,
    TEST_QUALITY_FILE,
    TRAIN_QUALITY_FILE,
)
from data_ingestion.scada_connector import load_scada_like_data
from digital_twin.simulator import simulate_control_effect
from features.lag_features import add_process_features
from models.isolation_forest import score_anomalies
from models.lstm_autoencoder import score_sequence_risk
from models.quality_regression import (
    build_prediction_frame,
    build_training_frame,
    predict_quality,
    train_quality_model,
)
from mpc.controller import build_mpc_recommendations
from preprocessing.cleaning import clean_process_data
from project_config import OUTPUT_DIR, PROCESS_FEATURES, PROJECT_BUDGET_TL, ROOT
from reports.economic_analysis import (
    build_budget_summary,
    build_economic_scenarios,
    build_work_package_table,
)


def _risk_level(score: float) -> str:
    if score >= 0.72:
        return "kritik"
    if score >= 0.48:
        return "uyari"
    return "normal"


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _build_validation_notes(metrics_df: pd.DataFrame) -> pd.DataFrame:
    notes = [
        {
            "check": "Etiket kaynagi",
            "status": "demo_only",
            "risk_level": "high",
            "explanation": "Gercek saha etiketi yok; pseudo_ground_truth model skorlarindan ve proses heuristiklerinden turetildi.",
            "recommended_action": "SCADA alarm, kalite laboratuvari, durus ve bakim kayitlariyla gercek etiket seti olustur.",
        },
        {
            "check": "Overfitting yorumu",
            "status": "not_measured",
            "risk_level": "medium",
            "explanation": "Bu demo agirlik egitimi yapmadigi icin klasik train overfit'i olculmedi; ancak etiket-skor benzerligi degerlendirme sızıntısı yaratabilir.",
            "recommended_action": "Zaman bazli train/test ayrimi ve unit-disjoint test ile tekrar olc.",
        },
        {
            "check": "Metrik kullanimi",
            "status": "presentation_ready",
            "risk_level": "medium",
            "explanation": "Precision/recall/F1 demo tutarliligini gosterir; gercek ariza yakalama basarimi olarak raporlanmamali.",
            "recommended_action": "Sunumda 'deterministik demo metriği' olarak adlandir.",
        },
        {
            "check": "Saha dogrulama",
            "status": "pending_real_data",
            "risk_level": "high",
            "explanation": "False alarm/day, detection lead time ve operator kabul orani henuz olculmedi.",
            "recommended_action": "En az 4-8 haftalik zaman damgali saha verisiyle backtest yap.",
        },
    ]
    best_f1 = float(metrics_df["f1"].max()) if not metrics_df.empty else 0.0
    notes.append(
        {
            "check": "Demo F1 yorumu",
            "status": "informational",
            "risk_level": "medium" if best_f1 >= 0.70 else "high",
            "explanation": f"En iyi demo F1={best_f1:.4f}; bu sayi prototip entegrasyonunu gosterir, saha performans garantisi degildir.",
            "recommended_action": "Gercek etiket gelene kadar karar destek sistemi acik cevrim/operator onayli calissin.",
        }
    )
    return pd.DataFrame(notes)


def _build_validation_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "P1",
                "validation_step": "Etiket toplama",
                "required_data": "SCADA alarmlari, durus kayitlari, kalite lab sonuclari, operator notlari",
                "success_metric": "Etiket kapsami >= 90%, zaman damgasi eslesme hatasi < 5 dk",
            },
            {
                "phase": "P2",
                "validation_step": "Zaman serisi backtest",
                "required_data": "Ilk %70 egitim/kalibrasyon, son %30 test",
                "success_metric": "Test F1, false alarm/day, detection lead time",
            },
            {
                "phase": "P3",
                "validation_step": "Unit-disjoint test",
                "required_data": "Egitimde olmayan hat/unit verileri",
                "success_metric": "Yeni unitlerde recall dususu < %15",
            },
            {
                "phase": "P4",
                "validation_step": "Acik cevrim saha pilotu",
                "required_data": "Operator onay/red kararlari ve enerji tuketimi",
                "success_metric": "Operator kabul orani, tasarruf %, kalite sapmasi yok",
            },
        ]
    )


def _build_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    remaining_life = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    model_alarm = (
        (df["if_anomaly"] == 1)
        | (df["lstm_anomaly"] == 1)
        | (df["ensemble_risk_score"] >= df["ensemble_risk_score"].quantile(0.975))
    )
    late_life_energy_drift = (
        (remaining_life <= 12)
        & (df["Estimated_Energy"] >= df["Estimated_Energy"].quantile(0.82))
        & (df["ensemble_risk_score"] >= df["ensemble_risk_score"].quantile(0.90))
    )
    return (model_alarm | late_life_energy_drift).astype(int)


def _write_sqlite(
    db_path: Path,
    sensor_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    recommendations: pd.DataFrame,
    economics: pd.DataFrame,
    quality_predictions: pd.DataFrame,
    quality_metrics: pd.DataFrame,
) -> None:
    schema = (ROOT / "data_ingestion" / "schema.sql").read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        conn.execute("DELETE FROM sensor_readings")
        conn.execute("DELETE FROM model_predictions")
        conn.execute("DELETE FROM mpc_recommendations")
        conn.execute("DELETE FROM economic_scenarios")
        conn.execute("DELETE FROM quality_predictions")
        conn.execute("DELETE FROM quality_regression_metrics")

        sensor_sql = sensor_df.rename(
            columns={
                "Temp_1": "temp_1",
                "Temp_2": "temp_2",
                "Process_Load": "process_load",
                "Flow_Pressure": "flow_pressure",
                "Motor_Fan_1": "motor_fan_1",
                "Motor_Fan_2": "motor_fan_2",
                "Estimated_Energy": "estimated_energy",
            }
        )
        sensor_sql[
            [
                "unit",
                "cycle",
                "timestamp",
                "temp_1",
                "temp_2",
                "process_load",
                "flow_pressure",
                "motor_fan_1",
                "motor_fan_2",
                "estimated_energy",
                "sensor_reliability_score",
            ]
        ].to_sql("sensor_readings", conn, if_exists="append", index=False)

        prediction_df[
            [
                "unit",
                "cycle",
                "anomaly_score",
                "anomaly_confidence",
                "reconstruction_error",
                "quality_risk_60m",
                "ensemble_risk_score",
                "ensemble_anomaly",
                "risk_level",
            ]
        ].to_sql("model_predictions", conn, if_exists="append", index=False)

        recommendations[
            [
                "unit",
                "cycle",
                "mode",
                "fuel_rate_delta_pct",
                "fan_speed_delta_pct",
                "kiln_feed_delta_pct",
                "predicted_energy_saving_pct",
                "safety_score",
                "recommendation_confidence",
                "operator_message",
            ]
        ].to_sql("mpc_recommendations", conn, if_exists="append", index=False)

        economics.to_sql("economic_scenarios", conn, if_exists="append", index=False)

        if not quality_predictions.empty:
            quality_sql = quality_predictions.rename(
                columns={"Timestamp_Shifted": "timestamp", "Prediction": "prediction"}
            )[["timestamp", "prediction", "prediction_confidence"]].copy()
            quality_sql["timestamp"] = quality_sql["timestamp"].astype(str)
            quality_sql.to_sql("quality_predictions", conn, if_exists="append", index=False)

        if not quality_metrics.empty:
            quality_metrics.rename(columns={"value": "value"})[["metric", "value"]].to_sql(
                "quality_regression_metrics", conn, if_exists="append", index=False
            )


def _run_cax_quality_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not cax_files_available():
        empty_metrics = pd.DataFrame(
            [{"metric": "status", "value": 0, "note": "CAX dataset not found"}]
        )
        return empty_metrics, pd.DataFrame(), {}

    train_long = read_cax_quality(TRAIN_QUALITY_FILE)
    test_long = read_cax_quality(TEST_QUALITY_FILE) if TEST_QUALITY_FILE.exists() else pd.DataFrame(columns=train_long.columns)
    submission_times = read_submission_timestamps()

    train_frame = build_training_frame(train_long)
    holdout, metric_values, model = train_quality_model(train_frame)
    prediction_frame = build_prediction_frame(train_long, test_long, submission_times)
    predictions = predict_quality(prediction_frame, model)

    metrics = pd.DataFrame(
        [
            {"metric": "mae", "value": metric_values["mae"], "note": "Mean absolute error on time holdout"},
            {"metric": "rmse", "value": metric_values["rmse"], "note": "Root mean squared error on time holdout"},
            {"metric": "r2", "value": metric_values["r2"], "note": "R2 on later-period holdout"},
            {"metric": "within_0_25", "value": metric_values["within_0_25"], "note": "Share of predictions within 0.25 Free Lime"},
            {"metric": "within_0_50", "value": metric_values["within_0_50"], "note": "Share of predictions within 0.50 Free Lime"},
            {"metric": "train_rows", "value": model["train_rows"], "note": "Rows before time split"},
            {"metric": "holdout_rows", "value": model["holdout_rows"], "note": "Rows after time split"},
        ]
    ).round(5)

    holdout_output = holdout[
        ["Timestamp_Shifted", "Output Parameter", "prediction"]
    ].rename(columns={"Output Parameter": "actual"})
    holdout_output.to_csv(OUTPUT_DIR / "cax_quality_holdout_predictions.csv", index=False)
    return metrics, predictions, metric_values


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = load_scada_like_data()
    cleaned = clean_process_data(raw)
    featured = add_process_features(cleaned)
    scored = score_sequence_risk(score_anomalies(featured))
    scored["ensemble_risk_score"] = (
        0.42 * scored["anomaly_score"]
        + 0.38 * scored["quality_risk_60m"]
        + 0.12 * (1 - scored["sensor_reliability_score"])
        + 0.08 * scored["reconstruction_error"]
    ).clip(0, 1)
    ensemble_threshold = scored["ensemble_risk_score"].quantile(0.965)
    scored["ensemble_anomaly"] = (
        (scored["ensemble_risk_score"] >= ensemble_threshold)
        | (scored["if_anomaly"] == 1)
        | (scored["lstm_anomaly"] == 1)
    ).astype(int)
    scored["risk_level"] = scored["ensemble_risk_score"].apply(_risk_level)
    scored["pseudo_ground_truth"] = _build_pseudo_labels(scored)

    metrics = []
    metric_specs = {
        "IsolationForest_Mock": "if_anomaly",
        "LSTM_Autoencoder_Mock": "lstm_anomaly",
        "Ensemble_MPC_Risk": "ensemble_anomaly",
    }
    for model_name, pred_col in metric_specs.items():
        row = {"model": model_name}
        row.update(_metrics(scored["pseudo_ground_truth"], scored[pred_col]))
        row["avg_confidence"] = (
            scored["anomaly_confidence"].mean()
            if pred_col == "if_anomaly"
            else scored["sequence_confidence"].mean()
            if pred_col == "lstm_anomaly"
            else (0.5 * scored["anomaly_confidence"] + 0.5 * scored["sequence_confidence"]).mean()
        )
        metrics.append(row)
    metrics_df = pd.DataFrame(metrics).round(4)
    validation_notes = _build_validation_notes(metrics_df)
    validation_plan = _build_validation_plan()
    quality_metrics, quality_predictions, quality_metric_values = _run_cax_quality_pipeline()

    recommendations = simulate_control_effect(build_mpc_recommendations(scored))
    economics = build_economic_scenarios()
    budget = build_budget_summary()
    work_packages = build_work_package_table()

    prediction_cols = [
        "unit",
        "cycle",
        "timestamp",
        *PROCESS_FEATURES,
        "Estimated_Energy",
        "Specific_Energy_kwh_t",
        "Specific_Heat_kcal_kg",
        "Clinker_Quality_Index",
        "sensor_reliability_score",
        "anomaly_score",
        "anomaly_confidence",
        "if_anomaly",
        "reconstruction_error",
        "quality_risk_60m",
        "sequence_confidence",
        "lstm_anomaly",
        "ensemble_risk_score",
        "ensemble_anomaly",
        "risk_level",
        "pseudo_ground_truth",
    ]
    prediction_output = scored[prediction_cols].copy()
    numeric_cols = prediction_output.select_dtypes(include="number").columns
    prediction_output[numeric_cols] = prediction_output[numeric_cols].round(5)
    prediction_output.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    recommendations.round(4).to_csv(OUTPUT_DIR / "mpc_recommendations.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    validation_notes.to_csv(OUTPUT_DIR / "metric_reliability_notes.csv", index=False)
    validation_plan.to_csv(OUTPUT_DIR / "real_validation_plan.csv", index=False)
    quality_metrics.to_csv(OUTPUT_DIR / "cax_quality_metrics.csv", index=False)
    quality_predictions[
        ["Timestamp_Shifted", "Prediction", "prediction_confidence"]
    ].round({"Prediction": 5, "prediction_confidence": 3}).to_csv(
        OUTPUT_DIR / "cax_freelime_predictions.csv",
        index=False,
    )
    economics.to_csv(OUTPUT_DIR / "economic_scenarios.csv", index=False)
    budget.to_csv(OUTPUT_DIR / "budget_summary.csv", index=False)
    work_packages.to_csv(OUTPUT_DIR / "work_packages.csv", index=False)

    summary = {
        "project": "Cimento Endustrisi Otonom MPC Sistemi",
        "teydep_target": "THS 3 -> THS 6 prototip demonstrasyonu",
        "total_rows": int(len(scored)),
        "total_units": int(scored["unit"].nunique()),
        "ensemble_anomaly_rate_pct": float(scored["ensemble_anomaly"].mean() * 100),
        "avg_confidence": float((0.5 * scored["anomaly_confidence"] + 0.5 * scored["sequence_confidence"]).mean()),
        "expected_payback_months": float(economics.loc[economics["scenario"] == "Beklenen", "payback_months"].iloc[0]),
        "quality_mae": float(quality_metric_values.get("mae", 0.0)),
        "quality_r2": float(quality_metric_values.get("r2", 0.0)),
        "project_budget_tl": PROJECT_BUDGET_TL,
        "outputs": {
            "dashboard": "outputs/dashboard.html",
            "database": "outputs/prototype.sqlite",
            "predictions": "outputs/model_predictions.csv",
            "recommendations": "outputs/mpc_recommendations.csv",
            "metrics": "outputs/model_metrics.csv",
            "quality_metrics": "outputs/cax_quality_metrics.csv",
            "quality_predictions": "outputs/cax_freelime_predictions.csv",
        },
    }
    api_payload = {
        "summary": summary,
        "metrics": metrics_df.to_dict(orient="records"),
        "metric_reliability_notes": validation_notes.to_dict(orient="records"),
        "real_validation_plan": validation_plan.to_dict(orient="records"),
        "quality_metrics": quality_metrics.to_dict(orient="records"),
        "quality_predictions": quality_predictions[
            ["Timestamp_Shifted", "Prediction", "prediction_confidence"]
        ]
        .head(30)
        .assign(Timestamp_Shifted=lambda df: df["Timestamp_Shifted"].astype(str))
        .round({"Prediction": 5, "prediction_confidence": 3})
        .to_dict(orient="records"),
        "recommendations": recommendations.sort_values("recommendation_confidence", ascending=False)
        .head(20)
        .round(4)
        .to_dict(orient="records"),
        "economics": economics.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "api_payload.json").write_text(
        json.dumps(api_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_sqlite(
        OUTPUT_DIR / "prototype.sqlite",
        scored[prediction_cols],
        scored[prediction_cols],
        recommendations,
        economics,
        quality_predictions,
        quality_metrics,
    )
    build_dashboard(
        OUTPUT_DIR / "dashboard.html",
        summary,
        metrics_df,
        validation_notes,
        quality_metrics,
        quality_predictions[
            ["Timestamp_Shifted", "Prediction", "prediction_confidence"]
        ]
        .head(20)
        .round({"Prediction": 5, "prediction_confidence": 3}),
        recommendations[
            [
                "unit",
                "cycle",
                "fuel_rate_delta_pct",
                "fan_speed_delta_pct",
                "kiln_feed_delta_pct",
                "predicted_energy_saving_pct",
                "safety_score",
                "recommendation_confidence",
                "digital_twin_status",
            ]
        ].sort_values("recommendation_confidence", ascending=False),
        economics,
        budget,
    )

    print("TEYDEP prototype demo generated.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
