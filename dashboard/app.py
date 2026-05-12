from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame, limit: int = 12) -> str:
    head = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.head(limit).iterrows():
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _metric_value(df: pd.DataFrame, key: str, default: str = "-") -> str:
    if df.empty or "metric" not in df.columns:
        return default
    match = df.loc[df["metric"] == key, "value"]
    if match.empty:
        return default
    value = match.iloc[0]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key in {"r2", "mae", "rmse"}:
        return f"{value:.3f}"
    return f"{value:.2f}"


def build_dashboard(
    output_path: Path,
    summary: dict,
    metrics: pd.DataFrame,
    validation_notes: pd.DataFrame,
    quality_metrics: pd.DataFrame,
    quality_predictions: pd.DataFrame,
    recommendations: pd.DataFrame,
    economics: pd.DataFrame,
    budget: pd.DataFrame,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards = [
        ("Toplam Gozlem", f"{summary['total_rows']:,}"),
        ("Anomali Orani", f"{summary['ensemble_anomaly_rate_pct']:.2f}%"),
        ("Ortalama Guven", f"{summary['avg_confidence']:.2f}"),
        ("Beklenen Geri Odeme", f"{summary['expected_payback_months']:.1f} ay"),
        ("Free Lime MAE", f"{summary.get('quality_mae', 0):.3f}"),
    ]
    card_html = "".join(
        f"<section class='card'><span>{title}</span><strong>{value}</strong></section>"
        for title, value in cards
    )
    quality_snapshot = [
        ("MAE", _metric_value(quality_metrics, "mae")),
        ("RMSE", _metric_value(quality_metrics, "rmse")),
        ("R2", _metric_value(quality_metrics, "r2")),
        ("<=0.50 Hata", _metric_value(quality_metrics, "within_0_50")),
    ]
    quality_snapshot_html = "".join(
        f"<div class='mini-stat'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in quality_snapshot
    )
    html = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cimento MPC TEYDEP Demo</title>
  <style>
    :root {{
      --ink: #14212b;
      --muted: #576775;
      --line: #d6dee6;
      --paper: #f3f6f8;
      --steel: #2f6c8f;
      --green: #4d8a62;
      --sand: #d6b56a;
      --panel: rgba(255, 255, 255, 0.82);
      --coal: #1b2d39;
      --risk: #8e4b3e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background:
        linear-gradient(180deg, #edf3f7 0%, #f7fafb 42%, #eef2f4 100%);
    }}
    header {{
      padding: 44px clamp(20px, 5vw, 64px) 18px;
      border-bottom: 1px solid rgba(20, 33, 43, 0.08);
      background:
        linear-gradient(120deg, rgba(47, 108, 143, 0.10), transparent 40%),
        linear-gradient(180deg, rgba(77, 138, 98, 0.08), transparent 62%);
    }}
    .eyebrow {{
      color: var(--steel);
      font: 700 12px/1.2 "Segoe UI", sans-serif;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    h1 {{
      max-width: 960px;
      margin: 12px 0 8px;
      font-size: 52px;
      line-height: 1;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .lead {{
      max-width: 860px;
      color: var(--muted);
      font: 15px/1.65 "Segoe UI", sans-serif;
    }}
    main {{
      padding: 18px clamp(20px, 5vw, 64px) 56px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin: 0 0 18px;
    }}
    .card, .panel {{
      border: 1px solid rgba(20, 33, 43, 0.10);
      background: var(--panel);
      box-shadow: 0 10px 28px rgba(37, 58, 75, 0.08);
    }}
    .card {{
      min-height: 104px;
      padding: 16px;
      border-radius: 8px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font: 700 11px/1.2 "Segoe UI", sans-serif;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    .card strong {{
      display: block;
      margin-top: 22px;
      font-size: 34px;
      line-height: 1;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .section-band {{
      margin-bottom: 18px;
      border: 1px solid rgba(20, 33, 43, 0.08);
      background: rgba(255, 255, 255, 0.54);
      border-radius: 8px;
      overflow: hidden;
    }}
    .band-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(20, 33, 43, 0.08);
      background: linear-gradient(90deg, rgba(47, 108, 143, 0.08), rgba(77, 138, 98, 0.06));
    }}
    .band-head h2 {{
      margin: 0;
      font-size: 24px;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .band-head p {{
      margin: 4px 0 0;
      color: var(--muted);
      font: 13px/1.55 "Segoe UI", sans-serif;
    }}
    .band-tag {{
      white-space: nowrap;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(20, 33, 43, 0.06);
      color: var(--coal);
      font: 700 11px/1.2 "Segoe UI", sans-serif;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .grid-two {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 14px;
      align-items: start;
      padding: 14px;
    }}
    .panel {{
      overflow: auto;
      border-radius: 8px;
    }}
    .panel h2 {{
      margin: 0;
      padding: 16px 18px 8px;
      font-size: 20px;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .panel p {{
      margin: 0;
      padding: 0 18px 14px;
      color: var(--muted);
      font: 13px/1.6 "Segoe UI", sans-serif;
    }}
    .mini-stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 0 14px 14px;
    }}
    .mini-stat {{
      border: 1px solid rgba(20, 33, 43, 0.08);
      background: rgba(243, 246, 248, 0.92);
      border-radius: 8px;
      padding: 12px;
    }}
    .mini-stat span {{
      display: block;
      color: var(--muted);
      font: 700 10px/1.2 "Segoe UI", sans-serif;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .mini-stat strong {{
      display: block;
      margin-top: 10px;
      font-size: 24px;
      font-family: Georgia, "Times New Roman", serif;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font: 12px/1.45 "Segoe UI", sans-serif;
    }}
    th, td {{
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
      vertical-align: top;
    }}
    th {{
      color: var(--coal);
      background: rgba(47, 108, 143, 0.09);
      font-size: 10px;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    footer {{
      color: var(--muted);
      font: 12px/1.6 "Segoe UI", sans-serif;
      padding-top: 4px;
    }}
    .footnote {{
      margin-top: 12px;
      padding: 12px 14px;
      border: 1px dashed rgba(142, 75, 62, 0.32);
      border-radius: 8px;
      background: rgba(142, 75, 62, 0.04);
      color: var(--risk);
      font: 12px/1.6 "Segoe UI", sans-serif;
    }}
    @media (max-width: 920px) {{
      .cards,
      .grid-two,
      .mini-stats {{
        grid-template-columns: 1fr;
      }}
      h1 {{ font-size: 36px; }}
      .band-head {{
        display: block;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">TEYDEP 1507 prototip demonstrasyonu</div>
    <h1>Otonom MPC karar katmani ve enerji verimliligi panosu</h1>
    <p class="lead">Bu ekran iki farkli veri hikayesini birlikte gostermek icin guncellendi: NASA tabanli temsilî anomali karar destegi ve CAX cimento kalite verisi uzerinde Free Lime tahmini. Teknik demo ve daha gercekci kalite sinyali artik ayni sayfada, ama karistirilmadan duruyor.</p>
  </header>
  <main>
    <section class="cards">{card_html}</section>
    <section class="section-band">
      <div class="band-head">
        <div>
          <h2>Anomali ve MPC Demo Katmani</h2>
          <p>NASA kaynakli temsilî zaman serisi uzerinden anomali, risk ve operator onayli MPC onerileri.</p>
        </div>
        <div class="band-tag">Demo veri</div>
      </div>
      <div class="grid-two">
        <div>
          <article class="panel"><h2>Model basarim metrikleri</h2><p>Bu tablo prototip karar katmaninin ic tutarliligini gosterir; gercek saha etiketi yerine pseudo-ground-truth kullanir.</p>{_table(metrics, 8)}</article>
          <article class="panel"><h2>Metrik guvenilirlik notlari</h2><p>Overfitting ve leakage riski gercek etiketli backtest yapilana kadar acik teknik borc olarak izleniyor.</p>{_table(validation_notes, 8)}</article>
        </div>
        <div>
          <article class="panel"><h2>MPC operator onay kuyrugu</h2><p>Enerji tasarrufu ve proses riski arasinda denge kuran acik cevrim tavsiyeler.</p>{_table(recommendations, 14)}</article>
        </div>
      </div>
    </section>

    <section class="section-band">
      <div class="band-head">
        <div>
          <h2>CAX Free Lime Tahmin Katmani</h2>
          <p>Kaggle CAX cimento kalite verisinden uretilen daha gercekci kalite tahmini; burada hedef degisken gercek.</p>
        </div>
        <div class="band-tag">Gercek hedef</div>
      </div>
      <div class="grid-two">
        <div>
          <article class="panel">
            <h2>Kalite metrik ozeti</h2>
            <p>Zaman bazli holdout kullaniyoruz; yani daha sonraki donemleri gorulmemis veri gibi test ediyoruz.</p>
            <div class="mini-stats">{quality_snapshot_html}</div>
            {_table(quality_metrics, 8)}
          </article>
          <article class="panel"><h2>Submission tahminleri</h2><p>Test kalite olaylarindan son bilinen proses parametreleri as-of eslestirme ile baglandi.</p>{_table(quality_predictions, 12)}</article>
        </div>
        <div>
          <article class="panel"><h2>Ekonomik senaryolar</h2><p>Beklenen geri odeme, kalite ve enerji katmaninin birlikte sahaya alinmasi varsayimiyla hesaplanir.</p>{_table(economics, 8)}</article>
          <article class="panel"><h2>Butce ozeti</h2><p>TEYDEP is paketleriyle uyumlu toplam yatirim kalemleri.</p>{_table(budget, 8)}</article>
        </div>
      </div>
    </section>

    <div class="footnote">Anomali F1 degerleri demo seviyesindedir. CAX kalite metrikleri daha anlamli olsa da, proses degiskenlerinin fiziksel anlami ve zaman kaymasi saha uzmaniyla teyit edilmeden uretim karari icin tek basina kullanilmamali.</div>
    <footer>Uretilen dosyalar: model_predictions.csv, mpc_recommendations.csv, cax_quality_metrics.csv, cax_freelime_predictions.csv, prototype.sqlite, api_payload.json.</footer>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
