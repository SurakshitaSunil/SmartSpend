"""
SmartSpend — AI Personal Finance Tracker
app.py: Flask web application
Trained on: Personal Finance + BudgetWise + UPI 2024 + Daily Household (249,696 rows)
"""

import os, io, base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from flask import Flask, render_template, request
from model import run_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SECTOR_COLORS = {
    'Food'         : '#4CAF50',
    'Transport'    : '#2196F3',
    'EMI & Rent'   : '#F44336',
    'Shopping'     : '#FF9800',
    'Utilities'    : '#9C27B0',
    'Entertainment': '#009688',
    'Health'       : '#E91E63',
    'Education'    : '#FF5722',
    'Others'       : '#9E9E9E',
}

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='#0F1117')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def make_sector_bar(pivot, n_days=60):
    """Stacked bar — daily sector spending, last n_days"""
    all_dates = list(pivot.index)
    dates     = all_dates[-n_days:]
    sub       = pivot.loc[pivot.index.isin(dates)]
    sectors   = [s for s in SECTOR_COLORS if s in sub.columns]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')
    bottom = np.zeros(len(sub))
    for sec in sectors:
        vals = sub[sec].values
        ax.bar(range(len(sub)), vals, bottom=bottom,
               color=SECTOR_COLORS[sec], label=sec, width=0.85)
        bottom += vals
    step = max(1, len(sub)//15)
    ax.set_xticks(range(0, len(sub), step))
    ax.set_xticklabels([sub.index[i] for i in range(0, len(sub), step)],
                       rotation=65, fontsize=7, color='#aaa')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.spines[:].set_color('#333')
    ax.set_title('Daily Spending by Sector (last 60 days)',
                 color='#fff', fontsize=12, pad=10)
    ax.legend(loc='upper right', fontsize=7,
              facecolor='#1a1a2e', labelcolor='white',
              framealpha=0.8, ncol=3)
    ax.grid(axis='y', color='#333', linewidth=0.4)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_anomaly_scatter(df, n=2000):
    """Scatter plot — normal vs anomaly, sample for performance"""
    sample = df.sample(min(n, len(df)), random_state=42)
    normal  = sample[~sample['is_anomaly']]
    flagged = sample[sample['is_anomaly']]
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')
    ax.scatter(normal['date'],  normal['amount'],
               c='#2196F3', s=12, alpha=0.3, label='Normal')
    ax.scatter(flagged['date'], flagged['amount'],
               c='#F44336', s=50, marker='^',
               label='Anomaly', zorder=5, alpha=0.8)
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.spines[:].set_color('#333')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    ax.set_title('Transaction Anomaly Timeline (sample of 2,000)',
                 color='#fff', fontsize=12, pad=10)
    ax.legend(fontsize=8, facecolor='#1a1a2e',
              labelcolor='white', framealpha=0.8)
    ax.grid(color='#333', linewidth=0.4)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_forecast_chart(forecast_data):
    monthly = forecast_data['monthly']
    if len(monthly) < 2:
        return None
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')
    labels = [str(p) for p in monthly['date']]
    actual = monthly['total_spend'].values
    ax.plot(labels, actual, 'o-', color='#2196F3',
            lw=2, ms=4, label='Actual spend')
    if forecast_data['forecast']:
        labels_ext = labels + ['Next month']
        fval = forecast_data['forecast']
        ax.plot([labels[-1], 'Next month'],
                [actual[-1], fval],
                '--o', color='#FF9800', lw=2, ms=7,
                label=f'Forecast: ₹{fval:,.0f}')
    used_labels = labels_ext if forecast_data['forecast'] else labels
    step = max(1, len(used_labels)//10)
    ax.set_xticks(range(0, len(used_labels), step))
    ax.set_xticklabels(used_labels[::step],
                       rotation=45, fontsize=8, color='#aaa')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.spines[:].set_color('#333')
    ax.set_title('Monthly Budget Forecast — Linear Regression',
                 color='#fff', fontsize=12, pad=10)
    ax.legend(fontsize=8, facecolor='#1a1a2e',
              labelcolor='white', framealpha=0.8)
    ax.grid(color='#333', linewidth=0.4)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_cluster_bar(daily_clusters, n=90):
    cmap = {'Low day':'#4CAF50','Medium day':'#FF9800','High day':'#F44336'}
    recent = daily_clusters.tail(n)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')
    colors = [cmap.get(t,'#9E9E9E') for t in recent['day_type']]
    ax.bar(range(len(recent)), recent['total'].values,
           color=colors, width=0.85)
    step = max(1, len(recent)//15)
    ax.set_xticks(range(0, len(recent), step))
    ax.set_xticklabels([str(recent.iloc[i]['date']) for i in range(0, len(recent), step)],
                       rotation=65, fontsize=7, color='#aaa')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.spines[:].set_color('#333')
    ax.set_title('Daily Spending Clusters — K-Means  '
                 '(Green=Low · Orange=Medium · Red=High)',
                 color='#fff', fontsize=11, pad=10)
    ax.grid(axis='y', color='#333', linewidth=0.4)
    fig.tight_layout()
    return fig_to_b64(fig)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    mode = request.form.get('mode', 'master')  # 'master' or 'upload'
    filepath = None

    if mode == 'upload':
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html',
                                   error='Please select a CSV file to upload.')
        f = request.files['file']
        if not f.filename.endswith('.csv'):
            return render_template('index.html',
                                   error='Only CSV files are supported.')
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)

    try:
        result = run_pipeline(filepath)
    except Exception as e:
        return render_template('index.html', error=str(e))

    df   = result['df']
    pivot= result['pivot']

    chart_sector   = make_sector_bar(pivot)
    chart_anomaly  = make_anomaly_scatter(df)
    chart_forecast = make_forecast_chart(result['forecast'])
    chart_cluster  = make_cluster_bar(result['daily_clusters'])

    anom_rows = [{
        'date'   : str(r['date'].date()),
        'amount' : f"₹{r['amount']:,.0f}",
        'sector' : r['sector'],
        'z_score': f"{r['sector_z']:.2f}",
        'score'  : f"{r['anomaly_score']:.3f}",
    } for _, r in result['anomalies'].iterrows()]

    sector_data = [{
        'sector': s,
        'amount': f"₹{v:,.0f}",
        'color' : SECTOR_COLORS.get(s, '#9E9E9E'),
    } for s, v in result['sector_totals'].items()]

    sec_forecast = [{
        'sector' : s,
        'forecast': f"₹{v:,.0f}",
    } for s, v in result['forecast']['sector_forecast'].items()]

    data_source = ("Kaggle Master Dataset — 249,696 transactions "
                   "(Personal Finance + BudgetWise + UPI 2024 + Daily Household)"
                   if filepath is None else f"Uploaded: {os.path.basename(filepath)}")

    return render_template('result.html',
        data_source   = data_source,
        total_spend   = f"₹{result['total_spend']:,.0f}",
        total_txns    = f"{result['total_txns']:,}",
        anomaly_count = f"{result['anomaly_count']:,}",
        top_sector    = result['top_sector'],
        forecast      = (f"₹{result['forecast']['forecast']:,.0f}"
                         if result['forecast']['forecast'] else 'N/A'),
        mae           = (f"₹{result['forecast']['mae']:,.0f}"
                         if result['forecast']['mae'] else 'N/A'),
        chart_sector  = chart_sector,
        chart_anomaly = chart_anomaly,
        chart_forecast= chart_forecast,
        chart_cluster = chart_cluster,
        anomaly_rows  = anom_rows,
        sector_data   = sector_data,
        sec_forecast  = sec_forecast,
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
