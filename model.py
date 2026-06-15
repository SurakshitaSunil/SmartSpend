"""
SmartSpend — AI Personal Finance Tracker
model.py: All ML logic trained on real Kaggle master dataset
Sources: Personal Finance Dataset + BudgetWise + UPI 2024 + Daily Household
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(BASE_DIR, 'master_dataset.csv')


def load_master():
    df = pd.read_csv(MASTER_CSV)
    df['date']   = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df[df['amount'] > 0].dropna(subset=['date','amount','sector'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    df['day_of_week']    = df['date'].dt.dayofweek
    df['day_of_month']   = df['date'].dt.day
    df['month']          = df['date'].dt.month
    df['is_weekend']     = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
    df['rolling_7d']     = df['amount'].rolling(7,  min_periods=1).mean()
    df['rolling_30d']    = df['amount'].rolling(30, min_periods=1).mean()
    mean_a = df['amount'].mean()
    std_a  = df['amount'].std() or 1
    df['z_score']     = (df['amount'] - mean_a) / std_a
    df['spend_ratio'] = df['amount'] / df['rolling_30d'].replace(0, 1)
    df['sector_z']    = 0.0
    for sec in df['sector'].unique():
        mask = df['sector'] == sec
        mu   = df.loc[mask, 'amount'].mean()
        sig  = df.loc[mask, 'amount'].std() or 1
        df.loc[mask, 'sector_z'] = (df.loc[mask, 'amount'] - mu) / sig
    return df


def detect_anomalies(df):
    df = df.copy()
    df['is_anomaly']    = False
    df['anomaly_score'] = 0.0
    features = ['amount','day_of_week','day_of_month',
                'is_weekend','rolling_7d','z_score',
                'spend_ratio','sector_z']
    for sec, grp in df.groupby('sector'):
        if len(grp) < 5:
            continue
        X    = grp[features].fillna(0).values
        X_sc = StandardScaler().fit_transform(X)
        contamination = min(0.05, max(0.01, 1.0 / len(grp)))
        iso  = IsolationForest(n_estimators=200,
                               contamination=contamination,
                               random_state=42)
        preds  = iso.fit_predict(X_sc)
        scores = iso.decision_function(X_sc)
        df.loc[grp.index, 'is_anomaly']    = (preds == -1)
        df.loc[grp.index, 'anomaly_score'] = -scores
    return df


def forecast_budget(df):
    monthly = df.groupby(df['date'].dt.to_period('M')).agg(
        total_spend=('amount','sum'),
        txn_count  =('amount','count'),
        avg_txn    =('amount','mean')
    ).reset_index()
    monthly['month_num'] = range(len(monthly))
    result = {'monthly': monthly, 'forecast': None,
              'mae': None, 'sector_forecast': {}}
    if len(monthly) >= 3:
        X  = monthly[['month_num','txn_count','avg_txn']].values
        y  = monthly['total_spend'].values
        lr = LinearRegression()
        lr.fit(X[:-1], y[:-1])
        next_X = [[len(monthly), monthly['txn_count'].mean(),
                   monthly['avg_txn'].mean()]]
        result['forecast'] = round(max(lr.predict(next_X)[0], 0), 2)
        if len(monthly) >= 4:
            result['mae'] = round(
                mean_absolute_error(y[1:], lr.predict(X[1:])), 2)
    sec_monthly = (df.groupby(['sector', df['date'].dt.to_period('M')])
                   ['amount'].sum().reset_index())
    sec_monthly.columns = ['sector','period','total']
    for sec in df['sector'].unique():
        sd = sec_monthly[sec_monthly['sector']==sec].reset_index(drop=True)
        if len(sd) >= 3:
            sd['mn'] = range(len(sd))
            lr2 = LinearRegression()
            lr2.fit(sd[['mn']].values, sd['total'].values)
            pred = lr2.predict([[len(sd)]])[0]
            result['sector_forecast'][sec] = round(max(pred, 0), 2)
    return result


def cluster_days(df):
    daily = df.groupby(df['date'].dt.date).agg(
        total  =('amount','sum'),
        count  =('amount','count'),
        avg    =('amount','mean'),
        max_amt=('amount','max')
    ).reset_index()
    daily.columns = ['date','total','count','avg','max_amt']
    if len(daily) < 3:
        daily['day_type'] = 'Medium day'
        return daily
    X_sc = StandardScaler().fit_transform(
        daily[['total','count','avg','max_amt']])
    km   = KMeans(n_clusters=3, random_state=42, n_init=10)
    daily['cluster'] = km.fit_predict(X_sc)
    means   = daily.groupby('cluster')['total'].mean()
    lbl_map = {means.idxmin():'Low day', means.idxmax():'High day'}
    for k in range(3):
        if k not in lbl_map:
            lbl_map[k] = 'Medium day'
    daily['day_type'] = daily['cluster'].map(lbl_map)
    return daily


def daily_sector_pivot(df):
    pivot = df.pivot_table(
        index  =df['date'].dt.date,
        columns='sector',
        values ='amount',
        aggfunc='sum'
    ).fillna(0)
    pivot.index = [str(d) for d in pivot.index]
    return pivot


SECTOR_KEYWORDS = {
    'EMI & Rent'   : ['emi', 'rent', 'loan', 'mortgage', 'housing', 'flat', 'apartment', 'equated'],
    'Food'         : ['food', 'restaurant', 'cafe', 'coffee', 'swiggy', 'zomato', 'grocery',
                      'supermarket', 'meal', 'lunch', 'dinner', 'breakfast', 'snack', 'pizza',
                      'burger', 'bakery', 'kitchen', 'eat', 'dining', 'blinkit', 'zepto'],
    'Transport'    : ['transport', 'taxi', 'uber', 'ola', 'petrol', 'fuel', 'metro', 'bus',
                      'train', 'auto', 'cab', 'parking', 'toll', 'rapido', 'irctc', 'flight',
                      'airline', 'rapido', 'namma'],
    'Shopping'     : ['shopping', 'amazon', 'flipkart', 'myntra', 'clothing', 'fashion', 'mall',
                      'store', 'purchase', 'meesho', 'nykaa', 'bigbasket', 'retail'],
    'Utilities'    : ['utility', 'electricity', 'water', 'gas', 'internet', 'mobile', 'phone',
                      'bill', 'recharge', 'broadband', 'wifi', 'dth', 'airtel', 'jio', 'bsnl',
                      'bescom', 'tata power', 'indane'],
    'Entertainment': ['entertainment', 'movie', 'cinema', 'netflix', 'spotify', 'hotstar',
                      'disney', 'gaming', 'game', 'concert', 'event', 'prime', 'youtube',
                      'bookmyshow', 'pvr', 'inox'],
    'Health'       : ['health', 'medical', 'hospital', 'doctor', 'medicine', 'pharmacy',
                      'clinic', 'dental', 'gym', 'fitness', 'apollo', 'medplus', 'pharmeasy'],
    'Education'    : ['education', 'school', 'college', 'university', 'course', 'tuition',
                      'book', 'stationery', 'coaching', 'udemy', 'coursera', 'byju'],
}


def _sector_from_text(text):
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return sector
    return None


def _sector_rule(row):
    amt, dom, wknd = row['amount'], row['day_of_month'], row['is_weekend']
    if   2000 <= amt <= 35000 and dom <= 7:  return 'EMI & Rent'
    elif   20 <= amt <= 300   and not wknd:  return 'Transport'
    elif   50 <= amt <= 600   and not wknd:  return 'Food'
    elif  500 <= amt <= 8000  and wknd:      return 'Shopping'
    elif  200 <= amt <= 2500  and wknd:      return 'Entertainment'
    elif  200 <= amt <= 2000  and dom <= 10: return 'Utilities'
    elif  100 <= amt <= 4000  and not wknd:  return 'Health'
    else:                                    return 'Others'


def _load_user_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    date_col_keys = ['date', 'time', 'txn_date', 'transaction_date', 'value_date',
                     'posting_date', 'booking_date', 'trans_date']
    date_cols = [c for c in df.columns
                 if any(k in c for k in date_col_keys)]
    if not date_cols:
        raise ValueError(
            "No date column found. Ensure your CSV has a column named 'Date', "
            "'Transaction Date', or similar.")
    df.rename(columns={date_cols[0]: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='mixed',
                                dayfirst=True, errors='coerce')

    amt_col_keys = ['amount', 'debit', 'spend', 'withdrawal', 'dr',
                    'credit', 'cr', 'amt', 'transaction_amount', 'net_amount']
    amt_cols = [c for c in df.columns
                if any(x in c for x in amt_col_keys)]
    if not amt_cols:
        raise ValueError(
            "No amount column found. Ensure your CSV has a column named 'Amount', "
            "'Debit', 'Withdrawal', or similar.")
    df.rename(columns={amt_cols[0]: 'amount'}, inplace=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()
    df = df[df['amount'] > 0].dropna(subset=['date', 'amount'])
    if df.empty:
        raise ValueError(
            "No valid transactions found after parsing. "
            "Check that your date and amount columns contain valid data.")
    df = df.sort_values('date').reset_index(drop=True)
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend']   = (df['date'].dt.dayofweek >= 5).astype(int)
    df['day_of_week']  = df['date'].dt.dayofweek

    desc_col_keys = ['description', 'note', 'narration', 'remarks', 'particulars',
                     'details', 'merchant', 'category', 'memo', 'reference']
    desc_cols = [c for c in df.columns
                 if any(k in c for k in desc_col_keys)]

    if desc_cols:
        df['sector'] = df[desc_cols[0]].apply(_sector_from_text)
        unmatched = df['sector'].isna()
        if unmatched.any():
            df.loc[unmatched, 'sector'] = df[unmatched].apply(_sector_rule, axis=1)
    else:
        df['sector'] = df.apply(_sector_rule, axis=1)

    return df


def run_pipeline(filepath=None):
    """
    filepath=None  → analyse full Kaggle master dataset (249k rows)
    filepath given → analyse user-uploaded CSV
    """
    df = load_master() if filepath is None else _load_user_csv(filepath)
    df = engineer_features(df)
    df = detect_anomalies(df)

    forecast_data  = forecast_budget(df)
    daily_clusters = cluster_days(df)
    pivot          = daily_sector_pivot(df)

    return {
        'df'            : df,
        'total_spend'   : round(df['amount'].sum(), 2),
        'total_txns'    : len(df),
        'anomaly_count' : int(df['is_anomaly'].sum()),
        'top_sector'    : df.groupby('sector')['amount'].sum().idxmax(),
        'anomalies'     : (df[df['is_anomaly']]
                           .sort_values('anomaly_score', ascending=False)
                           .head(10)),
        'sector_totals' : (df.groupby('sector')['amount'].sum()
                           .sort_values(ascending=False)),
        'forecast'      : forecast_data,
        'daily_clusters': daily_clusters,
        'pivot'         : pivot,
    }
