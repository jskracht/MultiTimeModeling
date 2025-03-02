import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from fredapi import Fred

FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY environment variable is not set. Please set it before running this script.")

fred = Fred(api_key=FRED_API_KEY)

DATA_CACHE_FILE = 'fred_data_cache.csv'
RATE_LIMIT_DELAY = 0.25

current_month = datetime.now().strftime('%Y-%m-%d')
start_date = "1970-01-01"
end_date = current_month

def fetch_fred_series(series_id, start_date, end_date):
    try:
        time.sleep(RATE_LIMIT_DELAY)
        try:
            data = fred.get_series(series_id, start_date=start_date, end_date=end_date, frequency='m')
        except Exception:
            data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
            data = data.asfreq('ME', method='ffill') 
        return pd.Series(data, name=series_id)
    except Exception as e:
        print(f"Error fetching {series_id}: {str(e)}")
        return pd.Series(name=series_id)

def load_or_fetch_data(features, start_date, end_date):
    if os.path.exists(DATA_CACHE_FILE):
        print("Loading data from local cache...")
        cached_data = pd.read_csv(DATA_CACHE_FILE, index_col=0, parse_dates=True)
        print(f"\nLoaded data date range: {cached_data.index.min()} to {cached_data.index.max()}")
        cached_data = cached_data[start_date:end_date]
        print(f"After filtering: {cached_data.index.min()} to {cached_data.index.max()}\n")
        if cached_data.index[-1] >= (pd.to_datetime(end_date) - pd.DateOffset(months=1)):
            print("Cache is up to date!")
            return clean_and_validate_data(cached_data)
        else:
            print("Cache exists but needs updating...")
    
    all_series = []
    failed_series = []
    total_features = len(features)
    
    for i, series_id in enumerate(features, 1):
        print(f"Fetching {series_id} ({i}/{total_features})...")
        series = fetch_fred_series(series_id, start_date, end_date)
        if series.empty or series.isna().all():
            failed_series.append(series_id)
            continue
        all_series.append(series)
    
    if failed_series:
        print("\nWarning: The following series failed to fetch or contained no data:")
        for series_id in failed_series:
            print(f"- {series_id}")
    
    dataframe = pd.concat(all_series, axis=1)
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe.index = pd.to_datetime(dataframe.index)
    
    dataframe = dataframe.resample('M').mean() 
    print(f"\nCombined dataset date range after resampling: {dataframe.index.min()} to {dataframe.index.max()}")
    dataframe = clean_and_validate_data(dataframe)
    
    print("Saving data to cache...")
    dataframe.to_csv(DATA_CACHE_FILE)
    
    return dataframe

def clean_and_validate_data(df):
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    initial_cols = len(df.columns)
    cleaned_series = []
    
    for col in df.columns:
        series = df[col].copy()
        valid_dates = series.dropna().index
        if not valid_dates.empty:                     
            series = series.interpolate(method='time', limit_direction='both')
            series.ffill(limit=3)
            series.bfill(limit=3)
            if not series.isna().any():
                cleaned_series.append(series)
                print(f"Series {col}: {start_date} to {end_date}")
            else:
                print(f"Dropping {col} due to remaining NaN values")
    
    cleaned_df = pd.concat(cleaned_series, axis=1)
    final_cols = len(cleaned_df.columns)
    if initial_cols != final_cols:
        print(f"\nRemoved {initial_cols - final_cols} problematic columns. {final_cols} columns remaining.")
    
    return cleaned_df

def make_future_forecast(model, last_known_values, n_future_steps):
    future_predictions = []
    current_input = last_known_values.reshape(1, 1, -1)
    
    for _ in range(n_future_steps):
        next_pred = model.predict(current_input, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_input = current_input.copy()
    
    return np.array(future_predictions)

features = [
    "RECPROUSM156N", "ACOILBRENTEU", "ACOILWTICO", "AHETPI", "AISRSA", "AUINSA", 
    "AWHAETP", "BAA10Y", "BUSINV", "CANTOT", "CBI", "CDSP", "CES0500000003", 
    "CEU0500000002", "CEU0500000003", "CEU0500000008", "CHNTOT", "CIVPART", 
    "CNP16OV", "COMPNFB", "COMPRNFB", "COMREPUSQ159N", "CPIAUCNS", "CPIAUCSL", 
    "DCOILBRENTEU", "DCOILWTICO", "DDDM01USA156NWDB", "DED1", "DED3", "DED6", 
    "DEXBZUS", "DEXCAUS", "DEXCHUS", "DEXJPUS", "DEXKOUS", "DEXMXUS", "DEXNOUS", 
    "DEXSDUS", "DEXSFUS", "DEXSIUS", "DEXSZUS", "DEXUSAL", "DEXUSEU", "DEXUSNZ", 
    "DEXUSUK", "DGORDER", "DGS10", "DSPIC96", "DSWP1", "DSWP10", "DSWP2", 
    "DSWP3", "DSWP30", "DSWP4", "DSWP5", "DSWP7", "DTWEXB", "DTWEXM", "ECIWAG", 
    "ECOMNSA", "ECOMSA", "EECTOT", "EMRATIO", "ETOTALUSQ176N", "EVACANTUSQ176N", 
    "FEDFUNDS", "FRNTOT", "FYFSGDA188S", "GASREGCOVM", "GASREGCOVW", "GASREGM", 
    "GASREGW", "GERTOT", "HCOMPBS", "HDTGPDUSQ163N", "HOABS", "HOANBS", "HOUST", 
    "HPIPONM226N", "HPIPONM226S", "IC4WSA", "INDPRO", "INTDSRUSM193N", "IPBUSEQ", 
    "IPDBS", "IPMAN", "IPMAT", "IPMINE", "IR", "IR10010", "IREXPET", "ISRATIO", 
    "JCXFE", "JPNTOT", "JTS1000HIL", "JTS1000HIR", "JTSHIL", "JTSHIR", "JTSJOL", 
    "JTSJOR", "JTSLDL", "JTSLDR", "JTSQUL", "JTSQUR", "JTSTSL", "JTSTSR", 
    "JTU1000HIL", "JTU1000HIR", "JTUHIL", "JTUHIR", "JTUJOL", "JTUJOR", 
    "JTULDL", "JTULDR", "JTUQUL", "JTUQUR", "JTUTSL", "JTUTSR", "LNS12032194", 
    "LNS12032196", "LNS14027660", "LNS15000000", "LNU05026642", "M12MTVUSM227NFWA", 
    "M2V", "MCOILBRENTEU", "MCOILWTICO", "MCUMFN", "MEHOINUSA646N", "MEHOINUSA672N", 
    "MFGOPH", "MFGPROD", "MNFCTRIRNSA", "MNFCTRIRSA", "MNFCTRMPCSMNSA", 
    "MNFCTRMPCSMSA", "MNFCTRSMNSA", "MNFCTRSMSA", "MYAGM2USM052N", "MYAGM2USM052S", 
    "NILFWJN", "NILFWJNN", "NROU", "NROUST", "OPHMFG", "OPHNFB", "OPHPBS", 
    "OUTBS", "OUTMS", "OUTNFB", "PAYEMS", "PAYNSA", "PCE", "PCEPI", "PCEPILFE", 
    "PCETRIM12M159SFRBDAL", "PCETRIM1M158SFRBDAL", "PNRESCON", "PNRESCONS", 
    "POP", "POPTHM", "PPIACO", "PRRESCON", "PRRESCONS", "PRS30006013", 
    "PRS30006023", "PRS84006013", "PRS84006023", "PRS84006163", "PRS84006173", 
    "PRS85006023", "PRS85006163", "PRS85006173", "RCPHBS", "RETAILIMSA", 
    "RETAILIRSA", "RETAILMPCSMNSA", "RETAILMPCSMSA", "RETAILSMNSA", "RETAILSMSA", 
    "RHORUSQ156N", "RIFLPCFANNM", "RPI", "RRSFS", "RSAFS", "RSAFSNA", 
    "RSAHORUSQ156S", "RSEAS", "RSFSXMV", "RSNSR", "RSXFS", "T10Y2Y", "T10Y3M", 
    "T10YFF", "T10YIEM", "T5YIEM", "T5YIFR", "TB3SMFFM", "TCU", "TDSP", 
    "TEDRATE", "TLCOMCON", "TLCOMCONS", "TLNRESCON", "TLNRESCONS", "TLPBLCON", 
    "TLPBLCONS", "TLPRVCON", "TLPRVCONS", "TLRESCON", "TLRESCONS", "TOTBUSIMNSA", 
    "TOTBUSIRNSA", "TOTBUSMPCIMNSA", "TOTBUSMPCIMSA", "TOTBUSMPCSMNSA", 
    "TOTBUSMPCSMSA", "TOTBUSSMNSA", "TOTBUSSMSA", "TOTDTEUSQ163N", 
    "TRFVOLUSM227NFWA", "TTLCON", "TTLCONS", "U4RATE", "U4RATENSA", "U6RATE", 
    "U6RATENSA", "UEMPMED", "UKTOT", "ULCBS", "ULCMFG", "ULCNFB", "UNRATE", 
    "USAGDPDEFAISMEI", "USAGDPDEFQISMEI", "USAGFCFADSMEI", "USAGFCFQDSMEI", 
    "USAGFCFQDSNAQ", "USARECDM", "USARGDPC", "USASACRAISMEI", "USASACRMISMEI", 
    "USASACRQISMEI", "USPRIV", "USRECD", "USRECDM", "USSLIND", "USSTHPI", 
    "WCOILBRENTEU", "WCOILWTICO", "WHLSLRIRNSA", "WHLSLRIRSA"
]

dataframe = load_or_fetch_data(features, start_date, end_date)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(dataframe)
dataframe = pd.DataFrame(np_scaled, columns=dataframe.columns)


split = int(len(dataframe.values) * 0.8)
train = dataframe.values[:split, :]
test = dataframe.values[split:, :]
all_X = dataframe.values[:, 1:]
all_Y = dataframe.values[:, 0]

# Split Data Into Inputs and Output
train_X, train_Y = train[:, 1:], train[:, 0]
test_X, test_Y = test[:, 1:], test[:, 0]

# Reshape Input to be 3D [Samples, Timesteps, Features]
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

# Build Model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
multi_step_model.add(tf.keras.layers.Dropout(0.2))
multi_step_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
multi_step_model.compile(loss='mae', optimizer='adam')

# Train Model
multi_step_model.fit(train_X, train_Y, epochs=100, batch_size=12, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

# After making predictions
prediction_Y = multi_step_model.predict(test_X)

# Create a full prediction array to hold all values
full_prediction = np.full_like(all_Y, np.nan, dtype=np.float32)
full_prediction[len(train_Y):len(train_Y) + len(prediction_Y)] = prediction_Y.flatten()

plt.figure(figsize=(15, 7))
plt.plot(dataframe.index, all_Y * 100, label='Actual', color='blue')
plt.plot(dataframe.index[:len(train_Y)], train_Y * 100, label='Training', color='green') 
plt.plot(dataframe.index[len(train_Y):len(train_Y) + len(prediction_Y)], full_prediction[len(train_Y):] * 100, label='Prediction', color='red', linestyle='--')
plt.title('Predictions vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Probability of Recession (%)')
plt.legend()
plt.show()


last_known_values = test_X[-1] 
n_future_months = 3
future_predictions = make_future_forecast(multi_step_model, last_known_values, n_future_months)

num_rows = len(dataframe)
date_range = pd.date_range(start=start_date, periods=num_rows, freq='ME')
dataframe.index = date_range
last_date = pd.to_datetime(dataframe.index[-1])
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                              periods=n_future_months, 
                              freq='ME')

plt.figure(figsize=(15, 7))
plt.plot(dataframe.index[-len(test_Y):], test_Y * 100, label='Actual', color='blue')
plt.plot(future_dates, future_predictions * 100, label='Forecast', color='red', linestyle='--')
plt.title('Forecast')
plt.xlabel('Date')
plt.ylabel('Probability of Recession (%)')
plt.legend()
plt.show()

print("\nFuture predictions for the next {} months:".format(n_future_months))
for date, pred in zip(future_dates, future_predictions):
    print(f"{date.strftime('%Y-%m')}: {pred*100:.2f}%")