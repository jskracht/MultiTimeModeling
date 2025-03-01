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

# Function to fetch data for a single series with rate limiting
def fetch_fred_series(series_id, start_date, end_date):
    try:
        time.sleep(RATE_LIMIT_DELAY)
        
        # First try monthly frequency
        try:
            data = fred.get_series(series_id, start_date=start_date, end_date=end_date, frequency='m')
        except Exception as freq_error:
            data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
            data = data.asfreq('ME', method='ffill') 
            # If there are still NaN values, interpolate them
            data = data.interpolate(method='time')

        return pd.Series(data, name=series_id)
    except Exception as e:
        print(f"Error fetching {series_id}: {str(e)}")
        return pd.Series(name=series_id)

def load_or_fetch_data(features, start_date, end_date):
    if os.path.exists(DATA_CACHE_FILE):
        print("Loading data from local cache...")
        try:
            # First try loading with date as index
            cached_data = pd.read_csv(DATA_CACHE_FILE, index_col=0, parse_dates=True)
        except Exception:
            # If that fails, try loading with date as a column
            cached_data = pd.read_csv(DATA_CACHE_FILE)
            if 'date' in cached_data.columns:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                cached_data.set_index('date', inplace=True)
            else:
                raise ValueError("Cache file does not contain date information")
        
        print(f"\nLoaded data date range: {cached_data.index.min()} to {cached_data.index.max()}")
        
        # Filter data by date range
        cached_data = cached_data[start_date:end_date]
        print(f"After filtering: {cached_data.index.min()} to {cached_data.index.max()}\n")
        
        # Check if we need to update the cache
        if cached_data.index[-1].strftime('%Y-%m-%d') >= end_date:
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
    
    # Ensure index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe.index = pd.to_datetime(dataframe.index)
    
    print(f"\nCombined dataset date range: {dataframe.index.min()} to {dataframe.index.max()}")
    
    # Clean and validate before saving
    dataframe = clean_and_validate_data(dataframe)
    
    # Save to cache with date as index
    print("Saving data to cache...")
    dataframe.to_csv(DATA_CACHE_FILE)
    
    return dataframe

def clean_and_validate_data(df):
    # Filter to only include data from 1970 to present
    current_date = pd.Timestamp.now()
    df = df[(df.index >= '1970-01-01') & (df.index <= current_date)]
    
    initial_cols = len(df.columns)
    cleaned_series = []
    
    for col in df.columns:
        series = df[col].copy()
        # Find valid date range for this series
        valid_dates = series.dropna().index
        if not valid_dates.empty:
            start_date = valid_dates[0]
            end_date = valid_dates[-1]
            
            # Only keep data between first and last valid dates
            series = series[start_date:end_date]
            
            # Interpolate missing values within the series' own date range
            series = series.interpolate(method='time', limit_direction='both')
            
            # Forward and backward fill any remaining gaps (limited to 3 months)
            series = series.fillna(method='ffill', limit=3)
            series = series.fillna(method='bfill', limit=3)
            
            if not series.isna().any():  # Only keep series with no remaining NaNs
                cleaned_series.append(series)
                print(f"Series {col}: {start_date} to {end_date}")
            else:
                print(f"Dropping {col} due to remaining NaN values")
    
    # Combine all cleaned series
    cleaned_df = pd.concat(cleaned_series, axis=1)
    
    final_cols = len(cleaned_df.columns)
    if initial_cols != final_cols:
        print(f"\nRemoved {initial_cols - final_cols} problematic columns. {final_cols} columns remaining.")
    
    return cleaned_df

# Pull Raw Data
current_month = datetime.now().strftime('%Y-%m-%d')
start_date = "1970-01-01"
end_date = current_month

# Define features list
features = ["RECPROUSM156N", "ACOILBRENTEU","ACOILWTICO","AHETPI","AISRSA","AUINSA","AWHAETP","BAA10Y","BUSINV","CANTOT","CBI","CDSP","CES0500000003","CEU0500000002","CEU0500000003","CEU0500000008","CHNTOT","CIVPART","CNP16OV","COMPNFB","COMPRNFB","COMREPUSQ159N","CPIAUCNS","CPIAUCSL","DCOILBRENTEU","DCOILWTICO","DDDM01USA156NWDB","DED1","DED3","DED6","DEXBZUS","DEXCAUS","DEXCHUS","DEXJPUS","DEXKOUS","DEXMXUS","DEXNOUS","DEXSDUS","DEXSFUS","DEXSIUS","DEXSZUS","DEXUSAL","DEXUSEU","DEXUSNZ","DEXUSUK","DGORDER","DGS10","DSPIC96","DSWP1","DSWP10","DSWP2","DSWP3","DSWP30","DSWP4","DSWP5","DSWP7","DTWEXB","DTWEXM","ECIWAG","ECOMNSA","ECOMSA","EECTOT","EMRATIO","ETOTALUSQ176N","EVACANTUSQ176N","FEDFUNDS","FRNTOT","FYFSGDA188S","GASREGCOVM","GASREGCOVW","GASREGM","GASREGW","GERTOT","HCOMPBS","HDTGPDUSQ163N","HOABS","HOANBS","HOUST","HPIPONM226N","HPIPONM226S","IC4WSA","INDPRO","INTDSRUSM193N","IPBUSEQ","IPDBS","IPMAN","IPMAT","IPMINE","IR","IR10010","IREXPET","ISRATIO","JCXFE","JPNTOT","JTS1000HIL","JTS1000HIR","JTSHIL","JTSHIR","JTSJOL","JTSJOR","JTSLDL","JTSLDR","JTSQUL","JTSQUR","JTSTSL","JTSTSR","JTU1000HIL","JTU1000HIR","JTUHIL","JTUHIR","JTUJOL","JTUJOR","JTULDL","JTULDR","JTUQUL","JTUQUR","JTUTSL","JTUTSR","LNS12032194","LNS12032196","LNS14027660","LNS15000000","LNU05026642","M12MTVUSM227NFWA","M2V","MCOILBRENTEU","MCOILWTICO","MCUMFN","MEHOINUSA646N","MEHOINUSA672N","MFGOPH","MFGPROD","MNFCTRIRNSA","MNFCTRIRSA","MNFCTRMPCSMNSA","MNFCTRMPCSMSA","MNFCTRSMNSA","MNFCTRSMSA","MYAGM2USM052N","MYAGM2USM052S","NILFWJN","NILFWJNN","NROU","NROUST","OPHMFG","OPHNFB","OPHPBS","OUTBS","OUTMS","OUTNFB","PAYEMS","PAYNSA","PCE","PCEPI","PCEPILFE","PCETRIM12M159SFRBDAL","PCETRIM1M158SFRBDAL","PNRESCON","PNRESCONS","POP","POPTHM","PPIACO","PRRESCON","PRRESCONS","PRS30006013", "PRS30006023","PRS84006013","PRS84006023","PRS84006163","PRS84006173","PRS85006023","PRS85006163","PRS85006173","RCPHBS","RETAILIMSA","RETAILIRSA","RETAILMPCSMNSA","RETAILMPCSMSA","RETAILSMNSA","RETAILSMSA","RHORUSQ156N","RIFLPCFANNM","RPI","RRSFS","RSAFS","RSAFSNA","RSAHORUSQ156S","RSEAS","RSFSXMV","RSNSR","RSXFS","T10Y2Y","T10Y3M","T10YFF","T10YIEM","T5YIEM","T5YIFR","TB3SMFFM","TCU","TDSP","TEDRATE","TLCOMCON","TLCOMCONS","TLNRESCON","TLNRESCONS","TLPBLCON","TLPBLCONS","TLPRVCON","TLPRVCONS","TLRESCON","TLRESCONS","TOTBUSIMNSA","TOTBUSIRNSA","TOTBUSMPCIMNSA","TOTBUSMPCIMSA","TOTBUSMPCSMNSA","TOTBUSMPCSMSA","TOTBUSSMNSA","TOTBUSSMSA","TOTDTEUSQ163N","TRFVOLUSM227NFWA","TTLCON","TTLCONS","U4RATE","U4RATENSA","U6RATE","U6RATENSA","UEMPMED","UKTOT","ULCBS","ULCMFG","ULCNFB","UNRATE","USAGDPDEFAISMEI","USAGDPDEFQISMEI","USAGFCFADSMEI","USAGFCFQDSMEI","USAGFCFQDSNAQ","USARECDM","USARGDPC","USASACRAISMEI","USASACRMISMEI","USASACRQISMEI","USPRIV","USRECD","USRECDM","USSLIND","USSTHPI","WCOILBRENTEU","WCOILWTICO","WHLSLRIRNSA","WHLSLRIRSA"]

def validate_date_range(df, start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Filter data to be within bounds
    mask = (df.index >= start) & (df.index <= end)
    filtered_df = df.loc[mask]
    
    return filtered_df

# Load or fetch data
dataframe = load_or_fetch_data(features, start_date, end_date)

# Validate and filter date range
dataframe = validate_date_range(dataframe, start_date, end_date)

# Convert data types and handle missing values
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
dataframe = dataframe.infer_objects()

# Ensure we have a proper datetime index for interpolation
if not isinstance(dataframe.index, pd.DatetimeIndex):
    dataframe.index = pd.to_datetime(dataframe.index)

# Interpolate Data
dataframe = dataframe.interpolate(method='time', limit_direction='both')

# Normalize Dataset
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(dataframe)
dataframe = pd.DataFrame(np_scaled, columns=dataframe.columns)
dataset = dataframe.values

# Split Data Into Training and Test
split = int(len(dataset) * 0.8)  # Use 80% for training
train = dataset[:split, :]
test = dataset[split:, :]
all_X = dataset[:, 1:]
all_Y = dataset[:, 0]

# Split Data Into Inputs and Output
train_X, train_Y = train[:, 1:], train[:, 0]
test_X, test_Y = test[:, 1:], test[:, 0]

# Reshape Input to be 3D [Samples, Timesteps, Features]
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

# Build Model
multi_step_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(256, input_shape=(1, train_X.shape[2]), return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

multi_step_model.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Print model summary
print("\nModel Summary:")
multi_step_model.summary()

# Train Model
history = multi_step_model.fit(train_X, train_Y, 
                             epochs=30, 
                             batch_size=12,
                             validation_data=(test_X, test_Y),
                             verbose=2,
                             shuffle=False)

prediction_Y = multi_step_model.predict(test_X)
test_dates = dataframe.index[split:]

# Plot the full test set predictions
plt.figure(figsize=(15, 7))
plt.plot(test_dates, test_Y * 100, label='Actual', color='blue')
plt.plot(test_dates, prediction_Y * 100, label='Predicted', color='red', linestyle='--')
plt.title('Full Historical Test Set Predictions vs Actual Values (1970-Present)')
plt.xlabel('Date')
plt.ylabel('Probability of Recession (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def make_future_forecast(model, last_known_values, n_future_steps):
    future_predictions = []
    current_input = last_known_values.reshape(1, 1, -1)  # Shape: (1, 1, features)
    
    for _ in range(n_future_steps):
        next_pred = model.predict(current_input, verbose=0)
        future_predictions.append(next_pred[0, 0])
        
        # Update the input for next prediction - keep the same features
        current_input = current_input.copy()
    
    return np.array(future_predictions)

# Get the last known values - only the features, not the target
last_known_values = test_X[-1] 

# Make future predictions
n_future_months = 3
print("\nMaking future predictions...")
future_predictions = make_future_forecast(multi_step_model, last_known_values, n_future_months)

# Create future dates for plotting
last_date = pd.to_datetime(dataframe.index[-1])
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                           periods=n_future_months, 
                           freq='ME')

print(f"Forecasting from {last_date.strftime('%Y-%m')} to {future_dates[-1].strftime('%Y-%m')}")

# Plot historical data and future predictions
plt.figure(figsize=(15, 7))

# Use all historical data
historical_dates = pd.DatetimeIndex(dataframe.index)
historical_values = all_Y

plt.plot(historical_dates, historical_values * 100, label='Historical Data', color='blue')
plt.plot(future_dates, future_predictions * 100, label='3-Month Forecast', color='red', linestyle='--')
plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5, label='Present')
plt.title('Full Historical Data (1970-Present) and 3-Month Forecast')
plt.xlabel('Date')
plt.ylabel('Probability of Recession (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the future predictions
print("\nFuture predictions for the next {} months:".format(n_future_months))
for date, pred in zip(future_dates, future_predictions):
    print(f"{date.strftime('%Y-%m')}: {pred*100:.2f}%")