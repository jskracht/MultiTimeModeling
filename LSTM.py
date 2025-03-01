import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from fredapi import Fred
fred = Fred(api_key="a8b743935c751548fe996ee153f3fbde")

DATA_CACHE_FILE = 'fred_data_cache.csv'
RATE_LIMIT_DELAY = 0.5

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
        cached_data = pd.read_csv(DATA_CACHE_FILE)
        cached_data['date'] = pd.to_datetime(cached_data['date'])
        cached_data.set_index('date', inplace=True)
        
        # Check if we need to update the cache
        if cached_data.index[-1].strftime('%Y-%m-%d') >= end_date:
            print("Cache is up to date!")
            return cached_data
        else:
            print("Cache exists but needs updating...")
    
    all_series = []
    total_features = len(features)
    
    for i, series_id in enumerate(features, 1):
        print(f"Fetching {series_id} ({i}/{total_features})...")
        series = fetch_fred_series(series_id, start_date, end_date)
        all_series.append(series)
    
    dataframe = pd.concat(all_series, axis=1)
    
    # Ensure index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe.index = pd.to_datetime(dataframe.index)
    
    # Save to cache
    print("Saving data to cache...")
    dataframe.to_csv(DATA_CACHE_FILE)
    
    return dataframe

# Pull Raw Data
current_month = datetime.now().strftime('%Y-%m-%d')
start_date = "1970-01-01"

features = ["RECPROUSM156N", "ACOILBRENTEU","ACOILWTICO","AHETPI","AISRSA","AUINSA","AWHAETP","BAA10Y","BUSINV","CANTOT","CBI","CDSP","CES0500000003","CEU0500000002","CEU0500000003","CEU0500000008","CHNTOT","CIVPART","CNP16OV","COMPNFB","COMPRNFB","COMREPUSQ159N","CPIAUCNS","CPIAUCSL","DCOILBRENTEU","DCOILWTICO","DDDM01USA156NWDB","DED1","DED3","DED6","DEXBZUS","DEXCAUS","DEXCHUS","DEXJPUS","DEXKOUS","DEXMXUS","DEXNOUS","DEXSDUS","DEXSFUS","DEXSIUS","DEXSZUS","DEXUSAL","DEXUSEU","DEXUSNZ","DEXUSUK","DGORDER","DGS10","DSPIC96","DSWP1","DSWP10","DSWP2","DSWP3","DSWP30","DSWP4","DSWP5","DSWP7","DTWEXB","DTWEXM","ECIWAG","ECOMNSA","ECOMSA","EECTOT","EMRATIO","ETOTALUSQ176N","EVACANTUSQ176N","FEDFUNDS","FRNTOT","FYFSGDA188S","GASREGCOVM","GASREGCOVW","GASREGM","GASREGW","GCT1501US","GCT1502US","GCT1503US","GERTOT","HCOMPBS","HDTGPDUSQ163N","HOABS","HOANBS","HOUST","HPIPONM226N","HPIPONM226S","IC4WSA","INDPRO","INTDSRUSM193N","IPBUSEQ","IPDBS","IPMAN","IPMAT","IPMINE","IR","IR10010","IREXPET","ISRATIO","JCXFE","JPNTOT","JTS1000HIL","JTS1000HIR","JTSHIL","JTSHIR","JTSJOL","JTSJOR","JTSLDL","JTSLDR","JTSQUL","JTSQUR","JTSTSL","JTSTSR","JTU1000HIL","JTU1000HIR","JTUHIL","JTUHIR","JTUJOL","JTUJOR","JTULDL","JTULDR","JTUQUL","JTUQUR","JTUTSL","JTUTSR","LNS12032194","LNS12032196","LNS14027660","LNS15000000","LNU05026642","M12MTVUSM227NFWA","M2V","MCOILBRENTEU","MCOILWTICO","MCUMFN","MEHOINUSA646N","MEHOINUSA672N","MFGOPH","MFGPROD","MNFCTRIRNSA","MNFCTRIRSA","MNFCTRMPCSMNSA","MNFCTRMPCSMSA","MNFCTRSMNSA","MNFCTRSMSA","MYAGM2USM052N","MYAGM2USM052S","NILFWJN","NILFWJNN","NMFBAI","NMFBI","NMFEI","NMFEXI","NMFIMI","NMFINI","NMFINSI","NMFNOI","NMFPI","NMFSDI","NROU","NROUST","OPHMFG","OPHNFB","OPHPBS","OUTBS","OUTMS","OUTNFB","PAYEMS","PAYNSA","PCE","PCEPI","PCEPILFE","PCETRIM12M159SFRBDAL","PCETRIM1M158SFRBDAL","PNRESCON","PNRESCONS","POP","POPTHM","PPIACO","PRRESCON","PRRESCONS","PRS30006013", "PRS30006023","PRS84006013","PRS84006023","PRS84006163","PRS84006173","PRS85006023","PRS85006163","PRS85006173","RCPHBS","RETAILIMSA","RETAILIRSA","RETAILMPCSMNSA","RETAILMPCSMSA","RETAILSMNSA","RETAILSMSA","RHORUSQ156N","RIFLPCFANNM","RPI","RRSFS","RSAFS","RSAFSNA","RSAHORUSQ156S","RSEAS","RSFSXMV","RSNSR","RSXFS","T10Y2Y","T10Y3M","T10YFF","T10YIEM","T5YIEM","T5YIFR","TB3SMFFM","TCU","TDSP","TEDRATE","TLCOMCON","TLCOMCONS","TLNRESCON","TLNRESCONS","TLPBLCON","TLPBLCONS","TLPRVCON","TLPRVCONS","TLRESCON","TLRESCONS","TOTBUSIMNSA","TOTBUSIRNSA","TOTBUSMPCIMNSA","TOTBUSMPCIMSA","TOTBUSMPCSMNSA","TOTBUSMPCSMSA","TOTBUSSMNSA","TOTBUSSMSA","TOTDTEUSQ163N","TRFVOLUSM227NFWA","TTLCON","TTLCONS","U4RATE","U4RATENSA","U6RATE","U6RATENSA","UEMPMED","UKTOT","ULCBS","ULCMFG","ULCNFB","UNRATE","USAGDPDEFAISMEI","USAGDPDEFQISMEI","USAGFCFADSMEI","USAGFCFQDSMEI","USAGFCFQDSNAQ","USARECDM","USARGDPC","USASACRAISMEI","USASACRMISMEI","USASACRQISMEI","USPRIV","USRECD","USRECDM","USSLIND","USSTHPI","WCOILBRENTEU","WCOILWTICO","WHLSLRIRNSA","WHLSLRIRSA"]

# Load or fetch data
dataframe = load_or_fetch_data(features, start_date, current_month)

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
dataframe = pd.DataFrame(np_scaled, columns = dataframe.columns)
dataset = dataframe.values

# Split Data Into Training and Test
split = 440
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
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_step_model.add(tf.keras.layers.Dense(1))
multi_step_model.compile(loss='mae', optimizer='adam')

# Train Model
multi_step_model.fit(train_X, train_Y, epochs=30, batch_size=12, validation_data=(test_X, test_Y), verbose=2, shuffle=False)

# Make Test Prediction
prediction_Y = multi_step_model.predict(test_X)
plt.figure(figsize=(12, 6))
plt.plot(test_Y, label='actual')
plt.plot(prediction_Y, label='predicted')
plt.title('Test Set Predictions vs Actual Values')
plt.legend()
plt.show()

def make_future_forecast(model, last_known_values, n_future_steps):
    future_predictions = []
    current_input = last_known_values.reshape(1, 1, -1)  # Reshape to match model input shape
    
    for _ in range(n_future_steps):
        next_pred = model.predict(current_input)
        future_predictions.append(next_pred[0, 0])

        current_input = current_input
    
    return np.array(future_predictions)

# Get the last known values (all features except the target)
last_known_values = all_X[-1]

# Make future predictions
n_future_months = 12
future_predictions = make_future_forecast(multi_step_model, last_known_values, n_future_months)

# Plot historical data and future predictions
plt.figure(figsize=(15, 7))
plt.plot(range(len(all_Y)), all_Y, label='Historical Data', color='blue')
plt.plot(range(len(all_Y), len(all_Y) + n_future_months), future_predictions, 
         label='Future Forecast', color='red', linestyle='--')
plt.axvline(x=len(all_Y)-1, color='gray', linestyle='--', alpha=0.5)
plt.title('Historical Data and Future Forecast')
plt.legend()
plt.show()

# Print the future predictions
print("\nFuture predictions for the next {} months:".format(n_future_months))
for i, pred in enumerate(future_predictions, 1):
    print(f"Month {i}: {pred:.4f}")