# Data from https://www.kaggle.com/datasets/statmunch/5-minute-candlestick-crypto-data

import pandas as pd
import os

BASE_PATH = os.path.abspath(os.path.join(__file__,'..','..'))

DATA_PATH = os.path.join(BASE_PATH,'data','crypto-5-min')

def read_usdt_csv_files(folder_path):
    first = True
    for filename in os.listdir(folder_path):
        if "USDT" in filename and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            asset = filename.split('USDT')[0]
            df = pd.read_csv(file_path)
            # The file with the close price will take the name of the asset
            column_names = ['unix_timestamp','open','high','low',asset,'volume','close_time','quote_asset_volume','numer_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','NA']
            df.columns = column_names
            df = df[['close_time',asset]]
            df['close_time'] = pd.to_datetime(df['close_time'],unit='ms')
            df.index = df['close_time']
            df = df[[asset]]

            if first:
                df_complete = df.copy()
                first = False
            else:
                df_complete = df_complete.merge(df,left_index=True,right_index=True,how='outer')
    
    return df_complete

df_complete = read_usdt_csv_files(DATA_PATH)
df_complete.to_csv(os.path.join(DATA_PATH,'complete_close_price.csv'))
pass