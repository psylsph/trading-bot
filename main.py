import pandas as pd
import yfinance as yf
import pandas_ta as ta
from backtesting import Strategy, Backtest # Corrected import
import plotly.graph_objects as go
# from plotly.subplots import make_subplots # Was not used
# from datetime import datetime # yfinance handles datetime objects
import numpy as np

# This function is used for pre-calculating 'TotalSignal' column on the main DataFrame.
# It now takes df_ref and column names as arguments for better reusability,
# though in this script it will still rely on the main 'df' being correctly prepared.
def calculate_total_signal_for_row(row_index, df_reference, bbl_col_name, bbu_col_name):
    """
    Calculates the combined signal for a given row.
    Accesses columns using ['ColumnName'] for clarity and safety.
    """
    # Initial signals for indicators might be NaN, comparisons with NaN are False, so it's handled.
    if (df_reference['VWAPSignal'][row_index] == 2 and
        df_reference['Close'][row_index] <= df_reference[bbl_col_name][row_index] and
        df_reference['RSI'][row_index] < 45):
        return 2 # Buy condition
    if (df_reference['VWAPSignal'][row_index] == 1 and
        df_reference['Close'][row_index] >= df_reference[bbu_col_name][row_index] and
        df_reference['RSI'][row_index] > 55):
        return 1 # Sell condition
    return 0 # No signal / Hold

def pointposbreak(row_series): # x is a row (pd.Series)
    """Determines plotting position for signals."""
    # Ensure column names here match the DataFrame
    if row_series['TotalSignal'] == 1: # Sell signal indication
        return row_series['High'] + 1e-4 # Point above high
    elif row_series['TotalSignal'] == 2: # Buy signal indication
        return row_series['Low'] - 1e-4  # Point below low
    else:
        return np.nan

# The old global SIGNAL() function is removed.
# Signal logic is now incorporated into MyStrat.init using self.I with a custom function.

class MyStrat(Strategy):
    initsize = 0.99 # Initial proportion of cash to use for trade size
    # mysize = initsize # This was redundant, self.initsize is used by backtesting.py by default for size

    # Strategy parameters (can be tuned via optimization)
    rsi_buy_lvl = 45
    rsi_sell_lvl = 55
    rsi_exit_long_lvl = 90 
    rsi_exit_short_lvl = 10
    atr_sl_multiplier = 1.2
    tp_sl_ratio = 1.5
    
    def init(self):
        super().init()

        # Custom indicator function to generate signals for backtesting.py
        # This function is called by self.I and gets passed numpy arrays.
        def strat_custom_signal_indicator(vwap_signal_arr, close_arr, bbl_arr, bbu_arr, rsi_arr):
            # Ensure inputs are numpy arrays (backtesting.py usually handles this)
            vwap_signal_arr = np.asarray(vwap_signal_arr)
            close_arr = np.asarray(close_arr)
            bbl_arr = np.asarray(bbl_arr)
            bbu_arr = np.asarray(bbu_arr)
            rsi_arr = np.asarray(rsi_arr)
            
            signal = np.zeros_like(close_arr, dtype=int)
            
            # Buy Signal (2)
            buy_condition = (vwap_signal_arr == 2) & (close_arr <= bbl_arr) & (rsi_arr < self.rsi_buy_lvl)
            signal[buy_condition] = 2
            
            # Sell Signal (1)
            sell_condition = (vwap_signal_arr == 1) & (close_arr >= bbu_arr) & (rsi_arr > self.rsi_sell_lvl)
            signal[sell_condition] = 1
            
            return signal

        # Pass the required data series to the custom indicator function.
        # 'VWAPSignal', 'RSI', 'BBL', 'BBU', 'ATR' must be columns in the DataFrame passed to Backtest.
        # self.data.Close is a standard attribute. For custom columns, use self.data.df['ColumnName'].
        self.signal1 = self.I(strat_custom_signal_indicator,
                              self.data.df['VWAPSignal'],
                              self.data.Close, 
                              self.data.df['BBL'], # Renamed from BBL_14_2.0
                              self.data.df['BBU'], # Renamed from BBU_14_2.0
                              self.data.df['RSI'])
        
        # ATR is assumed to be pre-calculated and available as self.data.df['ATR']

    def next(self):
        super().next()

        # Ensure 'ATR' and 'RSI' columns exist and have valid data for the current step
        if 'ATR' not in self.data.df.columns or 'RSI' not in self.data.df.columns:
            # This check is more for setup; during iteration, data should be there.
            return 
        if len(self.data.Close) < 1 or \
           pd.isna(self.data.df['ATR'][-1]) or \
           pd.isna(self.data.df['RSI'][-1]):
            return # Not enough data or NaN values for current calculations

        slatr = self.atr_sl_multiplier * self.data.df['ATR'][-1]
        
        # Exit logic based on RSI (applied to the most recent trade if multiple allowed)
        # Iterate over a copy of self.trades list for safe closing of trades
        for trade in list(self.trades): 
            if trade.is_long and self.data.df['RSI'][-1] >= self.rsi_exit_long_lvl:
                trade.close()
            elif trade.is_short and self.data.df['RSI'][-1] <= self.rsi_exit_short_lvl:
                trade.close()
        
        current_signal_value = self.signal1[-1] # Get latest signal value

        # Entry logic (only if no other trades are open for this strategy configuration)
        if len(self.trades) == 0: 
            if current_signal_value == 2: # Buy signal
                sl1 = self.data.Close[-1] - slatr
                tp1 = self.data.Close[-1] + slatr * self.tp_sl_ratio
                self.buy(sl=sl1, tp=tp1, size=self.initsize) # Use self.initsize
            
            elif current_signal_value == 1: # Sell signal     
                sl1 = self.data.Close[-1] + slatr
                tp1 = self.data.Close[-1] - slatr * self.tp_sl_ratio
                self.sell(sl=sl1, tp=tp1, size=self.initsize) # Use self.initsize

if __name__ == '__main__':

    # Use a valid past date range for yfinance
    df = yf.download("EURUSD=X", start="2023-01-01", end="2023-05-18", interval='5m')

    if df.empty:
        print("Failed to download data from yfinance. Exiting.")
        exit()

    df.reset_index(inplace=True) # 'Datetime' (or 'Date') becomes a column
    current_time_column = 'Datetime' # yfinance typically names this 'Datetime' for intraday

    # Filter out rows with no price change (optional, but good practice)
    df = df[df['High'] != df['Low']]
    df.reset_index(drop=True, inplace=True)

    if len(df) < 50:
        print(f"Not enough data after initial filtering. Only {len(df)} rows. Exiting.")
        exit()

    # --- Ensure numeric types and handle potential None/non-numeric values ---
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            # Convert to numeric, forcing errors (like None or strings) to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Ensure float type as pandas_ta often expects floats
            df[col] = df[col].astype(float)
        else:
            print(f"Critical Error: Column '{col}' not found in DataFrame. yfinance might have changed column names.")
            exit()

    # --- Drop rows if essential HLCV data for VWAP is NaN AFTER conversion ---
    # These columns are essential for ta.vwap and other indicators
    df.dropna(subset=['High', 'Low', 'Close', 'Volume'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 20: # Need some data to proceed
        print(f"Not enough data after coercing to numeric and dropping NaNs from HLCV. Only {len(df)} rows. Exiting.")
        exit()

    # --- Calculate indicators ---
    # Now df['High'], df['Low'], df['Close'], df['Volume'] should be clean float Series
    df["VWAP"] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['RSI'] = ta.rsi(close=df['Close'], length=16)

    bbands_length = 14
    bbands_std = 2.0
    my_bbands = ta.bbands(close=df['Close'], length=bbands_length, std=bbands_std)

    bbl_col_name_orig = f'BBL_{bbands_length}_{bbands_std}'
    bbu_col_name_orig = f'BBU_{bbands_length}_{bbands_std}'
    # Ensure my_bbands is not empty and columns exist before trying to access/rename
    if my_bbands is not None and not my_bbands.empty and \
       bbl_col_name_orig in my_bbands.columns and \
       bbu_col_name_orig in my_bbands.columns:
        df = pd.concat([df, my_bbands[[bbl_col_name_orig, bbu_col_name_orig]]], axis=1)
        df.rename(columns={bbl_col_name_orig: 'BBL', bbu_col_name_orig: 'BBU'}, inplace=True)
    else:
        print(f"Warning: Bollinger Bands calculation failed or returned unexpected columns. Skipping BBands.")
        # Add NaN columns so the rest of the script doesn't fail if it expects them
        df['BBL'] = np.nan
        df['BBU'] = np.nan


    # --- VWAPSignal Calculation ---
    VWAPsignal = [0] * len(df)
    backcandles = 15

    # Ensure VWAP column exists and is not all NaN before using it
    if 'VWAP' not in df.columns or df['VWAP'].isnull().all():
        print("VWAP column is missing or all NaN. VWAPSignal cannot be calculated. Exiting.")
        # Or fill VWAPSignal with 0 and continue if appropriate for the strategy
        df['VWAPSignal'] = 0 
        # Potentially exit or handle this state, as signals will be affected
    else:
        # Fill initial NaNs in VWAP if they affect the loop starting at 'backcandles'
        # This is important if VWAP has leading NaNs longer than 'backcandles'
        df['VWAP'].fillna(method='bfill', inplace=True) # Backfill to avoid issues in early loop iterations
        df['VWAP'].fillna(method='ffill', inplace=True) # Forwardfill any remaining at the very start

        for row_idx in range(backcandles, len(df)):
            all_candles_entirely_above_vwap = True
            all_candles_entirely_below_vwap = True
            # Check if VWAP at current or lookback indices is NaN
            if df['VWAP'][row_idx - backcandles : row_idx + 1].isnull().any():
                # If there's a NaN in VWAP for the lookback window, this signal point is unreliable
                # You might skip this row_idx or ensure VWAP is filled
                VWAPsignal[row_idx] = 0 # Default to no signal if VWAP data is missing
                continue # Skip to next row_idx

            for i in range(row_idx - backcandles, row_idx + 1):
                if df['Low'][i] <= df['VWAP'][i]:
                    all_candles_entirely_above_vwap = False
                if df['High'][i] >= df['VWAP'][i]:
                    all_candles_entirely_below_vwap = False
                # Optimization: if both become false, no need to check further for this window
                if not all_candles_entirely_above_vwap and not all_candles_entirely_below_vwap:
                    break

            if all_candles_entirely_above_vwap:
                VWAPsignal[row_idx] = 2
            elif all_candles_entirely_below_vwap:
                VWAPsignal[row_idx] = 1
        df['VWAPSignal'] = VWAPsignal

    # Drop rows with NaNs that might have been generated by indicator calculations (at the start of series)
    # These columns are essential for the strategy signals
    essential_indicator_cols = ['VWAP', 'RSI', 'BBL', 'BBU', 'VWAPSignal']
    # Add other essential columns if any, e.g. 'ATR' will be added later to dfpl_backtest
    
    # Before dropping, check if columns exist to avoid KeyErrors
    cols_to_check_for_nan_drop = [col for col in essential_indicator_cols if col in df.columns]
    if cols_to_check_for_nan_drop:
         df.dropna(subset=cols_to_check_for_nan_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)


    if len(df) < backcandles + 20:
        print(f"Not enough data after indicator calculation and NaN drop. Only {len(df)} rows. Exiting.")
        exit()

    # --- TotalSignal Calculation ---
    TotSignal = [0] * len(df)
    # Check if BBL and BBU columns exist before using them
    if 'BBL' in df.columns and 'BBU' in df.columns:
        for row_idx in range(len(df)):
            # Check for NaN in required fields for calculate_total_signal_for_row
            if pd.isna(df['VWAPSignal'][row_idx]) or \
               pd.isna(df['Close'][row_idx]) or \
               pd.isna(df['BBL'][row_idx]) or \
               pd.isna(df['BBU'][row_idx]) or \
               pd.isna(df['RSI'][row_idx]):
                TotSignal[row_idx] = 0 # Or handle as per strategy logic
            else:
                TotSignal[row_idx] = calculate_total_signal_for_row(row_idx, df, 'BBL', 'BBU')
    else:
        print("Warning: BBL or BBU columns missing. TotalSignal will be all zeros.")
        # TotSignal is already initialized to zeros

    df['TotalSignal'] = TotSignal

    # --- Pointposbreak for plotting signals ---
    df['pointposbreak'] = df.apply(lambda row: pointposbreak(row), axis=1)

    # --- Plotting a sample of the data ---
    # (Plotting code remains largely the same, ensure dfpl_plot has data)
    plot_start_idx = max(0, len(df) - 1000) # Plot recent data or from start if not much data
    plot_length = 350
    if len(df) > plot_start_idx + 10 : # Ensure there's enough data to plot
        dfpl_plot = df.iloc[plot_start_idx : plot_start_idx + plot_length].copy() # Use iloc for position based
        dfpl_plot.reset_index(drop=True, inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=dfpl_plot.index,
                                     open=dfpl_plot['Open'], high=dfpl_plot['High'],
                                     low=dfpl_plot['Low'], close=dfpl_plot['Close'],
                                     name="Candlestick"))
        if 'VWAP' in dfpl_plot.columns:
            fig.add_trace(go.Scatter(x=dfpl_plot.index, y=dfpl_plot['VWAP'],
                                 line=dict(color='blue', width=1), name="VWAP"))
        if 'BBL' in dfpl_plot.columns:
            fig.add_trace(go.Scatter(x=dfpl_plot.index, y=dfpl_plot['BBL'],
                                 line=dict(color='orange', width=1), name="BBL"))
        if 'BBU' in dfpl_plot.columns:
            fig.add_trace(go.Scatter(x=dfpl_plot.index, y=dfpl_plot['BBU'],
                                 line=dict(color='orange', width=1), name="BBU"))

        if 'pointposbreak' in dfpl_plot.columns and 'TotalSignal' in dfpl_plot.columns:
            buy_signals_plot = dfpl_plot[dfpl_plot['TotalSignal'] == 2]
            fig.add_trace(go.Scatter(x=buy_signals_plot.index, y=buy_signals_plot['pointposbreak'],
                                     mode="markers", marker=dict(size=10, color="green", symbol='triangle-up'),
                                     name="Buy Signal"))
            sell_signals_plot = dfpl_plot[dfpl_plot['TotalSignal'] == 1]
            fig.add_trace(go.Scatter(x=sell_signals_plot.index, y=sell_signals_plot['pointposbreak'],
                                     mode="markers", marker=dict(size=10, color="red", symbol='triangle-down'),
                                     name="Sell Signal"))

        fig.update_layout(title="EURUSD 5m Chart with VWAP, BBands, and Signals",
                          xaxis_title="Index", yaxis_title="Price", xaxis_rangeslider_visible=False)
        fig.show()
    else:
        print(f"Not enough data for plotting sample. Available: {len(df)}, Required start: {plot_start_idx + 10}")


    # --- Prepare data for Backtesting ---
    dfpl_backtest = df.copy()

    if len(dfpl_backtest) < 50:
        print(f"Not enough data for backtesting: {len(dfpl_backtest)} rows. Exiting.")
        exit()

    dfpl_backtest['ATR'] = ta.atr(high=dfpl_backtest['High'], low=dfpl_backtest['Low'],
                                  close=dfpl_backtest['Close'], length=7)
    dfpl_backtest.dropna(inplace=True) # Drop NaNs from ATR and any other remaining
    dfpl_backtest.reset_index(drop=True, inplace=True)


    if current_time_column in dfpl_backtest.columns:
        dfpl_backtest[current_time_column] = pd.to_datetime(dfpl_backtest[current_time_column])
        dfpl_backtest.set_index(current_time_column, inplace=True)
    else:
        print(f"Warning: Time column '{current_time_column}' not found for setting index for backtest.")

    if dfpl_backtest.empty or len(dfpl_backtest.index) < 20:
        print("DataFrame for backtesting is empty or too small after final processing. Exiting.")
        exit()

    print(f"Starting backtest with {len(dfpl_backtest)} rows of data...")
    # For debugging, check columns passed to backtest:
    # print("Columns in dfpl_backtest for Backtest:", dfpl_backtest.columns)
    # print(dfpl_backtest.head())
    # print(dfpl_backtest.info())


    bt = Backtest(dfpl_backtest, MyStrat, cash=10000, margin=1/10, commission=0.0002)
    stats = bt.run()
    print(stats)

    try:
        bt.plot()
    except Exception as e:
        print(f"Error during bt.plot(): {e}")