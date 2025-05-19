import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta # For technical indicators
from datetime import datetime, timedelta, date # Added date for YESTERDAY
import os
import pytz # For timezone handling
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# --- Configuration ---
# Best practice: Use environment variables for API keys
# For testing, you can hardcode them but REMOVE before committing/sharing
# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets') # Use paper trading URL

SYMBOL = 'XRP/USD'
TIMEFRAME = tradeapi.TimeFrame.Day

# --- MODIFIED PARAMETERS FOR EXPERIMENTATION ---
START_DATE_STR = '2024-01-01' # Focused on the period with known data
YESTERDAY = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
END_DATE_STR = YESTERDAY # Set to yesterday to avoid future data issues

# Strategy Parameters
BB_LENGTH = 20
BB_STD_DEV = 2.0
RSI_LENGTH = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35 # <<<< Experiment 1: Relaxed from 30 to 35
VWAP_STOP_PERCENTAGE_BUFFER = 0.02 # <<<< Experiment 2: 1% buffer (0.01 means price must be 1% below VWAP)

INITIAL_CAPITAL = 10000.00
POSITION_SIZE_USD = 1000.00

# --- Alpaca API Setup ---
try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"Connected to Alpaca. Account Status: {account.status}")
except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    exit()

# --- 1. Fetch Historical Data ---
def get_historical_data(symbol, timeframe, start_str, end_str):
    print(f"Fetching data for {symbol} from {start_str} to {end_str} with timeframe {timeframe.value}")
    try:
        utc = pytz.UTC
        start_dt = utc.localize(datetime.strptime(start_str, '%Y-%m-%d'))
        end_dt_inclusive = utc.localize(datetime.strptime(end_str, '%Y-%m-%d'))
        end_dt_exclusive = end_dt_inclusive + timedelta(days=1)

        bars = api.get_crypto_bars(symbol, timeframe, start=start_dt.isoformat(), end=end_dt_exclusive.isoformat()).df
        
        if bars.empty:
            print("No data fetched from Alpaca.")
            return pd.DataFrame()

        if not bars.index.tz: bars = bars.tz_localize('UTC')
        else: bars = bars.tz_convert('UTC')
        
        bars = bars[bars.index <= end_dt_inclusive.replace(hour=23, minute=59, second=59)]
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if 'vwap' in bars.columns: required_ohlcv.append('vwap')
        bars = bars[required_ohlcv]

        print(f"Fetched {len(bars)} bars. Actual date range in data: {bars.index.min()} to {bars.index.max()}")
        return bars
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

# --- 2. Calculate Indicators ---
def add_indicators(df):
    if df.empty: return df
    print("\n--- Adding Indicators ---")
    df.ta.bbands(length=BB_LENGTH, std=BB_STD_DEV, append=True)
    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'
    upper_band_col = f'BBU_{BB_LENGTH}_{BB_STD_DEV}'
    df.ta.rsi(length=RSI_LENGTH, append=True)
    rsi_col = f'RSI_{RSI_LENGTH}'

    if 'vwap' not in df.columns:
        print("VWAP not in Alpaca data, calculating using pandas_ta...")
        df.ta.vwap(append=True)
        if 'VWAP_D' in df.columns: df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True)
        elif 'VWAP' not in df.columns and any(col.startswith('VWAP_') for col in df.columns):
            vwap_col_pandas_ta = [col for col in df.columns if col.startswith('VWAP_')][0]
            df.rename(columns={vwap_col_pandas_ta: 'VWAP'}, inplace=True)
            print(f"Renamed pandas_ta VWAP column '{vwap_col_pandas_ta}' to 'VWAP'")
    else:
        print("Using VWAP provided by Alpaca.")
        df.rename(columns={'vwap': 'VWAP'}, inplace=True)

    print(f"Columns after adding indicators: {df.columns.tolist()}")
    required_cols_for_signals = ['close', lower_band_col, upper_band_col, middle_band_col, rsi_col, 'VWAP']
    missing_at_indicator_stage = [col for col in required_cols_for_signals if col not in df.columns]
    if missing_at_indicator_stage:
        print(f"ERROR: Missing critical indicator columns BEFORE dropna: {missing_at_indicator_stage}")
    
    df.dropna(inplace=True)
    print(f"Shape of data after adding indicators and dropna: {df.shape}")
    if df.empty: print("DataFrame is empty after adding indicators and dropping NA.")
    return df

# --- 3. Define Strategy & Generate Signals ---
def generate_signals(df):
    print("\n--- Generating Signals ---")
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['sell_reason'] = ''

    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    upper_band_col = f'BBU_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}' # For potential future use
    rsi_col = f'RSI_{RSI_LENGTH}'
    vwap_col = 'VWAP'

    required_cols = [lower_band_col, upper_band_col, middle_band_col, rsi_col, vwap_col, 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns for signal generation: {missing_cols}\nAvailable: {df.columns.tolist()}")
        return signals

    # Buy Conditions
    cond1_buy_lbb = (df['close'] < df[lower_band_col])
    cond2_buy_rsi = (df[rsi_col] < RSI_OVERSOLD) # RSI_OVERSOLD is now 35
    buy_condition = cond1_buy_lbb & cond2_buy_rsi
    
    # Sell Conditions
    cond_sell_profit_upperbb = (df['close'] > df[upper_band_col])
    cond_sell_profit_rsi = (df[rsi_col] > RSI_OVERBOUGHT)
    cond_sell_profit_middlebb = (df['close'] > df[middle_band_col])
    # VWAP stop with buffer
    cond_sell_stop_vwap = (df['close'] < (df[vwap_col] * (1 - VWAP_STOP_PERCENTAGE_BUFFER)))

    # Apply sell signals and reasons with priority
    # Priority: UpperBB > RSI_OB > VWAP_Stop
    # A sell reason is set if any of the primary conditions are met.
    # The 'signal' column is then set to -1 if any sell_reason was populated.

    # 1. Profit Take - Upper Bollinger Band
    signals.loc[cond_sell_profit_upperbb, 'sell_reason'] = 'UpperBB'
    # 2. Profit Take - RSI Overbought (only if not already UpperBB)
    signals.loc[cond_sell_profit_rsi & (signals['sell_reason'] == ''), 'sell_reason'] = 'RSI_OB'
    # 3. Stop Loss - VWAP (only if not already a profit take)
    signals.loc[cond_sell_stop_vwap & (signals['sell_reason'] == ''), 'sell_reason'] = 'VWAP_Stop'

    # Refined sell reason logic with MiddleBB PT
    # Priority: UpperBB > RSI_OB > MiddleBB_PT > VWAP_Stop
    signals.loc[cond_sell_profit_upperbb, 'sell_reason'] = 'UpperBB'
    signals.loc[cond_sell_profit_rsi & (signals['sell_reason'] == ''), 'sell_reason'] = 'RSI_OB'
    signals.loc[cond_sell_profit_middlebb & (signals['sell_reason'] == ''), 'sell_reason'] = 'MiddleBB_PT' # New
    signals.loc[cond_sell_stop_vwap & (signals['sell_reason'] == ''), 'sell_reason'] = 'VWAP_Stop'

    signals.loc[signals['sell_reason'] != '', 'signal'] = -1
    
    # Set signal to -1 if any sell_reason was populated
    signals.loc[signals['sell_reason'] != '', 'signal'] = -1
    
    # Apply buy signals (can override a sell signal on the same bar if logic allows)
    # This ensures buy takes precedence if rare conditions for both buy and sell are met on the same bar.
    signals.loc[buy_condition, 'signal'] = 1
    signals.loc[buy_condition, 'sell_reason'] = '' # Clear sell reason for buy signals

    # --- DEBUGGING PRINTS ---
    print(f"RSI Oversold Threshold: {RSI_OVERSOLD}")
    print(f"VWAP Stop Buffer: {VWAP_STOP_PERCENTAGE_BUFFER*100}%")
    print(f"Number of potential buy signal points (raw conditions met): {buy_condition.sum()}")
    print(f"  Breakdown: LBB condition met: {cond1_buy_lbb.sum()}, RSI condition met: {cond2_buy_rsi.sum()}")
    
    print(f"Number of potential sell triggers (UpperBB): {cond_sell_profit_upperbb.sum()}")
    print(f"Number of potential sell triggers (RSI Overbought): {cond_sell_profit_rsi.sum()}")
    print(f"Number of potential sell triggers (VWAP Stop with buffer): {cond_sell_stop_vwap.sum()}")
    
    actual_buy_signals = signals[signals['signal'] == 1]
    actual_sell_signals = signals[signals['signal'] == -1]
    print(f"Total BUY signals generated: {len(actual_buy_signals)}")
    print(f"Total SELL signals generated (based on sell_reason): {len(signals[signals['sell_reason'] != ''])}") # Count actual potential exits

    if not actual_buy_signals.empty:
        print("Dates of first 5 BUY signals:")
        print(actual_buy_signals.head().index.strftime('%Y-%m-%d').tolist())
    if not actual_sell_signals.empty:
        print("Dates of first 5 SELL signals (and reasons):")
        print(actual_sell_signals[actual_sell_signals['sell_reason'] != ''].head()[['sell_reason']])
    
    return signals

# --- 4. Implement Backtester ---
def backtest_strategy(df_with_indicators, signals_df, initial_capital, position_size_usd):
    print("\n--- Running Backtest ---")
    if df_with_indicators.empty or signals_df.empty:
        print("Cannot backtest with empty data or signals.")
        return pd.DataFrame(), {}
    
    capital = initial_capital
    positions_asset_units = 0.0
    portfolio_value_history = []
    trades_log = [] 
    in_position = False

    for i in range(len(df_with_indicators)):
        current_date = df_with_indicators.index[i]
        current_price = df_with_indicators['close'].iloc[i]
        signal_action = signals_df['signal'].iloc[i]
        sell_reason_for_log = signals_df['sell_reason'].iloc[i] if 'sell_reason' in signals_df.columns else ''

        if signal_action == 1 and not in_position and capital >= position_size_usd:
            units_to_buy = position_size_usd / current_price
            positions_asset_units += units_to_buy
            capital -= position_size_usd
            in_position = True
            trades_log.append({
                'date': current_date, 'type': 'BUY', 'price': current_price,
                'units': units_to_buy, 'cost': position_size_usd, 'capital_after': capital,
                'proceeds': pd.NA, 'reason': '' })
        elif signal_action == -1 and in_position:
            proceeds_from_sale = positions_asset_units * current_price
            capital += proceeds_from_sale
            trades_log.append({
                'date': current_date, 'type': 'SELL', 'price': current_price,
                'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital,
                'proceeds': proceeds_from_sale, 'reason': sell_reason_for_log })
            positions_asset_units = 0.0
            in_position = False
        
        current_portfolio_value = capital + (positions_asset_units * current_price)
        portfolio_value_history.append(current_portfolio_value)

    portfolio_df = pd.DataFrame({'value': portfolio_value_history}, index=df_with_indicators.index)
    final_portfolio_value = portfolio_df['value'].iloc[-1] if not portfolio_df.empty else initial_capital
    total_return_abs = final_portfolio_value - initial_capital
    total_return_pct = (total_return_abs / initial_capital) * 100 if initial_capital > 0 else 0
    num_trades_executed = len([t for t in trades_log if t['type'] == 'BUY'])

    results = {
        'initial_capital': initial_capital, 'final_portfolio_value': final_portfolio_value,
        'total_return_usd': total_return_abs, 'total_return_pct': total_return_pct,
        'num_trades': num_trades_executed,
        'trades_log_df': pd.DataFrame(trades_log, columns=['date', 'type', 'price', 'units', 'cost', 'capital_after', 'proceeds', 'reason'])
    }
    return portfolio_df, results

# --- 5. Run and Analyze ---
if __name__ == "__main__":
    print(f"Starting trading strategy backtest for {SYMBOL}")
    print(f"Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"Parameters: BB({BB_LENGTH},{BB_STD_DEV}), RSI({RSI_LENGTH}, OB:{RSI_OVERBOUGHT}, OS:{RSI_OVERSOLD}), VWAP_Stop_Buffer: {VWAP_STOP_PERCENTAGE_BUFFER*100}%")
    print("---" * 10)

    data_df_raw = get_historical_data(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR)

    if not data_df_raw.empty:
        print("\n--- Raw Data Sample (head) ---")
        print(data_df_raw.head())
        
        data_with_indicators = add_indicators(data_df_raw.copy())
        
        if not data_with_indicators.empty:
            print("\n--- DETAILED DATA RANGE CHECK ---") # Added diagnostic
            print(f"data_df_raw (fetched) START: {data_df_raw.index.min()}, END: {data_df_raw.index.max()}, COUNT: {len(data_df_raw)}")
            print(f"data_with_indicators (after ind & dropna) START: {data_with_indicators.index.min()}, END: {data_with_indicators.index.max()}, COUNT: {len(data_with_indicators)}")
            
            print("\n--- Data with Indicators (tail sample) ---")
            print(data_with_indicators.tail())
            
            trading_signals_df = generate_signals(data_with_indicators)
            portfolio_history_df, backtest_summary = backtest_strategy(
                data_with_indicators, trading_signals_df, INITIAL_CAPITAL, POSITION_SIZE_USD
            )

            print("\n--- Backtest Results ---")
            print(f"Initial Capital: ${backtest_summary['initial_capital']:.2f}")
            print(f"Final Portfolio Value: ${backtest_summary['final_portfolio_value']:.2f}")
            print(f"Total Return: ${backtest_summary['total_return_usd']:.2f} ({backtest_summary['total_return_pct']:.2f}%)")
            print(f"Number of Trades Executed: {backtest_summary['num_trades']}")

            print("\n--- Trades Log ---")
            trades_df = backtest_summary['trades_log_df']
            if not trades_df.empty:
                trades_df_display = trades_df.copy()
                trades_df_display['date'] = pd.to_datetime(trades_df_display['date']).dt.strftime('%Y-%m-%d')
                print(trades_df_display.to_string())
            else: print("No trades were executed.")

            if backtest_summary['num_trades'] < 5 and (datetime.strptime(END_DATE_STR, '%Y-%m-%d') - datetime.strptime(START_DATE_STR, '%Y-%m-%d')).days > 180 :
                print("\nWARNING: Relatively few trades for the backtest period. Consider further parameter tuning or strategy adjustments.")

            if not portfolio_history_df.empty:
                try:
                    plt.figure(figsize=(14, 7))
                    plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label='Strategy Equity')
                    buy_dates = trades_df[trades_df['type'] == 'BUY']['date']
                    sell_dates = trades_df[trades_df['type'] == 'SELL']['date']
                    buy_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(buy_dates)]['value']
                    sell_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(sell_dates)]['value']
                    plt.plot(buy_values.index, buy_values, '^', markersize=10, color='g', lw=0, label='Buy Signal')
                    plt.plot(sell_values.index, sell_values, 'v', markersize=10, color='r', lw=0, label='Sell Signal')
                    plt.title(f'{SYMBOL} Strategy Equity Curve ({START_DATE_STR} to {END_DATE_STR})')
                    plt.ylabel('Portfolio Value (USD)'); plt.xlabel('Date')
                    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
                except Exception as e: print(f"\nError during plotting: {e}")
            else: print("\nPortfolio history is empty, cannot plot.")
        else: print("Could not generate indicators or data became empty. Aborting backtest.")
    else: print("No data fetched. Aborting backtest.")
    print("\n--- Script Finished ---")