import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta # For technical indicators
from datetime import datetime, timedelta
import os
import pytz # For timezone handling
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- Configuration ---
# Best practice: Use environment variables for API keys
# For testing, you can hardcode them but REMOVE before committing/sharing
# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets') # Use paper trading URL

SYMBOL = 'XRP/USD' # Alpaca uses '/' for crypto pairs
TIMEFRAME = tradeapi.TimeFrame.Day # Or TimeFrame.Hour, TimeFrame.Minute etc.
START_DATE_STR = '2024-01-01' # Backtesting start date
END_DATE_STR = '2024-12-31'   # Backtesting end date

# Strategy Parameters
BB_LENGTH = 20
BB_STD_DEV = 2.0
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
# VWAP is typically calculated daily or intraday. pandas_ta will handle it based on data.

INITIAL_CAPITAL = 10000.00
POSITION_SIZE_USD = 1000.00 # Invest $1000 per trade

# --- Alpaca API Setup ---
try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"Connected to Alpaca. Account Status: {account.status}")
except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    exit()

# --- 1. Fetch Historical Data ---
def get_historical_data(symbol, timeframe, start, end):
    """Fetches historical bar data from Alpaca."""
    print(f"Fetching data for {symbol} from {start} to {end} with timeframe {timeframe}")
    try:
        # Alpaca API expects start and end times in ISO format and UTC.
        # Ensure dates are timezone-aware (UTC)
        utc = pytz.UTC
        start_dt = utc.localize(datetime.strptime(start, '%Y-%m-%d'))
        end_dt = utc.localize(datetime.strptime(end, '%Y-%m-%d'))

        # Alpaca's get_crypto_bars returns data up to, but not including, the end date.
        # So, to include the end date, we might need to adjust or fetch one extra day.
        # For simplicity, we'll use what it returns.
        bars = api.get_crypto_bars(symbol, timeframe, start=start_dt.isoformat(), end=end_dt.isoformat()).df
        
        # Ensure data is localized to UTC and then convert to a common timezone if needed, e.g., 'America/New_York'
        # For crypto, data is often 24/7 and UTC is standard.
        # if not bars.index.tz:
        #     bars = bars.tz_localize('UTC')
        # else:
        #     bars = bars.tz_convert('UTC')

        # Select only OHLCV columns if others exist

        alpaca_client = CryptoHistoricalDataClient()

        request_params = CryptoBarsRequest(
            symbol_or_symbols=["XRP/USD"],
            timeframe=TimeFrame.Day,
            start=start_dt.isoformat(),
            end=end_dt.isoformat()
        )

        bars = alpaca_client.get_crypto_bars(request_params).df

        bars = bars[['open', 'high', 'low', 'close', 'volume', 'vwap']] # Alpaca crypto often provides vwap
        print(f"Fetched {len(bars)} bars.")
        return bars
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

# --- 2. Calculate Indicators ---
def add_indicators(df):
    """Adds Bollinger Bands, RSI, and VWAP to the DataFrame."""
    if df.empty:
        return df

    # Bollinger Bands
    df.ta.bbands(length=BB_LENGTH, std=BB_STD_DEV, append=True)
    # Column names will be BBL_20_2.0, BBM_20_2.0, BBU_20_2.0

    # RSI
    df.ta.rsi(length=RSI_LENGTH, append=True)
    # Column name will be RSI_14

    # VWAP - Alpaca crypto bars often provide 'vwap'. If not, pandas_ta can calculate it.
    # Let's use the one from Alpaca if available, or calculate if not.
    if 'vwap' not in df.columns:
        print("VWAP not in Alpaca data, calculating using pandas_ta...")
        df.ta.vwap(append=True) # Needs high, low, close, volume. Column: VWAP_D (if daily)
    else:
        print("Using VWAP provided by Alpaca.")
        # Rename Alpaca's vwap to match pandas_ta standard if needed, or just use 'vwap'
        # For this strategy, we'll just ensure a column named 'VWAP' exists.
        # If pandas_ta calculated it, it might be VWAP_D or similar. Let's standardize.
        if f'VWAP_D' in df.columns and 'vwap' not in df.columns: # Example for daily VWAP from pandas_ta
             df.rename(columns={f'VWAP_D': 'VWAP_calculated'}, inplace=True) # Keep Alpaca's 'vwap'
        elif 'vwap' in df.columns:
             df.rename(columns={'vwap': 'VWAP'}, inplace=True) # Ensure it's 'VWAP'

    df.dropna(inplace=True) # Remove rows with NaN values (due to indicator lookback periods)
    return df

# --- 3. Define Strategy & Generate Signals ---
def generate_signals(df):
    """Generates buy (1), sell (-1), or hold (0) signals."""
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0

    # Column names from pandas_ta (adjust if your version/settings differ)
    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    upper_band_col = f'BBU_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}' # Middle Bollinger Band
    rsi_col = f'RSI_{RSI_LENGTH}'
    vwap_col = 'VWAP' # Standardized VWAP column name

    # Check if all required columns exist
    required_cols = [lower_band_col, upper_band_col, middle_band_col, rsi_col, vwap_col, 'close', 'open']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns for signal generation: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return signals # Return empty signals

    # Buy Conditions
    buy_condition = (df['close'] < df[lower_band_col]) & \
                    (df[rsi_col] < RSI_OVERSOLD) & \
                    (df['close'] > df[vwap_col])
    
    signals.loc[buy_condition, 'signal'] = 1

    # Sell Conditions (to exit a long position)
    # Option 1: Exit on overbought RSI or hitting upper BB
    sell_condition_profit_take = (df['close'] > df[upper_band_col]) | \
                                 (df[rsi_col] > RSI_OVERBOUGHT)
    
    # Option 2: Exit if price drops below VWAP (protective stop)
    sell_condition_stop = (df['close'] < df[vwap_col])
    
    # Combine sell conditions:
    # We only generate a sell signal if we are potentially in a position.
    # The backtester will handle the actual "in position" logic.
    # Here, we mark potential exit points.
    signals.loc[sell_condition_profit_take, 'signal'] = -1
    signals.loc[sell_condition_stop, 'signal'] = -1 # This might overwrite profit take, order matters or combine with OR

    # More refined sell:
    # If we have a buy signal, we don't want an immediate sell signal on the same bar.
    # Prioritize buy signals if conditions for both met (unlikely with this logic but good practice)
    signals.loc[buy_condition, 'signal'] = 1 

    return signals


# --- 4. Implement Backtester ---
def backtest_strategy(df, signals, initial_capital, position_size_usd):
    """Performs a simple backtest of the trading strategy."""
    if df.empty or signals.empty:
        print("Cannot backtest with empty data or signals.")
        return pd.DataFrame(), {}

    capital = initial_capital
    positions = 0.0  # Number of units of the asset held
    portfolio_value = []
    trades_log = [] # To log individual trades

    # Align signals with data - crucial step!
    # Assume signal at close of bar 'i' is actionable at open of bar 'i+1'
    # For simplicity here, we'll assume action at close of bar 'i' where signal occurs.
    # If you want to trade on next bar's open, shift signals: `signals['signal'] = signals['signal'].shift(1)`
    # and then merge/join with df. For this example, direct iteration.

    in_position = False

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['close'].iloc[i] # Assume trade at close
        # current_price = df['open'].iloc[i] # If trading at next bar's open (after shifting signals)
        
        signal = signals['signal'].iloc[i]

        # --- Apply Trading Logic ---
        if signal == 1 and not in_position and capital >= position_size_usd: # Buy signal
            units_to_buy = position_size_usd / current_price
            positions += units_to_buy
            capital -= units_to_buy * current_price # More precise: capital -= position_size_usd
            in_position = True
            trades_log.append({
                'date': current_date, 'type': 'BUY', 'price': current_price,
                'units': units_to_buy, 'cost': position_size_usd, 'capital_after': capital
            })
            # print(f"{current_date}: BUY {units_to_buy:.4f} {SYMBOL.split('/')[0]} at ${current_price:.4f}")

        elif signal == -1 and in_position: # Sell signal
            proceeds = positions * current_price
            capital += proceeds
            # print(f"{current_date}: SELL {positions:.4f} {SYMBOL.split('/')[0]} at ${current_price:.4f}, Proceeds: ${proceeds:.2f}")
            trades_log.append({
                'date': current_date, 'type': 'SELL', 'price': current_price,
                'units': positions, 'proceeds': proceeds, 'capital_after': capital
            })
            positions = 0.0
            in_position = False
        
        # --- Update portfolio value ---
        current_portfolio_value = capital + (positions * current_price)
        portfolio_value.append(current_portfolio_value)

    # --- Prepare Results ---
    portfolio_df = pd.DataFrame({'value': portfolio_value}, index=df.index)
    
    total_return_abs = portfolio_df['value'].iloc[-1] - initial_capital
    total_return_pct = (total_return_abs / initial_capital) * 100
    
    num_trades = len([t for t in trades_log if t['type'] == 'BUY']) # Count buy entries

    results = {
        'initial_capital': initial_capital,
        'final_portfolio_value': portfolio_df['value'].iloc[-1],
        'total_return_usd': total_return_abs,
        'total_return_pct': total_return_pct,
        'num_trades': num_trades,
        'trades_log': pd.DataFrame(trades_log)
    }
    return portfolio_df, results

# --- 5. Run and Analyze ---
if __name__ == "__main__":
    print(f"Starting trading strategy backtest for {SYMBOL}")
    print("---" * 10)

    # Get Data
    data_df = get_historical_data(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR)

    if not data_df.empty:
        # Add Indicators
        data_with_indicators = add_indicators(data_df.copy()) # Use .copy() to avoid SettingWithCopyWarning
        
        if not data_with_indicators.empty:
            print("\n--- Data with Indicators (tail) ---")
            print(data_with_indicators.tail())

            # Generate Signals
            trading_signals = generate_signals(data_with_indicators)
            print("\n--- Trading Signals (where signal is not 0) ---")
            print(trading_signals[trading_signals['signal'] != 0].head(20))

            # Ensure signals DataFrame index matches data_with_indicators for backtesting
            # This is important if generate_signals modified the index or length
            # For this implementation, they should align if no errors.
            
            # Backtest
            print("\n--- Running Backtest ---")
            portfolio_df, backtest_results = backtest_strategy(
                data_with_indicators, 
                trading_signals, 
                INITIAL_CAPITAL,
                POSITION_SIZE_USD
            )

            print("\n--- Backtest Results ---")
            print(f"Initial Capital: ${backtest_results['initial_capital']:.2f}")
            print(f"Final Portfolio Value: ${backtest_results['final_portfolio_value']:.2f}")
            print(f"Total Return: ${backtest_results['total_return_usd']:.2f} ({backtest_results['total_return_pct']:.2f}%)")
            print(f"Number of Trades: {backtest_results['num_trades']}")

            print("\n--- Trades Log (first 10 trades) ---")
            if not backtest_results['trades_log'].empty:
                print(backtest_results['trades_log'].head(10))
            else:
                print("No trades were executed.")

            # Optional: Plotting results
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                portfolio_df['value'].plot(title=f'{SYMBOL} Strategy Equity Curve')
                plt.ylabel('Portfolio Value (USD)')
                plt.xlabel('Date')
                plt.grid(True)
                plt.show()
            except ImportError:
                print("\nMatplotlib not installed. Skipping plot. Install with: pip install matplotlib")
        else:
            print("Could not generate indicators. Aborting backtest.")
    else:
        print("No data fetched. Aborting backtest.")

    print("\n--- Script Finished ---")