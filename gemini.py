import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, date
import os
import pytz
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

START_DATE_STR = '2024-01-01'
YESTERDAY = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
END_DATE_STR = YESTERDAY

# Strategy Parameters
BB_LENGTH = 20
BB_STD_DEV = 2.0
RSI_LENGTH = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
# VWAP_STOP_PERCENTAGE_BUFFER = 0.02 # <<< REMOVED, will use ATR stop

ATR_PERIOD = 14 # <<< NEW: ATR Period
ATR_STOP_MULTIPLIER = 2.25 # <<< NEW: ATR Stop Multiplier

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
    # (Same as your last version)
    print(f"Fetching data for {symbol} from {start_str} to {end_str} with timeframe {timeframe.value}")
    try:
        utc = pytz.UTC; start_dt = utc.localize(datetime.strptime(start_str, '%Y-%m-%d'))
        end_dt_inclusive = utc.localize(datetime.strptime(end_str, '%Y-%m-%d'))
        end_dt_exclusive = end_dt_inclusive + timedelta(days=1)
        bars = api.get_crypto_bars(symbol, timeframe, start=start_dt.isoformat(), end=end_dt_exclusive.isoformat()).df
        if bars.empty: print("No data fetched."); return pd.DataFrame()
        if not bars.index.tz: bars = bars.tz_localize('UTC')
        else: bars = bars.tz_convert('UTC')
        bars = bars[bars.index <= end_dt_inclusive.replace(hour=23, minute=59, second=59)]
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if 'vwap' in bars.columns: required_ohlcv.append('vwap') # Keep VWAP for potential future use or reference
        bars = bars[required_ohlcv]
        print(f"Fetched {len(bars)} bars. Actual date range: {bars.index.min()} to {bars.index.max()}")
        return bars
    except Exception as e: print(f"Error fetching data: {e}"); return pd.DataFrame()

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
    if 'vwap' in df.columns: # Keep VWAP if provided, rename
        print("Using VWAP provided by Alpaca.")
        df.rename(columns={'vwap': 'VWAP'}, inplace=True)
    else: # Calculate if not provided (though Alpaca crypto usually has it)
        print("VWAP not in Alpaca data, calculating...")
        df.ta.vwap(append=True)
        if 'VWAP_D' in df.columns: df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True)


    # --- NEW: Add ATR ---
    df.ta.atr(length=ATR_PERIOD, append=True)
    atr_col = f'ATRr_{ATR_PERIOD}' # pandas-ta default name for ATR
    print(f"Added {atr_col}")

    print(f"Columns after adding all indicators: {df.columns.tolist()}")
    # VWAP is no longer strictly required for the ATR stop strategy, but good to have
    required_cols_check = ['close', lower_band_col, middle_band_col, upper_band_col, rsi_col, atr_col]
    if 'VWAP' in df.columns: required_cols_check.append('VWAP') # Add if exists for completeness

    missing_cols = [col for col in required_cols_check if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing critical indicator columns BEFORE dropna: {missing_cols}")

    df.dropna(inplace=True)
    print(f"Shape of data after indicators & dropna: {df.shape}")
    if df.empty: print("DataFrame empty after indicators & dropna.")
    return df

# --- 3. Define Strategy & Generate Signals (Profit Takes Only) ---
def generate_signals(df):
    print("\n--- Generating Signals (Profit Takes & Buys) ---")
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['sell_reason'] = '' # For profit-take reasons

    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    upper_band_col = f'BBU_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'
    rsi_col = f'RSI_{RSI_LENGTH}'
    # vwap_col = 'VWAP' # No longer used for stop in this function

    required_cols = [lower_band_col, upper_band_col, middle_band_col, rsi_col, 'close']
    missing_cols_check = [col for col in required_cols if col not in df.columns]
    if missing_cols_check:
        print(f"Error: Missing columns for signal gen: {missing_cols_check}\nAvailable: {df.columns.tolist()}")
        return signals

    # Buy Conditions (same as before)
    cond1_buy_lbb = (df['close'] < df[lower_band_col])
    cond2_buy_rsi = (df[rsi_col] < RSI_OVERSOLD)
    buy_condition = cond1_buy_lbb & cond2_buy_rsi
    
    # Sell Conditions (PROFIT TAKES ONLY)
    cond_sell_profit_upperbb = (df['close'] > df[upper_band_col])
    cond_sell_profit_rsi = (df[rsi_col] > RSI_OVERBOUGHT)
    cond_sell_profit_middlebb = (df['close'] > df[middle_band_col])
    # cond_sell_stop_vwap = ... # <<< REMOVED VWAP STOP FROM HERE

    # Apply profit-take sell signals (priority: UpperBB > RSI_OB > MiddleBB_PT)
    signals.loc[cond_sell_profit_upperbb, 'sell_reason'] = 'UpperBB'
    signals.loc[cond_sell_profit_rsi & (signals['sell_reason'] == ''), 'sell_reason'] = 'RSI_OB'
    signals.loc[cond_sell_profit_middlebb & (signals['sell_reason'] == ''), 'sell_reason'] = 'MiddleBB_PT'
    # The ATR stop will be handled in the backtester
    
    # Set signal to -1 if any profit-take reason was populated
    signals.loc[signals['sell_reason'] != '', 'signal'] = -1 # Mark as a potential sell for backtester
    
    # Apply buy signals
    signals.loc[buy_condition, 'signal'] = 1
    signals.loc[buy_condition, 'sell_reason'] = '' # Clear sell reason for buy signals

    # --- DEBUGGING PRINTS ---
    print(f"RSI Oversold: {RSI_OVERSOLD}, RSI Overbought: {RSI_OVERBOUGHT}")
    print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER} (Stop logic in backtester)")
    print(f"Number of potential buy signal points (raw conditions met): {buy_condition.sum()}")
    print(f"  Breakdown: LBB: {cond1_buy_lbb.sum()}, RSI: {cond2_buy_rsi.sum()}")
    
    print(f"Number of potential PROFIT TAKE triggers (UpperBB): {cond_sell_profit_upperbb.sum()}")
    print(f"Number of potential PROFIT TAKE triggers (RSI_OB {RSI_OVERBOUGHT}): {cond_sell_profit_rsi.sum()}")
    print(f"Number of potential PROFIT TAKE triggers (MiddleBB): {cond_sell_profit_middlebb.sum()}")
    
    actual_buy_signals = signals[signals['signal'] == 1]
    # Sell signals here are only profit-take signals from this function's perspective
    potential_profit_take_signals = signals[signals['sell_reason'] != '']
    print(f"Total BUY signals generated: {len(actual_buy_signals)}")
    print(f"Total PROFIT TAKE signals generated: {len(potential_profit_take_signals)}")

    if not actual_buy_signals.empty:
        print("Dates of first 5 BUY signals:")
        print(actual_buy_signals.head().index.strftime('%Y-%m-%d').tolist())
    if not potential_profit_take_signals.empty:
        print("Dates of first 5 potential PROFIT TAKE signals (and reasons):")
        print(potential_profit_take_signals.head()[['sell_reason']])
    
    return signals

# --- 4. Implement Backtester with ATR Stop ---
def backtest_strategy(df_with_indicators, signals_df, initial_capital, position_size_usd):
    print("\n--- Running Backtest with ATR Stop ---")
    if df_with_indicators.empty or signals_df.empty: print("Cannot backtest: empty data/signals."); return pd.DataFrame(), {}
    
    capital = initial_capital
    positions_asset_units = 0.0
    portfolio_value_history = []
    trades_log = [] 
    in_position = False
    
    current_entry_price = 0.0
    current_atr_stop_level = 0.0
    atr_col = f'ATRr_{ATR_PERIOD}' # Ensure this matches the column name from add_indicators

    for i in range(len(df_with_indicators)):
        current_date = df_with_indicators.index[i]
        current_price = df_with_indicators['close'].iloc[i]
        current_low_price = df_with_indicators['low'].iloc[i] # Use low for stop check for intra-bar trigger
        
        # Get profit-take signal from signals_df (if any)
        profit_take_signal_action = signals_df['signal'].iloc[i]
        profit_take_reason = signals_df['sell_reason'].iloc[i]

        # --- Exit Logic: Check ATR Stop first, then Profit Take Signals ---
        if in_position:
            # 1. Check ATR Stop-Loss
            if current_low_price <= current_atr_stop_level: # Stop triggered if low hits/crosses stop level
                proceeds_from_sale = positions_asset_units * current_atr_stop_level # Assume exit at stop level
                capital += proceeds_from_sale
                trades_log.append({'date': current_date, 'type': 'SELL', 'price': current_atr_stop_level, 
                                   'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital, 
                                   'proceeds': proceeds_from_sale, 'reason': 'ATR_Stop' })
                positions_asset_units = 0.0; in_position = False; current_entry_price = 0.0; current_atr_stop_level = 0.0
                # print(f"ATR Stop at {current_date} for price {current_atr_stop_level}")

            # 2. Check Profit-Take Signals (only if not already stopped by ATR)
            elif profit_take_signal_action == -1 and profit_take_reason != '':
                proceeds_from_sale = positions_asset_units * current_price # Exit at close for PT
                capital += proceeds_from_sale
                trades_log.append({'date': current_date, 'type': 'SELL', 'price': current_price, 
                                   'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital, 
                                   'proceeds': proceeds_from_sale, 'reason': profit_take_reason })
                positions_asset_units = 0.0; in_position = False; current_entry_price = 0.0; current_atr_stop_level = 0.0
                # print(f"Profit Take {profit_take_reason} at {current_date} for price {current_price}")

        # --- Entry Logic ---
        # Check for buy signal (signal_action == 1) only if not currently in a position
        # and not just exited on the same bar (important if profit_take_signal_action was processed above)
        if signals_df['signal'].iloc[i] == 1 and not in_position and capital >= position_size_usd:
            current_entry_price = current_price # Entry at close
            units_to_buy = position_size_usd / current_entry_price
            positions_asset_units += units_to_buy
            capital -= position_size_usd
            in_position = True
            
            # Set initial ATR stop for this new trade
            atr_at_entry = df_with_indicators[atr_col].iloc[i]
            current_atr_stop_level = current_entry_price - (ATR_STOP_MULTIPLIER * atr_at_entry)
            # print(f"BUY at {current_date}, Entry: {current_entry_price:.4f}, ATR: {atr_at_entry:.4f}, Initial ATR Stop: {current_atr_stop_level:.4f}")

            trades_log.append({'date': current_date, 'type': 'BUY', 'price': current_entry_price, 
                               'units': units_to_buy, 'cost': position_size_usd, 'capital_after': capital, 
                               'proceeds': pd.NA, 'reason': '' })
        
        current_portfolio_value = capital + (positions_asset_units * current_price)
        portfolio_value_history.append(current_portfolio_value)

    # --- Prepare Results ---
    # (Same as before)
    portfolio_df = pd.DataFrame({'value': portfolio_value_history}, index=df_with_indicators.index)
    final_val = portfolio_df['value'].iloc[-1] if not portfolio_df.empty else initial_capital; ret_abs = final_val - initial_capital
    ret_pct = (ret_abs / initial_capital) * 100 if initial_capital > 0 else 0; num_trades = len([t for t in trades_log if t['type'] == 'BUY'])
    results = {'initial_capital': initial_capital, 'final_portfolio_value': final_val, 'total_return_usd': ret_abs, 'total_return_pct': ret_pct, 'num_trades': num_trades, 'trades_log_df': pd.DataFrame(trades_log, columns=['date', 'type', 'price', 'units', 'cost', 'capital_after', 'proceeds', 'reason'])}
    return portfolio_df, results

# --- 5. Run and Analyze ---
if __name__ == "__main__":
    print(f"Starting trading strategy backtest for {SYMBOL}")
    print(f"Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"Parameters: BB({BB_LENGTH},{BB_STD_DEV}), RSI({RSI_LENGTH}, OB:{RSI_OVERBOUGHT}, OS:{RSI_OVERSOLD}), ATR_Stop({ATR_PERIOD}, Multiplier:{ATR_STOP_MULTIPLIER})")
    print("---" * 10)

    data_df_raw = get_historical_data(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR)

    if not data_df_raw.empty:
        data_with_indicators = add_indicators(data_df_raw.copy())
        
        if not data_with_indicators.empty:
            print("\n--- Data with Indicators (ATR tail sample) ---")
            atr_col_name_for_print = f'ATRr_{ATR_PERIOD}'
            if atr_col_name_for_print in data_with_indicators.columns:
                 print(data_with_indicators[['close', atr_col_name_for_print]].tail())
            else:
                print(f"'{atr_col_name_for_print}' not found in columns for tail print.")
            
            trading_signals_df = generate_signals(data_with_indicators) # Generates Buys & Profit-Take Sells
            portfolio_history_df, backtest_summary = backtest_strategy(
                data_with_indicators, trading_signals_df, INITIAL_CAPITAL, POSITION_SIZE_USD
            )

            print("\n--- Backtest Results ---")
            # (Print statements for results and trades log are same as before)
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
                print("\nWARNING: Relatively few trades. Consider parameter tuning.")

            # (Plotting code is same as before)
            if not portfolio_history_df.empty:
                try:
                    plt.figure(figsize=(14, 7)); plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label='Strategy Equity', alpha=0.7)
                    buy_dates = trades_df[trades_df['type'] == 'BUY']['date']; sell_dates = trades_df[trades_df['type'] == 'SELL']['date']
                    if not portfolio_history_df.index.is_monotonic_increasing: portfolio_history_df = portfolio_history_df.sort_index()
                    buy_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(buy_dates)]['value']
                    sell_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(sell_dates)]['value']
                    plt.plot(buy_values.index, buy_values, '^', markersize=10, color='g', lw=0, label='Buy Signal')
                    plt.plot(sell_values.index, sell_values, 'v', markersize=10, color='r', lw=0, label='Sell Signal')
                    plt.title(f'{SYMBOL} Strat Equity ({START_DATE_STR} to {END_DATE_STR}) - ATR Stop ({ATR_STOP_MULTIPLIER}x)'); plt.ylabel('Portfolio Value (USD)'); plt.xlabel('Date'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
                except Exception as e: print(f"\nError during plotting: {e}")
            else: print("\nPortfolio history empty, cannot plot.")
        else: print("Could not generate indicators. Aborting.")
    else: print("No data fetched. Aborting.")
    print("\n--- Script Finished ---")