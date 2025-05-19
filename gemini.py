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

# Use a specific period for consistent testing, e.g., your out-of-sample period or best full period
# For this example, let's use the Jan 2025 - May 2025 out-of-sample period you showed good results on
# Or adjust to your longer preferred testing period
START_DATE_STR = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d') # Example: Out-of-sample
END_DATE_STR = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d') # Use yesterday as end for fresh runs
# Or, for reproducing exact results from a specific test:
# END_DATE_STR = '2025-05-18' # If that was your last successful test end date

# Strategy Parameters
BB_LENGTH = 20
BB_STD_DEV = 2.0
RSI_LENGTH = 14
# RSI_OVERBOUGHT = 65 # Simplified: No longer used in generate_signals if MiddleBB is primary PT
RSI_OVERSOLD = 35
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.25 # Your best performing multiplier

TAKER_FEE_RATE = 0.0025 # 0.25% Taker Fee <<< NEW

INITIAL_CAPITAL = 1000.00
POSITION_SIZE_USD_TARGET = 100.00 # Target exposure before fees

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
        utc = pytz.UTC; start_dt = utc.localize(datetime.strptime(start_str, '%Y-%m-%d'))
        end_dt_inclusive = utc.localize(datetime.strptime(end_str, '%Y-%m-%d'))
        end_dt_exclusive = end_dt_inclusive + timedelta(days=1)
        bars = api.get_crypto_bars(symbol, timeframe, start=start_dt.isoformat(), end=end_dt_exclusive.isoformat()).df
        if bars.empty: print("No data fetched."); return pd.DataFrame()
        if not bars.index.tz: bars = bars.tz_localize('UTC')
        else: bars = bars.tz_convert('UTC')
        bars = bars[bars.index <= end_dt_inclusive.replace(hour=23, minute=59, second=59)]
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if 'vwap' in bars.columns: required_ohlcv.append('vwap')
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
    # upper_band_col = f'BBU_{BB_LENGTH}_{BB_STD_DEV}' # Not used if simplified
    df.ta.rsi(length=RSI_LENGTH, append=True)
    # rsi_col = f'RSI_{RSI_LENGTH}' # Not used if simplified
    if 'vwap' in df.columns: df.rename(columns={'vwap': 'VWAP'}, inplace=True)
    else: df.ta.vwap(append=True); df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True, errors='ignore')
    df.ta.atr(length=ATR_PERIOD, append=True)
    atr_col = f'ATRr_{ATR_PERIOD}'
    print(f"Added ATR: {atr_col}")
    print(f"Columns after adding all indicators: {df.columns.tolist()}")
    required_cols_check = ['close', 'low', lower_band_col, middle_band_col, f'RSI_{RSI_LENGTH}', atr_col] # low is needed for ATR stop
    missing_cols = [col for col in required_cols_check if col not in df.columns]
    if missing_cols: print(f"ERROR: Missing critical indicator columns BEFORE dropna: {missing_cols}")
    df.dropna(inplace=True); print(f"Shape of data after indicators & dropna: {df.shape}")
    if df.empty: print("DataFrame empty after indicators & dropna.")
    return df

# --- 3. Define Strategy & Generate Signals (Simplified Profit Takes & Buys) ---
def generate_signals(df):
    print("\n--- Generating Signals (Simplified Profit Takes & Buys) ---")
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['sell_reason'] = ''

    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'
    rsi_col = f'RSI_{RSI_LENGTH}'

    required_cols = [lower_band_col, middle_band_col, rsi_col, 'close']
    if any(col not in df.columns for col in required_cols):
        print(f"Error: Missing columns for signal gen. Req: {required_cols}. Avail: {df.columns.tolist()}"); return signals

    cond1_buy_lbb = (df['close'] < df[lower_band_col])
    cond2_buy_rsi = (df[rsi_col] < RSI_OVERSOLD)
    buy_condition = cond1_buy_lbb & cond2_buy_rsi
    
    cond_sell_profit_middlebb = (df['close'] > df[middle_band_col])
    # Removed UpperBB and RSI_OB profit take conditions for simplification
    
    signals.loc[cond_sell_profit_middlebb, 'sell_reason'] = 'MiddleBB_PT'
    signals.loc[signals['sell_reason'] != '', 'signal'] = -1
    
    signals.loc[buy_condition, 'signal'] = 1
    signals.loc[buy_condition, 'sell_reason'] = ''

    print(f"RSI Oversold: {RSI_OVERSOLD}") # RSI_Overbought no longer printed if not used
    print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER} (Stop logic in backtester)")
    print(f"Number of potential buy signal points (raw): {buy_condition.sum()}")
    print(f"Number of potential PROFIT TAKE triggers (MiddleBB): {cond_sell_profit_middlebb.sum()}")
    print(f"Total BUY signals generated: {len(signals[signals['signal'] == 1])}")
    print(f"Total PROFIT TAKE signals generated: {len(signals[signals['sell_reason'] != ''])}")
    return signals

# --- 4. Implement Backtester with ATR Stop and Transaction Costs ---
# (Use the modified function from the snippet above)
def backtest_strategy(df_with_indicators, signals_df, initial_capital, position_size_usd_target):
    print("\n--- Running Backtest with ATR Stop & Transaction Costs ---")
    if df_with_indicators.empty or signals_df.empty: print("Cannot backtest: empty data/signals."); return pd.DataFrame(), {}
    
    TAKER_FEE_RATE = 0.0025 # 0.25% Taker Fee

    capital = initial_capital; positions_asset_units = 0.0; portfolio_value_history = []; trades_log = [] 
    in_position = False; current_entry_price_actual = 0.0; current_atr_stop_level = 0.0
    atr_col = f'ATRr_{ATR_PERIOD}'

    for i in range(len(df_with_indicators)):
        current_date = df_with_indicators.index[i]; current_price_market = df_with_indicators['close'].iloc[i]
        current_low_price_market = df_with_indicators['low'].iloc[i]
        profit_take_signal_action = signals_df['signal'].iloc[i]; profit_take_reason = signals_df['sell_reason'].iloc[i]

        if in_position:
            if current_low_price_market <= current_atr_stop_level: # ATR Stop
                sell_price_execution = current_atr_stop_level 
                proceeds_before_fees = positions_asset_units * sell_price_execution; fees_on_sell = proceeds_before_fees * TAKER_FEE_RATE
                proceeds_after_fees = proceeds_before_fees - fees_on_sell; capital += proceeds_after_fees
                trades_log.append({'date': current_date, 'type': 'SELL', 'price': sell_price_execution, 'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital, 'proceeds': proceeds_after_fees, 'fees': fees_on_sell, 'reason': 'ATR_Stop' })
                positions_asset_units = 0.0; in_position = False; current_entry_price_actual = 0.0; current_atr_stop_level = 0.0
            elif profit_take_signal_action == -1 and profit_take_reason != '': # Profit Take
                sell_price_execution = current_price_market 
                proceeds_before_fees = positions_asset_units * sell_price_execution; fees_on_sell = proceeds_before_fees * TAKER_FEE_RATE
                proceeds_after_fees = proceeds_before_fees - fees_on_sell; capital += proceeds_after_fees
                trades_log.append({'date': current_date, 'type': 'SELL', 'price': sell_price_execution, 'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital, 'proceeds': proceeds_after_fees, 'fees': fees_on_sell, 'reason': profit_take_reason })
                positions_asset_units = 0.0; in_position = False; current_entry_price_actual = 0.0; current_atr_stop_level = 0.0

        if signals_df['signal'].iloc[i] == 1 and not in_position and capital > 0: # Entry
            current_entry_price_actual = current_price_market
            units_to_buy = position_size_usd_target / current_entry_price_actual
            cost_before_fees = units_to_buy * current_entry_price_actual; fees_on_buy = cost_before_fees * TAKER_FEE_RATE
            total_cost_with_fees = cost_before_fees + fees_on_buy
            if capital >= total_cost_with_fees:
                positions_asset_units += units_to_buy; capital -= total_cost_with_fees; in_position = True
                atr_at_entry = df_with_indicators[atr_col].iloc[i]
                current_atr_stop_level = current_entry_price_actual - (ATR_STOP_MULTIPLIER * atr_at_entry)
                trades_log.append({'date': current_date, 'type': 'BUY', 'price': current_entry_price_actual, 'units': units_to_buy, 'cost': total_cost_with_fees, 'capital_after': capital, 'proceeds': pd.NA, 'fees': fees_on_buy, 'reason': '' })
        
        current_portfolio_value = capital + (positions_asset_units * current_price_market)
        portfolio_value_history.append(current_portfolio_value)

    portfolio_df = pd.DataFrame({'value': portfolio_value_history}, index=df_with_indicators.index)
    final_val = portfolio_df['value'].iloc[-1] if not portfolio_df.empty else initial_capital; ret_abs = final_val - initial_capital
    ret_pct = (ret_abs / initial_capital) * 100 if initial_capital > 0 else 0; num_trades = len([t for t in trades_log if t['type'] == 'BUY'])
    total_fees_paid = sum(t['fees'] for t in trades_log if pd.notna(t['fees']))
    results = {'initial_capital': initial_capital, 'final_portfolio_value': final_val, 'total_return_usd': ret_abs, 'total_return_pct': ret_pct, 'num_trades': num_trades, 'total_fees_paid': total_fees_paid, 'trades_log_df': pd.DataFrame(trades_log, columns=['date', 'type', 'price', 'units', 'cost', 'capital_after', 'proceeds', 'fees', 'reason'])}
    return portfolio_df, results

# --- 5. Run and Analyze ---
if __name__ == "__main__":
    print(f"Starting trading strategy backtest for {SYMBOL}")
    print(f"Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"Parameters: BB({BB_LENGTH},{BB_STD_DEV}), RSI_OS:{RSI_OVERSOLD}, ATR_Stop({ATR_PERIOD}, Multiplier:{ATR_STOP_MULTIPLIER}), TakerFee:{TAKER_FEE_RATE*100}%") # Simplified PTs
    print("---" * 10)

    data_df_raw = get_historical_data(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR)

    if not data_df_raw.empty:
        data_with_indicators = add_indicators(data_df_raw.copy())
        if not data_with_indicators.empty:
            trading_signals_df = generate_signals(data_with_indicators)
            portfolio_history_df, backtest_summary = backtest_strategy(
                data_with_indicators, trading_signals_df, INITIAL_CAPITAL, POSITION_SIZE_USD_TARGET
            )

            print("\n--- Backtest Results ---")
            print(f"Initial Capital: ${backtest_summary['initial_capital']:.2f}")
            print(f"Final Portfolio Value: ${backtest_summary['final_portfolio_value']:.2f}")
            print(f"Total Return (Net): ${backtest_summary['total_return_usd']:.2f} ({backtest_summary['total_return_pct']:.2f}%)")
            print(f"Number of Trades Executed: {backtest_summary['num_trades']}")
            print(f"Total Fees Paid: ${backtest_summary['total_fees_paid']:.2f}") # <<< NEW

            print("\n--- Trades Log (with fees) ---")
            trades_df = backtest_summary['trades_log_df']
            if not trades_df.empty:
                trades_df_display = trades_df.copy()
                trades_df_display['date'] = pd.to_datetime(trades_df_display['date']).dt.strftime('%Y-%m-%d')
                # Optionally format currency columns
                for col in ['price', 'cost', 'capital_after', 'proceeds', 'fees']:
                    if col in trades_df_display.columns:
                         trades_df_display[col] = trades_df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

                print(trades_df_display.to_string())
            else: print("No trades were executed.")

            if not portfolio_history_df.empty:
                try:
                    plt.figure(figsize=(14, 7)); plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label='Strategy Equity (Net)', alpha=0.7)
                    # ... (rest of plotting code remains the same) ...
                    buy_dates = trades_df[trades_df['type'] == 'BUY']['date']; sell_dates = trades_df[trades_df['type'] == 'SELL']['date']
                    if not portfolio_history_df.index.is_monotonic_increasing: portfolio_history_df = portfolio_history_df.sort_index()
                    buy_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(buy_dates)]['value']
                    sell_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(sell_dates)]['value']
                    plt.plot(buy_values.index, buy_values, '^', markersize=10, color='g', lw=0, label='Buy Signal')
                    plt.plot(sell_values.index, sell_values, 'v', markersize=10, color='r', lw=0, label='Sell Signal')
                    plt.title(f'{SYMBOL} Strat Equity ({START_DATE_STR} to {END_DATE_STR}) - ATR Stop ({ATR_STOP_MULTIPLIER}x) - Net of Fees'); plt.ylabel('Portfolio Value (USD)'); plt.xlabel('Date'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

                except Exception as e: print(f"\nError during plotting: {e}")
        else: print("Could not generate indicators. Aborting.")
    else: print("No data fetched. Aborting.")
    print("\n--- Script Finished ---")