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
RSI_OVERSOLD = 35
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.25

TAKER_FEE_RATE = 0.0025 # 0.25% Taker Fee

INITIAL_CAPITAL = 1000.00 # Or 10000.00, consistent with your previous tests
# POSITION_SIZE_USD_TARGET = 100.00 # <<< REMOVED
PERCENT_CAPITAL_TO_RISK = 0.1 # <<< NEW: Risk 1% of current capital per trade

MIN_RISK_PER_UNIT_THRESHOLD = 0.0001 # Safety: minimum $ risk per unit to consider trade

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
    # (Same as before)
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
    # (Same as before, simplified for MiddleBB PT)
    if df.empty: return df
    print("\n--- Adding Indicators ---")
    df.ta.bbands(length=BB_LENGTH, std=BB_STD_DEV, append=True)
    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'
    df.ta.rsi(length=RSI_LENGTH, append=True)
    if 'vwap' in df.columns: df.rename(columns={'vwap': 'VWAP'}, inplace=True)
    else: df.ta.vwap(append=True); df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True, errors='ignore')
    df.ta.atr(length=ATR_PERIOD, append=True)
    atr_col = f'ATRr_{ATR_PERIOD}'
    print(f"Added ATR: {atr_col}")
    print(f"Columns after adding all indicators: {df.columns.tolist()}")
    required_cols_check = ['close', 'low', lower_band_col, middle_band_col, f'RSI_{RSI_LENGTH}', atr_col]
    missing_cols = [col for col in required_cols_check if col not in df.columns]
    if missing_cols: print(f"ERROR: Missing critical indicator columns BEFORE dropna: {missing_cols}")
    df.dropna(inplace=True); print(f"Shape of data after indicators & dropna: {df.shape}")
    if df.empty: print("DataFrame empty after indicators & dropna.")
    return df

# --- 3. Define Strategy & Generate Signals (Simplified Profit Takes & Buys) ---
def generate_signals(df):
    # (Same as before, simplified for MiddleBB PT)
    print("\n--- Generating Signals (Simplified Profit Takes & Buys) ---")
    signals = pd.DataFrame(index=df.index); signals['signal'] = 0; signals['sell_reason'] = ''
    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'; middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'; rsi_col = f'RSI_{RSI_LENGTH}'
    required_cols = [lower_band_col, middle_band_col, rsi_col, 'close']
    if any(col not in df.columns for col in required_cols):
        print(f"Error: Missing cols for signal gen. Req: {required_cols}. Avail: {df.columns.tolist()}"); return signals
    cond1_buy_lbb = (df['close'] < df[lower_band_col]); cond2_buy_rsi = (df[rsi_col] < RSI_OVERSOLD)
    buy_condition = cond1_buy_lbb & cond2_buy_rsi
    cond_sell_profit_middlebb = (df['close'] > df[middle_band_col])
    signals.loc[cond_sell_profit_middlebb, 'sell_reason'] = 'MiddleBB_PT'
    signals.loc[signals['sell_reason'] != '', 'signal'] = -1
    signals.loc[buy_condition, 'signal'] = 1; signals.loc[buy_condition, 'sell_reason'] = ''
    print(f"RSI Oversold: {RSI_OVERSOLD}"); print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER} (Stop logic in backtester)")
    print(f"Number of potential buy signal points (raw): {buy_condition.sum()}")
    print(f"Number of potential PROFIT TAKE triggers (MiddleBB): {cond_sell_profit_middlebb.sum()}")
    print(f"Total BUY signals generated: {len(signals[signals['signal'] == 1])}")
    print(f"Total PROFIT TAKE signals generated: {len(signals[signals['sell_reason'] != ''])}")
    return signals

# --- 4. Implement Backtester with ATR Stop, Transaction Costs, and Percent Risk Sizing ---
def backtest_strategy(df_with_indicators, signals_df, initial_capital, percent_capital_to_risk):
    print("\n--- Running Backtest with ATR Stop, Transaction Costs & Percent Risk Sizing ---")
    if df_with_indicators.empty or signals_df.empty: print("Cannot backtest: empty data/signals."); return pd.DataFrame(), {}
    
    capital = initial_capital; positions_asset_units = 0.0; portfolio_value_history = []; trades_log = [] 
    in_position = False; current_entry_price_actual = 0.0; current_atr_stop_level = 0.0
    atr_col = f'ATRr_{ATR_PERIOD}'

    for i in range(len(df_with_indicators)):
        current_date = df_with_indicators.index[i]; current_price_market = df_with_indicators['close'].iloc[i]
        current_low_price_market = df_with_indicators['low'].iloc[i]
        profit_take_signal_action = signals_df['signal'].iloc[i]; profit_take_reason = signals_df['sell_reason'].iloc[i]
        atr_on_bar = df_with_indicators[atr_col].iloc[i] # ATR value for the current bar

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

        # --- Entry Logic with Percent Risk Sizing ---
        if signals_df['signal'].iloc[i] == 1 and not in_position and capital > 0:
            current_entry_price_actual = current_price_market
            
            # Calculate ATR stop level based on THIS bar's ATR for position sizing decision
            # (even though the actual trade's stop will use entry bar's ATR if different,
            # for sizing, we use current info to decide if trade is viable)
            potential_atr_stop_level = current_entry_price_actual - (ATR_STOP_MULTIPLIER * atr_on_bar)
            
            risk_per_unit_asset = current_entry_price_actual - potential_atr_stop_level

            if risk_per_unit_asset > MIN_RISK_PER_UNIT_THRESHOLD: # Ensure risk is meaningful and positive
                dollar_amount_to_risk = capital * percent_capital_to_risk
                units_to_buy_ideal = dollar_amount_to_risk / risk_per_unit_asset
                
                cost_before_fees_ideal = units_to_buy_ideal * current_entry_price_actual
                fees_on_buy_ideal = cost_before_fees_ideal * TAKER_FEE_RATE
                total_cost_with_fees_ideal = cost_before_fees_ideal + fees_on_buy_ideal

                if total_cost_with_fees_ideal > 0 and capital >= total_cost_with_fees_ideal: # Can afford the trade
                    # Actual units and cost might be slightly adjusted if position_size_usd_target was used,
                    # but with percent risk, units_to_buy_ideal is our target
                    positions_asset_units = units_to_buy_ideal # Use the calculated ideal units
                    capital -= total_cost_with_fees_ideal 
                    in_position = True
                    
                    # The actual ATR stop for the trade is based on the ATR of THIS entry bar
                    current_atr_stop_level = potential_atr_stop_level # Already calculated

                    trades_log.append({'date': current_date, 'type': 'BUY', 'price': current_entry_price_actual, 
                                       'units': positions_asset_units, 'cost': total_cost_with_fees_ideal, 
                                       'capital_after': capital, 'proceeds': pd.NA, 'fees': fees_on_buy_ideal, 'reason': '' })
                # else:
                    # print(f"Skipping BUY on {current_date}: Insufficient capital or position size too small/invalid.")
            # else:
                # print(f"Skipping BUY on {current_date}: Risk per unit too small or negative ({risk_per_unit_asset:.4f}). ATR: {atr_on_bar:.4f}")
        
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
    print(f"Parameters: BB({BB_LENGTH},{BB_STD_DEV}), RSI_OS:{RSI_OVERSOLD}, ATR_Stop({ATR_PERIOD}, Multiplier:{ATR_STOP_MULTIPLIER}), TakerFee:{TAKER_FEE_RATE*100}%, RiskPerTrade:{PERCENT_CAPITAL_TO_RISK*100}%")
    print("---" * 10)

    data_df_raw = get_historical_data(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR)

    if not data_df_raw.empty:
        data_with_indicators = add_indicators(data_df_raw.copy())
        if not data_with_indicators.empty:
            trading_signals_df = generate_signals(data_with_indicators)
            portfolio_history_df, backtest_summary = backtest_strategy( # Pass new param
                data_with_indicators, trading_signals_df, INITIAL_CAPITAL, PERCENT_CAPITAL_TO_RISK
            )

            print("\n--- Backtest Results ---")
            print(f"Initial Capital: ${backtest_summary['initial_capital']:.2f}")
            print(f"Final Portfolio Value: ${backtest_summary['final_portfolio_value']:.2f}")
            print(f"Total Return (Net): ${backtest_summary['total_return_usd']:.2f} ({backtest_summary['total_return_pct']:.2f}%)")
            print(f"Number of Trades Executed: {backtest_summary['num_trades']}")
            print(f"Total Fees Paid: ${backtest_summary['total_fees_paid']:.2f}")

            print("\n--- Trades Log (with fees & dynamic sizing) ---")
            trades_df = backtest_summary['trades_log_df']
            if not trades_df.empty:
                trades_df_display = trades_df.copy()
                trades_df_display['date'] = pd.to_datetime(trades_df_display['date']).dt.strftime('%Y-%m-%d')
                for col in ['price', 'cost', 'capital_after', 'proceeds', 'fees']:
                    if col in trades_df_display.columns:
                         trades_df_display[col] = trades_df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
                print(trades_df_display.to_string())
            else: print("No trades were executed.")

            if not portfolio_history_df.empty:
                try:
                    plt.figure(figsize=(14, 7)); plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label='Strategy Equity (Net, %Risk Sizing)', alpha=0.7)
                    buy_dates = trades_df[trades_df['type'] == 'BUY']['date']; sell_dates = trades_df[trades_df['type'] == 'SELL']['date']
                    if not portfolio_history_df.index.is_monotonic_increasing: portfolio_history_df = portfolio_history_df.sort_index()
                    buy_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(buy_dates)]['value']
                    sell_values = portfolio_history_df.loc[portfolio_history_df.index.intersection(sell_dates)]['value']
                    plt.plot(buy_values.index, buy_values, '^', markersize=10, color='g', lw=0, label='Buy Signal')
                    plt.plot(sell_values.index, sell_values, 'v', markersize=10, color='r', lw=0, label='Sell Signal')
                    plt.title(f'{SYMBOL} Strat Equity ({START_DATE_STR} to {END_DATE_STR}) - ATR Stop ({ATR_STOP_MULTIPLIER}x) - Net of Fees - {PERCENT_CAPITAL_TO_RISK*100}% Risk'); plt.ylabel('Portfolio Value (USD)'); plt.xlabel('Date'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
                except Exception as e: print(f"\nError during plotting: {e}")
        else: print("Could not generate indicators. Aborting.")
    else: print("No data fetched. Aborting.")
    print("\n--- Script Finished ---")