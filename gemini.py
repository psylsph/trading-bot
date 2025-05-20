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
TIMEFRAME = tradeapi.TimeFrame.Hour # UPDATED TO HOURLY

START_DATE_STR = (date.today() - timedelta(days=400)).strftime('%Y-%m-%d')
END_DATE_STR = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# Strategy Parameters (These will now apply to hourly data - Option A1 "Faster")
BB_LENGTH = 24
BB_STD_DEV = 2.0
RSI_LENGTH = 14
RSI_OVERSOLD = 20 # Adjusted from 35 for potentially more signals/faster reaction
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.7 # Adjusted from 2.25, can be tuned (e.g., 1.5 for tighter)

TAKER_FEE_RATE = 0.0025 # 0.25% Taker Fee

INITIAL_CAPITAL = 1000.00
PERCENT_CAPITAL_TO_RISK = 0.015 # Risk of current capital per trade

MIN_RISK_PER_UNIT_THRESHOLD = 0.00001

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
        end_dt_inclusive_day = utc.localize(datetime.strptime(end_str, '%Y-%m-%d'))
        end_dt_api_exclusive = end_dt_inclusive_day + timedelta(days=1)

        bars = api.get_crypto_bars(symbol, timeframe, start=start_dt.isoformat(), end=end_dt_api_exclusive.isoformat()).df
        
        if bars.empty:
            print("No data fetched.")
            return pd.DataFrame()

        if not bars.index.tz:
            bars = bars.tz_localize('UTC')
        else:
            bars = bars.tz_convert('UTC')

        end_of_day_filter = end_dt_inclusive_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        bars = bars[bars.index <= end_of_day_filter]
        
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if 'vwap' in bars.columns:
            required_ohlcv.append('vwap')
        bars = bars[required_ohlcv]
        
        print(f"Fetched {len(bars)} bars. Actual date range: {bars.index.min()} to {bars.index.max()}")
        return bars
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- 2. Calculate Indicators ---
def add_indicators(df):
    if df.empty: return df
    print("\n--- Adding Indicators ---")
    df.ta.bbands(length=BB_LENGTH, std=BB_STD_DEV, append=True)
    lower_band_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
    middle_band_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'
    df.ta.rsi(length=RSI_LENGTH, append=True)
    if 'vwap' in df.columns:
        df.rename(columns={'vwap': 'VWAP'}, inplace=True)
    else:
        df.ta.vwap(append=True)
        df.rename(columns={'VWAP_D': 'VWAP'}, inplace=True, errors='ignore')
    df.ta.atr(length=ATR_PERIOD, append=True)
    atr_col = f'ATRr_{ATR_PERIOD}'
    print(f"Added ATR: {atr_col}")
    print(f"Columns after adding all indicators: {df.columns.tolist()}")
    required_cols_check = ['close', 'low', lower_band_col, middle_band_col, f'RSI_{RSI_LENGTH}', atr_col]
    missing_cols = [col for col in required_cols_check if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing critical indicator columns BEFORE dropna: {missing_cols}")
    df.dropna(inplace=True)
    print(f"Shape of data after indicators & dropna: {df.shape}")
    if df.empty:
        print("DataFrame empty after indicators & dropna.")
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
        print(f"Error: Missing cols for signal gen. Req: {required_cols}. Avail: {df.columns.tolist()}")
        return signals
    
    cond1_buy_lbb = (df['close'] < df[lower_band_col])
    cond2_buy_rsi = (df[rsi_col] < RSI_OVERSOLD)
    buy_condition = cond1_buy_lbb & cond2_buy_rsi
    
    cond_sell_profit_middlebb = (df['close'] > df[middle_band_col])
    
    signals.loc[cond_sell_profit_middlebb, 'sell_reason'] = 'MiddleBB_PT'
    signals.loc[signals['sell_reason'] != '', 'signal'] = -1
    
    signals.loc[buy_condition, 'signal'] = 1
    signals.loc[buy_condition, 'sell_reason'] = ''
    
    print(f"RSI Oversold: {RSI_OVERSOLD}")
    print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER} (Stop logic in backtester)")
    print(f"Number of potential buy signal points (raw): {buy_condition.sum()}")
    print(f"Number of potential PROFIT TAKE triggers (MiddleBB): {cond_sell_profit_middlebb.sum()}")
    print(f"Total BUY signals generated: {len(signals[signals['signal'] == 1])}")
    print(f"Total PROFIT TAKE signals generated: {len(signals[signals['sell_reason'] != ''])}")
    return signals

# --- 4. Implement Backtester with ATR Stop, Transaction Costs, and Percent Risk Sizing ---
def backtest_strategy(df_with_indicators, signals_df, initial_capital, percent_capital_to_risk):
    print("\n--- Running Backtest with ATR Stop, Transaction Costs & Percent Risk Sizing ---")
    if df_with_indicators.empty or signals_df.empty:
        print("Cannot backtest: empty data/signals.")
        return pd.DataFrame(), {}
    
    capital = initial_capital
    positions_asset_units = 0.0
    portfolio_value_history = []
    trades_log = [] 
    in_position = False
    current_entry_price_actual = 0.0
    current_atr_stop_level = 0.0
    atr_col = f'ATRr_{ATR_PERIOD}'

    for i in range(len(df_with_indicators)):
        current_date = df_with_indicators.index[i]
        current_price_market = df_with_indicators['close'].iloc[i]
        current_low_price_market = df_with_indicators['low'].iloc[i]
        
        current_signal_row = signals_df.loc[signals_df.index == current_date]
        if current_signal_row.empty:
            profit_take_signal_action = 0
            profit_take_reason = ''
        else:
            profit_take_signal_action = current_signal_row['signal'].iloc[0]
            profit_take_reason = current_signal_row['sell_reason'].iloc[0]

        atr_on_bar = df_with_indicators[atr_col].iloc[i]

        if in_position:
            if current_low_price_market <= current_atr_stop_level:
                sell_price_execution = current_atr_stop_level
                proceeds_before_fees = positions_asset_units * sell_price_execution
                fees_on_sell = proceeds_before_fees * TAKER_FEE_RATE
                proceeds_after_fees = proceeds_before_fees - fees_on_sell
                capital += proceeds_after_fees
                trades_log.append({'date': current_date, 'type': 'SELL', 'price': sell_price_execution, 
                                   'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital, 
                                   'proceeds': proceeds_after_fees, 'fees': fees_on_sell, 'reason': 'ATR_Stop' })
                positions_asset_units = 0.0; in_position = False; current_entry_price_actual = 0.0; current_atr_stop_level = 0.0
            elif profit_take_signal_action == -1 and profit_take_reason != '':
                sell_price_execution = current_price_market
                proceeds_before_fees = positions_asset_units * sell_price_execution
                fees_on_sell = proceeds_before_fees * TAKER_FEE_RATE
                proceeds_after_fees = proceeds_before_fees - fees_on_sell
                capital += proceeds_after_fees
                trades_log.append({'date': current_date, 'type': 'SELL', 'price': sell_price_execution, 
                                   'units': positions_asset_units, 'cost': pd.NA, 'capital_after': capital, 
                                   'proceeds': proceeds_after_fees, 'fees': fees_on_sell, 'reason': profit_take_reason })
                positions_asset_units = 0.0; in_position = False; current_entry_price_actual = 0.0; current_atr_stop_level = 0.0

        if not current_signal_row.empty and current_signal_row['signal'].iloc[0] == 1 and not in_position and capital > 0:
            current_entry_price_actual = current_price_market
            potential_atr_stop_level = current_entry_price_actual - (ATR_STOP_MULTIPLIER * atr_on_bar)
            risk_per_unit_asset = current_entry_price_actual - potential_atr_stop_level

            if risk_per_unit_asset > MIN_RISK_PER_UNIT_THRESHOLD:
                dollar_amount_to_risk = capital * percent_capital_to_risk
                units_to_buy_ideal = dollar_amount_to_risk / risk_per_unit_asset
                cost_before_fees_ideal = units_to_buy_ideal * current_entry_price_actual
                fees_on_buy_ideal = cost_before_fees_ideal * TAKER_FEE_RATE
                total_cost_with_fees_ideal = cost_before_fees_ideal + fees_on_buy_ideal

                if total_cost_with_fees_ideal > 0 and capital >= total_cost_with_fees_ideal:
                    positions_asset_units = units_to_buy_ideal
                    capital -= total_cost_with_fees_ideal 
                    in_position = True
                    current_atr_stop_level = potential_atr_stop_level
                    trades_log.append({'date': current_date, 'type': 'BUY', 'price': current_entry_price_actual, 
                                       'units': positions_asset_units, 'cost': total_cost_with_fees_ideal, 
                                       'capital_after': capital, 'proceeds': pd.NA, 'fees': fees_on_buy_ideal, 'reason': '' })
        
        current_portfolio_value = capital + (positions_asset_units * current_price_market)
        portfolio_value_history.append({'date': current_date, 'value': current_portfolio_value})

    portfolio_df = pd.DataFrame(portfolio_value_history).set_index('date')
    final_val = portfolio_df['value'].iloc[-1] if not portfolio_df.empty else initial_capital
    ret_abs = final_val - initial_capital
    ret_pct = (ret_abs / initial_capital) * 100 if initial_capital > 0 else 0
    num_trades = len([t for t in trades_log if t['type'] == 'BUY'])
    total_fees_paid = sum(t['fees'] for t in trades_log if pd.notna(t['fees']))
    
    results = {
        'initial_capital': initial_capital, 
        'final_portfolio_value': final_val, 
        'total_return_usd': ret_abs, 
        'total_return_pct': ret_pct, 
        'num_trades': num_trades, 
        'total_fees_paid': total_fees_paid, 
        'trades_log_df': pd.DataFrame(trades_log, columns=['date', 'type', 'price', 'units', 'cost', 'capital_after', 'proceeds', 'fees', 'reason'])
    }
    return portfolio_df, results

# --- 5. Run and Analyze ---
if __name__ == "__main__":
    print(f"Starting trading strategy backtest for {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME.value}")
    print(f"Period: {START_DATE_STR} to {END_DATE_STR}")
    print(f"Parameters: BB({BB_LENGTH},{BB_STD_DEV}), RSI_OS:{RSI_OVERSOLD}, ATR_Stop({ATR_PERIOD}, Multiplier:{ATR_STOP_MULTIPLIER}), TakerFee:{TAKER_FEE_RATE*100}%, RiskPerTrade:{PERCENT_CAPITAL_TO_RISK*100}%")
    print("---" * 10)

    data_df_raw = get_historical_data(SYMBOL, TIMEFRAME, START_DATE_STR, END_DATE_STR)

    if not data_df_raw.empty:
        data_with_indicators = add_indicators(data_df_raw.copy())
        
        if not data_with_indicators.empty and not data_with_indicators.index.has_duplicates:
            trading_signals_df_unaligned = generate_signals(data_with_indicators)
            trading_signals_df = trading_signals_df_unaligned.reindex(data_with_indicators.index, fill_value=0)
            if 'sell_reason' in trading_signals_df.columns:
                 trading_signals_df['sell_reason'] = trading_signals_df_unaligned['sell_reason'].reindex(data_with_indicators.index, fill_value='')

            portfolio_history_df, backtest_summary = backtest_strategy(
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
                trades_df_display['date'] = pd.to_datetime(trades_df_display['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                for col in ['price', 'cost', 'capital_after', 'proceeds', 'fees', 'units']:
                    if col in trades_df_display.columns:
                         trades_df_display[col] = trades_df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
                print(trades_df_display.to_string(index=False))
            else:
                print("No trades were executed.")

            if not portfolio_history_df.empty:
                try:
                    plt.figure(figsize=(14, 7))
                    plt.plot(portfolio_history_df.index, portfolio_history_df['value'], label='Strategy Equity (Net, %Risk Sizing)', alpha=0.7)
                    
                    if not trades_df.empty:
                        buy_dates = trades_df[trades_df['type'] == 'BUY']['date']
                        sell_dates = trades_df[trades_df['type'] == 'SELL']['date']
                        buy_dates_dt = pd.to_datetime(buy_dates)
                        sell_dates_dt = pd.to_datetime(sell_dates)
                        if not isinstance(portfolio_history_df.index, pd.DatetimeIndex):
                            portfolio_history_df.index = pd.to_datetime(portfolio_history_df.index)
                        
                        valid_buy_dates = buy_dates_dt[buy_dates_dt.isin(portfolio_history_df.index)]
                        valid_sell_dates = sell_dates_dt[sell_dates_dt.isin(portfolio_history_df.index)]

                        if not valid_buy_dates.empty:
                             buy_values = portfolio_history_df.loc[valid_buy_dates, 'value']
                             plt.plot(buy_values.index, buy_values, '^', markersize=8, color='g', lw=0, label='Buy Signal')
                        if not valid_sell_dates.empty:
                             sell_values = portfolio_history_df.loc[valid_sell_dates, 'value']
                             plt.plot(sell_values.index, sell_values, 'v', markersize=8, color='r', lw=0, label='Sell Signal')
                    
                    plt.title(f'{SYMBOL} Strat Equity ({TIMEFRAME.value}) - {START_DATE_STR} to {END_DATE_STR}\nATR Stop ({ATR_STOP_MULTIPLIER}x) - Net of Fees - {PERCENT_CAPITAL_TO_RISK*100}% Risk')
                    plt.ylabel('Portfolio Value (USD)')
                    plt.xlabel('Date')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"\nError during plotting: {e}")
            else:
                print("Portfolio history is empty, cannot plot.")
        elif data_with_indicators.index.has_duplicates:
            print("Data with indicators has duplicate index entries. Aborting backtest.")
            print("Duplicates:", data_with_indicators.index[data_with_indicators.index.duplicated()].unique())
        else:
            print("Could not generate indicators or data is empty after indicators. Aborting.")
    else:
        print("No data fetched. Aborting.")
    print("\n--- Script Finished ---")