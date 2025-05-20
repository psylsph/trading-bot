import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
import time
import os
from alpaca.data.requests import CryptoLatestQuoteRequest
from alpaca.data.historical import CryptoHistoricalDataClient
# import pytz # pytz is not explicitly used, timezone from datetime is sufficient

from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets') # Use paper trading URL

SYMBOL_ALPACA = 'XRPUSD' # Alpaca often uses no slash for crypto orders
SYMBOL_DATA = 'XRP/USD' # For fetching bars (Alpaca data API uses '/')

# Risk Management
RISK_PER_TRADE_PERCENT = 1.0  # Risk 1% of equity per trade
MIN_NOTIONAL_USD = 1.0 # Minimum order value in USD for crypto on Alpaca

# Strategy Parameters (same as your best backtest)
BB_LENGTH = 20
BB_STD_DEV = 2.0
RSI_LENGTH = 14
RSI_OVERSOLD = 35
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.25

# State Variables (global or managed in a class) - In a real app, use a database or file
in_position = False
current_entry_price = 0.0
current_position_qty = 0.0
current_atr_stop_level = 0.0
last_known_atr_at_entry = 0.0 # Store ATR at time of entry

# --- Alpaca API Setup ---
try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    initial_account_info = api.get_account() # Fetch account info once at start for initial checks
    print(f"Connected to Alpaca Paper Trading. Account: {initial_account_info.account_number}, Status: {initial_account_info.status}, Equity: ${initial_account_info.equity}")
    if initial_account_info.trading_blocked:
        print("ACCOUNT IS TRADING BLOCKED.")
        exit()
except Exception as e:
    print(f"Error connecting to Alpaca: {e}")
    exit()

def get_latest_bars(symbol, timeframe, limit=200):
    """Fetches the most recent N bars for calculations."""
    try:
        now_utc = datetime.now(timezone.utc)
        # Calculate start date approximately based on limit (conservative)
        start_dt_approx = now_utc - timedelta(days=limit + 50) # Generous buffer for non-trading days etc.

        bars_df = api.get_crypto_bars(
            symbol,
            timeframe,
            start=start_dt_approx.isoformat(),
            end=now_utc.isoformat()
        ).df

        if bars_df.empty:
            print(f"No bars returned for {symbol}")
            return pd.DataFrame()

        if not bars_df.index.tz: bars_df = bars_df.tz_localize('UTC')
        else: bars_df = bars_df.tz_convert('UTC')

        bars_df = bars_df[['open', 'high', 'low', 'close', 'volume']]
        return bars_df.tail(limit)
    except Exception as e:
        print(f"Error fetching latest bars for {symbol}: {e}")
        return pd.DataFrame()

def add_live_indicators(df):
    """Adds indicators to the DataFrame."""
    if df.empty or len(df) < max(BB_LENGTH, RSI_LENGTH, ATR_PERIOD) + 5: # Need enough data
        print("Not enough data to calculate all indicators.")
        return pd.DataFrame()

    df_copy = df.copy() # Work on a copy
    df_copy.ta.bbands(length=BB_LENGTH, std=BB_STD_DEV, append=True)
    df_copy.ta.rsi(length=RSI_LENGTH, append=True)
    df_copy.ta.atr(length=ATR_PERIOD, append=True) # ATR result is 'ATRr_LENGTH'
    df_copy.dropna(inplace=True)
    return df_copy

def get_current_market_price(symbol_alpaca, fallback_price):
    """Gets the current ask price, with fallback."""
    try:
        alpaca_client = CryptoHistoricalDataClient()
        request_params = CryptoLatestQuoteRequest(symbol_or_symbols=SYMBOL_DATA)
        latest_quote = alpaca_client.get_crypto_latest_quote(request_params)

        quote = latest_quote[SYMBOL_DATA]

        if quote and hasattr(quote, 'bid_price') and quote.bid_price is not None:
            return float(quote.bid_price)
        else:
            print(f"Could not get current market price (ask price was None or quote was invalid) for {symbol_alpaca}. Using fallback: {fallback_price}")
            return float(fallback_price)
    except Exception as e:
        print(f"Error fetching current market price for {symbol_alpaca}: {e}. Using fallback: {fallback_price}")
        return float(fallback_price)


def check_signals_and_trade():
    global in_position, current_entry_price, current_position_qty, current_atr_stop_level, last_known_atr_at_entry

    print(f"\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}: Checking for signals for {SYMBOL_DATA}...")
    bars = get_latest_bars(SYMBOL_DATA, tradeapi.TimeFrame.Day, limit=BB_LENGTH + RSI_LENGTH + ATR_PERIOD + 50) # Fetch enough data

    if bars.empty or len(bars) < 1:
        print("Could not fetch market data or data is empty.")
        return

    data_with_indicators = add_live_indicators(bars)
    if data_with_indicators.empty:
        print("Could not calculate indicators.")
        return

    latest_data = data_with_indicators.iloc[-1] # Current (just closed) D1 bar's data
    # previous_data = data_with_indicators.iloc[-2] # Previous D1 bar (if needed for crossover, etc.)

    # Fetch current market price for decision making and order placement
    # Fallback to latest close if live quote fails
    market_price_for_trade = get_current_market_price(SYMBOL_ALPACA, latest_data['close'])

    print(f"Latest D1 close: {latest_data['close']:.4f}, Current Market Ask (or fallback): {market_price_for_trade:.4f}")

    # --- EXIT LOGIC (Check before entry logic) ---
    if in_position:
        # Recalculate stop level based on stored entry price and ATR at entry
        if last_known_atr_at_entry > 0 and current_entry_price > 0:
             live_atr_stop = current_entry_price - (ATR_STOP_MULTIPLIER * last_known_atr_at_entry)
        else: # Fallback if somehow ATR at entry wasn't stored (should not happen)
            live_atr_stop = 0 # effectively no stop or handle error
            print("Warning: ATR at entry was not properly set for current position. Cannot determine ATR stop.")

        print(f"Position active. Entry: {current_entry_price:.4f}, Qty: {current_position_qty}, ATR at Entry: {last_known_atr_at_entry:.4f} Current ATR Stop Level: {live_atr_stop:.4f}, Last Low: {latest_data['low']:.4f}")

        # 1. Check ATR Stop
        # For live, you might want to check more frequently than D1 close if stop is hit intraday
        # Here, we simplify and check against latest_data['low'] or current_market_price
        if live_atr_stop > 0 and latest_data['low'] <= live_atr_stop : # Check if the D1 low breached the stop
            print(f"ATR STOP-LOSS triggered for {SYMBOL_ALPACA} at <= {live_atr_stop:.4f} (Last Low: {latest_data['low']:.4f})!")
            try:
                api.submit_order(
                    symbol=SYMBOL_ALPACA,
                    qty=current_position_qty, # Sell the entire current position
                    side='sell',
                    type='market',
                    time_in_force='gtc' # Good 'til canceled for market orders is fine
                )
                print(f"SELL order (ATR Stop) for {current_position_qty} {SYMBOL_ALPACA} submitted.")
                # Reset state variables
                in_position = False; current_entry_price = 0.0; current_position_qty = 0.0; current_atr_stop_level = 0.0; last_known_atr_at_entry = 0.0
            except Exception as e:
                print(f"Error submitting SELL order (ATR Stop): {e}")
            return # Exit after attempting to close position

        # 2. Check Profit Take (Middle Bollinger Band)
        middle_bb_col = f'BBM_{BB_LENGTH}_{BB_STD_DEV}'
        if latest_data['close'] > latest_data[middle_bb_col]:
            print(f"MiddleBB PROFIT TARGET triggered for {SYMBOL_ALPACA} at > {latest_data[middle_bb_col]:.4f} (Close: {latest_data['close']:.4f})!")
            try:
                api.submit_order(
                    symbol=SYMBOL_ALPACA,
                    qty=current_position_qty, # Sell the entire current position
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"SELL order (MiddleBB PT) for {current_position_qty} {SYMBOL_ALPACA} submitted.")
                # Reset state variables
                in_position = False; current_entry_price = 0.0; current_position_qty = 0.0; current_atr_stop_level = 0.0; last_known_atr_at_entry = 0.0
            except Exception as e:
                print(f"Error submitting SELL order (MiddleBB PT): {e}")
            return

    # --- ENTRY LOGIC ---
    if not in_position:
        lower_bb_col = f'BBL_{BB_LENGTH}_{BB_STD_DEV}'
        rsi_col = f'RSI_{RSI_LENGTH}'
        atr_col = f'ATRr_{ATR_PERIOD}' # pandas_ta appends 'r' to ATR

        buy_cond_lbb = latest_data['close'] < latest_data[lower_bb_col]
        buy_cond_rsi = latest_data[rsi_col] < RSI_OVERSOLD

        print(f"Checking Buy: Close < LBB ({latest_data['close']:.4f} < {latest_data[lower_bb_col]:.4f})? {buy_cond_lbb}. RSI < OS ({latest_data[rsi_col]:.2f} < {RSI_OVERSOLD})? {buy_cond_rsi}")

        if buy_cond_lbb and buy_cond_rsi:
            print(f"BUY SIGNAL for {SYMBOL_ALPACA} detected!")

            try:
                # --- Position Sizing based on Risk ---
                account_info = api.get_account() # Get fresh account info for equity
                account_equity = float(account_info.equity)
                buying_power = float(account_info.buying_power) # Available cash for trading

                max_risk_usd_for_trade = account_equity * (RISK_PER_TRADE_PERCENT / 100.0)
                print(f"Account Equity: ${account_equity:.2f}, Risk per trade: {RISK_PER_TRADE_PERCENT}%, Max Risk USD: ${max_risk_usd_for_trade:.2f}")

                current_atr_value = latest_data[atr_col]
                if pd.isna(current_atr_value) or current_atr_value <= 0:
                    print(f"Invalid ATR value ({current_atr_value:.4f}) at entry. Cannot calculate position size. Skipping trade.")
                    return

                # Risk per unit (e.g., per XRP) is based on ATR stop distance
                # Entry price for calculation will be current market price
                potential_entry_price = market_price_for_trade
                stop_loss_distance_per_unit = ATR_STOP_MULTIPLIER * current_atr_value

                if stop_loss_distance_per_unit <= 1e-8: # Effectively zero or too small (e.g. for stablecoins or errors)
                    print(f"Stop loss distance per unit ({stop_loss_distance_per_unit:.8f}) is too small. ATR might be near zero. Skipping trade.")
                    return

                calculated_qty = max_risk_usd_for_trade / stop_loss_distance_per_unit
                calculated_qty = round(calculated_qty, 8) # Round to typical crypto precision (adjust if needed for specific asset)

                notional_value = calculated_qty * potential_entry_price

                print(f"Potential Entry: {potential_entry_price:.4f}, ATR: {current_atr_value:.4f}, Stop Distance/Unit: {stop_loss_distance_per_unit:.4f}")
                print(f"Calculated Qty: {calculated_qty}, Notional Value: ${notional_value:.2f}")

                if calculated_qty <= 0:
                    print(f"Calculated quantity ({calculated_qty}) is zero or negative. Skipping trade.")
                    return

                if notional_value < MIN_NOTIONAL_USD: # Alpaca has min notional for crypto orders
                    print(f"Calculated notional value (${notional_value:.2f}) is below minimum ${MIN_NOTIONAL_USD:.2f}. Skipping trade.")
                    return

                if notional_value > buying_power:
                    print(f"Calculated notional value (${notional_value:.2f}) exceeds buying power (${buying_power:.2f}). Reducing quantity.")
                    # Option 1: Reduce qty to fit buying power (while respecting risk less)
                    # calculated_qty = round((buying_power * 0.98) / potential_entry_price, 8) # Use 98% of BP
                    # Option 2: Skip trade (safer from risk management perspective if risk calc is primary)
                    print("Skipping trade as it would exceed buying power.")
                    return


                # Submit buy order
                print(f"Attempting to buy {calculated_qty} of {SYMBOL_ALPACA} (Notional: ~${notional_value:.2f}).")
                order = api.submit_order(
                    symbol=SYMBOL_ALPACA,
                    qty=calculated_qty,
                    side='buy',
                    type='market',
                    time_in_force='day' # Good for D1 strategy, ensures it fills or cancels by EOD
                )
                print(f"BUY order for {calculated_qty} {SYMBOL_ALPACA} submitted. Order ID: {order.id}")

                # Update state (assuming market order will fill close to market_price_for_trade)
                # These values might be refined by check_and_update_open_positions once fill confirmed
                in_position = True
                current_entry_price = potential_entry_price # Use the price used for sizing as initial entry
                current_position_qty = calculated_qty # Use calculated qty; will be updated by actual fill
                last_known_atr_at_entry = current_atr_value
                current_atr_stop_level = potential_entry_price - stop_loss_distance_per_unit
                print(f"Position opened (tentatively). Entry: {current_entry_price:.4f}, Qty: {current_position_qty}, ATR@Entry: {last_known_atr_at_entry:.4f}, Initial Stop: {current_atr_stop_level:.4f}")

            except Exception as e:
                print(f"Error submitting BUY order or during position sizing: {e}")
        else:
            print("No buy signal.")
    else:
        print("Already in position. Holding.")


def check_and_update_open_positions():
    """Checks current Alpaca positions and updates bot's internal state if needed."""
    global in_position, current_entry_price, current_position_qty, current_atr_stop_level, last_known_atr_at_entry
    try:
        positions = api.list_positions()
        asset_position = next((p for p in positions if p.symbol == SYMBOL_ALPACA), None)

        if asset_position:
            actual_qty = float(asset_position.qty)
            actual_entry_price = float(asset_position.avg_entry_price)

            if not in_position: # Bot thought it wasn't in position, but Alpaca says it is
                print(f"Discrepancy: Alpaca shows position in {SYMBOL_ALPACA}, bot state was 'not in_position'. Updating state.")
                in_position = True
                current_position_qty = actual_qty
                current_entry_price = actual_entry_price
                print(f"State updated from Alpaca: Qty={current_position_qty}, AvgEntry={current_entry_price:.4f}")
                # If last_known_atr_at_entry is 0, it means the bot restarted or lost state for this specific entry.
                # We need to re-establish it. This is tricky for a LIVE running bot if it wasn't persisted.
                if last_known_atr_at_entry == 0.0 and current_entry_price > 0:
                    print("Warning: 'last_known_atr_at_entry' is 0. Attempting to set a temporary stop using current ATR.")
                    # This is an approximation. Ideally, ATR at entry should be persisted.
                    bars_for_atr = get_latest_bars(SYMBOL_DATA, tradeapi.TimeFrame.Day, limit=ATR_PERIOD + 5)
                    if not bars_for_atr.empty:
                        # Pass only necessary columns to avoid errors if other columns are missing
                        df_with_atr = add_live_indicators(bars_for_atr[['open', 'high', 'low', 'close', 'volume']])
                        if not df_with_atr.empty:
                            atr_col_name = f'ATRr_{ATR_PERIOD}'
                            if atr_col_name in df_with_atr.columns and not pd.isna(df_with_atr[atr_col_name].iloc[-1]):
                                current_live_atr = df_with_atr[atr_col_name].iloc[-1]
                                last_known_atr_at_entry = current_live_atr # Use current ATR as best guess
                                current_atr_stop_level = current_entry_price - (ATR_STOP_MULTIPLIER * last_known_atr_at_entry)
                                print(f"Temporary ATR stop set using current ATR ({current_live_atr:.4f}): {current_atr_stop_level:.4f}")
                            else:
                                print("Could not get current ATR to set temporary stop (ATR column missing or NaN).")
                        else:
                             print("Could not calculate indicators for temporary ATR stop.")
            else: # Bot is in position, sync with Alpaca's view
                # Compare floats with a small tolerance
                if abs(current_position_qty - actual_qty) > 1e-8 or abs(current_entry_price - actual_entry_price) > 1e-8 :
                    print(f"Syncing position details with Alpaca: Qty Alpaca: {actual_qty} vs Bot: {current_position_qty}, Entry Alpaca: {actual_entry_price:.4f} vs Bot: {current_entry_price:.4f}")
                    current_position_qty = actual_qty
                    current_entry_price = actual_entry_price
                    # Recalculate stop level with actual entry price if ATR at entry is known
                    if last_known_atr_at_entry > 0:
                        current_atr_stop_level = current_entry_price - (ATR_STOP_MULTIPLIER * last_known_atr_at_entry)
                        print(f"Position details updated. New Stop Level: {current_atr_stop_level:.4f}")

        else: # No position in Alpaca
            if in_position: # Bot thought it was in position, but Alpaca says no (e.g., manual close, or stop filled)
                print(f"Discrepancy: Bot state was 'in_position', but Alpaca shows no position in {SYMBOL_ALPACA}. Resetting state.")
                in_position = False; current_entry_price = 0.0; current_position_qty = 0.0; current_atr_stop_level = 0.0; last_known_atr_at_entry = 0.0

        # Display current bot state clearly
        stop_display = "N/A"
        if in_position:
            if last_known_atr_at_entry > 0 and current_atr_stop_level > 0:
                stop_display = f"{current_atr_stop_level:.4f}"
            else:
                stop_display = "N/A (ATR at entry unknown or stop not set)"

        print(f"Current bot state: In Position = {in_position}, Qty = {current_position_qty}, Entry = {current_entry_price:.4f}, Stop = {stop_display}")

    except Exception as e:
        print(f"Error checking/updating open positions: {e}")


if __name__ == "__main__":
    print("Performing initial position check with Alpaca...")
    check_and_update_open_positions()

    # For D1 strategy: stores the datetime of the last execution
    last_d1_strategy_execution_time = None

    # For 15-minute sync: flag to ensure it runs once per target minute
    has_run_15_min_sync_for_current_target_minute = False

    LOOP_SLEEP_SECONDS = 60 # Check conditions every 60 seconds

    print(f"Starting main trading loop. Checks will occur approximately every {LOOP_SLEEP_SECONDS} seconds.")

    while True:
        now_utc = datetime.now(timezone.utc)
        current_minute_of_hour = now_utc.minute

        # --- Periodic 15-minute position sync ---
        # Target minutes for sync: 0, 15, 30, 45
        is_target_sync_minute = (current_minute_of_hour % 15 == 0)

        if is_target_sync_minute:
            if not has_run_15_min_sync_for_current_target_minute:
                print(f"\n--- Interval Position Sync ({now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
                check_and_update_open_positions()
                has_run_15_min_sync_for_current_target_minute = True
        else:
            # If it's no longer a target minute, reset the flag so it's ready for the next one.
            if has_run_15_min_sync_for_current_target_minute:
                has_run_15_min_sync_for_current_target_minute = False


        # --- D1 Strategy Logic ---
        # Run once per day, shortly after 00:05 UTC.
        run_d1_strategy_now = False
        if last_d1_strategy_execution_time is None:
            # Bot just started. If it's already after 00:05 UTC today, run the D1 check.
            if now_utc.hour >= 0 and now_utc.minute >= 5:
                run_d1_strategy_now = True
        else:
            # Check if it's a new day compared to the last execution day,
            # AND it's after 00:05 UTC on this new day.
            if now_utc.date() > last_d1_strategy_execution_time.date():
                if now_utc.hour == 0 and now_utc.minute >= 5:
                    run_d1_strategy_now = True

        if run_d1_strategy_now:
            print(f"\n--- Daily Strategy Check ({now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
            check_signals_and_trade()
            last_d1_strategy_execution_time = now_utc # Record the time of this execution
            print(f"Strategy check complete for day {now_utc.strftime('%Y-%m-%d')}. Next D1 check after {(now_utc.date() + timedelta(days=1)).strftime('%Y-%m-%d')} 00:05 UTC.")

        # Main loop sleep
        print(f"Loop end. Sleeping for {LOOP_SLEEP_SECONDS}s. Next check around: {(now_utc + timedelta(seconds=LOOP_SLEEP_SECONDS)).strftime('%H:%M:%S')}")
        time.sleep(LOOP_SLEEP_SECONDS)