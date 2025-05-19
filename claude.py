import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure Alpaca API
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing

class BollingerRSIVWAPStrategy:
    def __init__(self, symbol='XRPUSD', timeframe='1H'):
        """
        Initialize the trading strategy.
        
        Parameters:
        - symbol: Trading pair symbol (default: XRPUSD)
        - timeframe: Data timeframe (default: 1H)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        # Strategy parameters
        self.bb_window = 20
        self.bb_std = 2
        self.rsi_window = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.vwap_window = 14
        
        # Performance tracking
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []

    def fetch_data(self, start_date, end_date):
        """
        Fetch historical data from Alpaca.
        
        Parameters:
        - start_date: Start date for data fetching
        - end_date: End date for data fetching
        
        Returns:
        - DataFrame with historical data
        """
        # Convert dates to ISO format
        start_date_iso = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_iso = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Fetch data from Alpaca
        bars = self.api.get_crypto_bars(
            self.symbol, 
            self.timeframe,
            start=start_date_iso,
            end=end_date_iso
        ).df
        
        return bars

    def calculate_indicators(self, data):
        """
        Calculate technical indicators for the strategy.
        
        Parameters:
        - data: DataFrame with historical price data
        
        Returns:
        - DataFrame with added indicators
        """
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.bb_window).mean()
        df['std'] = df['close'].rolling(window=self.bb_window).std()
        df['upper_band'] = df['sma'] + (df['std'] * self.bb_std)
        df['lower_band'] = df['sma'] - (df['std'] * self.bb_std)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['volume_x_price'] = df['typical_price'] * df['volume']
        
        df['cum_volume'] = df['volume'].rolling(window=self.vwap_window).sum()
        df['cum_volume_x_price'] = df['volume_x_price'].rolling(window=self.vwap_window).sum()
        
        df['vwap'] = df['cum_volume_x_price'] / df['cum_volume']
        
        # Drop NaN values
        df = df.dropna()
        
        return df

    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy.
        
        Parameters:
        - data: DataFrame with price data and indicators
        
        Returns:
        - DataFrame with added signals column
        """
        df = data.copy()
        
        # Initialize signals column
        df['signal'] = 0
        
        # Buy signal: Price below lower Bollinger Band, RSI oversold, price below VWAP
        buy_condition = (
            (df['close'] < df['lower_band']) & 
            (df['rsi'] < self.rsi_oversold) & 
            (df['close'] < df['vwap'])
        )
        
        # Sell signal: Price above upper Bollinger Band, RSI overbought, price above VWAP
        sell_condition = (
            (df['close'] > df['upper_band']) & 
            (df['rsi'] > self.rsi_overbought) & 
            (df['close'] > df['vwap'])
        )
        
        # Set signals
        df.loc[buy_condition, 'signal'] = 1  # Buy
        df.loc[sell_condition, 'signal'] = -1  # Sell
        
        return df

    def backtest(self, data):
        """
        Backtest the strategy on historical data.
        
        Parameters:
        - data: DataFrame with price data, indicators, and signals
        
        Returns:
        - Performance metrics
        """
        df = data.copy()
        
        # Initialize columns
        df['position'] = 0
        df['cash'] = self.initial_balance
        df['holdings'] = 0
        df['total'] = self.initial_balance
        
        position = 0
        cash = self.initial_balance
        
        for index, row in df.iterrows():
            # Check for buy signal
            if row['signal'] == 1 and position == 0:
                # Calculate how much to buy (90% of available cash)
                buy_amount = cash * 0.9
                position = buy_amount / row['close']
                cash -= buy_amount
                
                # Record trade
                self.trades.append({
                    'date': index,
                    'type': 'BUY',
                    'price': row['close'],
                    'amount': position,
                    'value': buy_amount
                })
                
            # Check for sell signal
            elif row['signal'] == -1 and position > 0:
                # Sell all position
                sell_value = position * row['close']
                cash += sell_value
                
                # Record trade
                self.trades.append({
                    'date': index,
                    'type': 'SELL',
                    'price': row['close'],
                    'amount': position,
                    'value': sell_value
                })
                
                position = 0
            
            # Update portfolio value
            df.at[index, 'position'] = position
            df.at[index, 'cash'] = cash
            df.at[index, 'holdings'] = position * row['close']
            df.at[index, 'total'] = cash + (position * row['close'])
        
        # Calculate performance metrics
        self.final_balance = df['total'].iloc[-1]
        self.return_pct = ((self.final_balance / self.initial_balance) - 1) * 100
        self.trade_count = len(self.trades)
        
        # Calculate drawdown
        df['previous_peak'] = df['total'].cummax()
        df['drawdown'] = (df['total'] - df['previous_peak']) / df['previous_peak'] * 100
        self.max_drawdown = df['drawdown'].min()
        
        return df

    def plot_results(self, data):
        """
        Plot the backtest results.
        
        Parameters:
        - data: DataFrame with backtest results
        """
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot price and indicators
        axs[0].plot(data.index, data['close'], label='Close Price', color='black')
        axs[0].plot(data.index, data['sma'], label='SMA', color='blue', alpha=0.6)
        axs[0].plot(data.index, data['upper_band'], label='Upper Band', color='red', linestyle='--')
        axs[0].plot(data.index, data['lower_band'], label='Lower Band', color='green', linestyle='--')
        axs[0].plot(data.index, data['vwap'], label='VWAP', color='purple', alpha=0.6)
        
        # Plot buy/sell signals
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        
        axs[0].scatter(buy_signals.index, buy_signals['close'], color='green', label='Buy Signal', marker='^', s=100)
        axs[0].scatter(sell_signals.index, sell_signals['close'], color='red', label='Sell Signal', marker='v', s=100)
        
        axs[0].set_title(f'{self.symbol} Price and Indicators')
        axs[0].set_ylabel('Price')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot RSI
        axs[1].plot(data.index, data['rsi'], label='RSI', color='orange')
        axs[1].axhline(y=self.rsi_oversold, color='green', linestyle='--', label=f'Oversold ({self.rsi_oversold})')
        axs[1].axhline(y=self.rsi_overbought, color='red', linestyle='--', label=f'Overbought ({self.rsi_overbought})')
        axs[1].set_title('RSI Indicator')
        axs[1].set_ylabel('RSI')
        axs[1].set_ylim(0, 100)
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot portfolio value
        axs[2].plot(data.index, data['total'], label='Portfolio Value', color='green')
        axs[2].set_title('Portfolio Performance')
        axs[2].set_ylabel('Value ($)')
        axs[2].set_xlabel('Date')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    def print_performance(self):
        """
        Print the performance metrics of the backtest.
        """
        print("=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.final_balance:.2f}")
        print(f"Return: {self.return_pct:.2f}%")
        print(f"Number of Trades: {self.trade_count}")
        print(f"Maximum Drawdown: {self.max_drawdown:.2f}%")
        print("=" * 50)

    def run_strategy(self, start_date, end_date):
        """
        Run the complete strategy from data fetching to backtest.
        
        Parameters:
        - start_date: Start date for the backtest
        - end_date: End date for the backtest
        
        Returns:
        - DataFrame with backtest results
        """
        # Fetch data
        data = self.fetch_data(start_date, end_date)
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Run backtest
        results = self.backtest(data)
        
        # Print performance
        self.print_performance()
        
        # Plot results
        self.plot_results(results)
        
        return results

def main():
    # Create .env file template if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('ALPACA_API_KEY=your_api_key_here\n')
            f.write('ALPACA_API_SECRET=your_api_secret_here\n')
        print("Created .env file. Please add your Alpaca API credentials.")
        return
    
    # Check if API keys are provided
    if API_KEY == 'your_api_key_here' or API_SECRET == 'your_api_secret_here':
        print("Please update your Alpaca API credentials in the .env file.")
        return
    
    # Initialize strategy
    strategy = BollingerRSIVWAPStrategy(symbol='XRPUSD', timeframe='1H')
    
    # Set date range for backtest (last 3 months)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=90)
    
    # Run strategy
    strategy.run_strategy(start_date, end_date)

if __name__ == "__main__":
    main()