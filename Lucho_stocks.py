# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:03:35 2024

@author: LUIBAQ
"""



import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from colorama import Fore, Style, init
import re
from scipy.signal import argrelextrema
import mplfinance as mpf
import matplotlib as mpl
from matplotlib import font_manager

# Tickers Tracking
def Track_Tickers():
    """
Provides a dictionary of predefined stock tickers and their corresponding company names.

Returns:
    dict: A dictionary where keys are stock tickers and values are company names.
"""
    tt = {"6GAA.f": "Grupo Aval",
          'EC': 'Ecopetrol',
          'BXK.DU': 'Bancolombia',
          'MLB1.F': 'mercadolibre'}
    return tt

# Initialize colorama
init()

def get_valid_ticker():
    """
    Prompts the user for a stock ticker and validates its availability using the `yfinance` library.

    Returns:
        tuple: The stock ticker and its name.
    """
    while True:
        ticker = input(f"{Fore.CYAN}Enter the stock ticker (e.g., 6GAA.f): {Style.RESET_ALL}") or "6GAA.f"
        stock = yf.Ticker(ticker)
        try:
            stock_info = stock.info
            if 'maxAge' in stock_info and stock_info['maxAge'] <= 86400:  # Ensure data is recent (24 hours or less)
                historical_data = stock.history(period='max')
                earliest_date = historical_data.index.min()

                print(f"\n{Fore.GREEN}Analysis of: {stock_info.get('longName', ticker)} ({ticker}){Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}Symbol: {stock_info['symbol']}")
                print(f"Quote Type: {stock_info['quoteType']}")
                print(f"Max Age: {stock_info['maxAge']} seconds")
                print(f"Earliest available data for {ticker} is from: {earliest_date}{Style.RESET_ALL}\n")

                return ticker, stock_info.get('longName', ticker)
            else:
                print(f"{Fore.RED}Data for '{ticker}' is not recent. Please try another ticker.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Could not retrieve data for '{ticker}'. Error: {e}. Restarting...{Style.RESET_ALL}")



def parse_period(period_str):
    """
    Converts a user-defined period string into a start date for historical stock data retrieval.

    Args:
        period_str (str): Period string (e.g., '1w', '1m', '1y', 'ytd', 'all').

    Returns:
        datetime | None: Start date for the data or None for 'all'.

    Raises:
        ValueError: If the input format is invalid.
    """
    match = re.match(r"(\d+)([wmy])", period_str)
    if match:
        num, unit = int(match.group(1)), match.group(2).lower()
        if unit == "w":
            return datetime.today() - timedelta(weeks=num)
        elif unit == "m":
            return datetime.today() - timedelta(days=num * 30)  # Approximate month as 30 days
        elif unit == "y":
            return datetime.today() - timedelta(days=num * 365)
    elif period_str.lower() == "ytd":
        return datetime(datetime.today().year, 1, 1)
    elif period_str.lower() == "all":
        return None
    else:
        raise ValueError("Invalid period format. Please use formats like '3m', '1y', 'ytd', or 'all'.")

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI) for the given stock data."""
    """Calculate Relative Strength Index (RSI) for the given stock data.
    
    El RSI se calcula a partir de las ganancias y pérdidas promedio durante un período determinado (generalmente 14 días). La fórmula básica es:
        RSI = 100 -(100/(1+RS)) Donde RS es la relación de la ganancia promedio a la pérdida promedio.
    
    3. Gráfico de rendimiento relativo (Relative Strength Index - RSI)
   
    ¿Qué muestra?: El RSI mide la velocidad y el cambio de los movimientos de precio para evaluar si una acción está sobrecomprada o sobrevendida.
    
    ¿Por qué es útil?: Un RSI por encima de 70 indica que la acción puede estar sobrecomprada (potencialmente sobrevalorada), mientras que un RSI por debajo de 30 puede sugerir que la acción está sobrevendida (potencialmente infravalorada).

    Cómo utilizarlo: Si planeas mantener la acción a largo plazo, un RSI bajo (por debajo de 30) podría indicar una buena oportunidad de compra
    
    
    """
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Utility function to format x-axis ticks
def format_xaxis_ticks(ax, rotation=45):
    """
   Formats x-axis ticks for better date readability.

   Args:
       ax (Axes): Matplotlib axes to format.
       rotation (int): Rotation angle for tick labels (default is 45).
   """
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))
    ax.tick_params(axis='x', rotation=rotation)

# Function to identify support and resistance levels
def find_support_resistance(data, order=5):
    """
    Identifies support (minima) and resistance (maxima) levels.
    :param data: pandas Series of price data (e.g., 'Close')
    :param order: Number of points to consider around local extrema
    :return: Tuple (supports, resistances)
    """
    local_minima = argrelextrema(data.values, np.less_equal, order=order)[0]
    local_maxima = argrelextrema(data.values, np.greater_equal, order=order)[0]

    supports = data.iloc[local_minima]
    resistances = data.iloc[local_maxima]

    return supports, resistances

# Function to identify support/resistance zones from consecutive points
def find_consecutive_zones(levels, threshold=4):
    """
    Identifies zones of consecutive support or resistance points.
    :param levels: pandas Series of support or resistance levels
    :param threshold: Number of consecutive levels to form a zone
    :return: List of tuples (start_index, end_index, lower_bound, upper_bound)
    """
    zones = []
    consecutive = []

    for i in range(1, len(levels)):
        if levels.index[i] - levels.index[i - 1] == 1:  # Check if consecutive
            consecutive.append(levels.index[i - 1])
            if i == len(levels) - 1:  # Include the last point
                consecutive.append(levels.index[i])
        else:
            if len(consecutive) >= threshold:
                low = levels.loc[consecutive].min()
                high = levels.loc[consecutive].max()
                zones.append((consecutive[0], consecutive[-1], low * 0.95, high * 1.05))
            consecutive = []

    return zones


# Función principal para graficar las velas
def plot_candlestick_chart(data, stock_name, ticker, period_str):
    try:
        # Check for data availability
        if data.empty:
            print("No data available for the specified period.")
            return

        # Define custom style with original colors
        custom_style = mpf.make_mpf_style(base_mpf_style="yahoo", 
                                          rc={"axes.grid": True,
                                              'font.size': 9, 
                                              'font.family': 'DejaVu Sans'})

        # Create the figure and axis
        figsize = (17, 9) 
        fig, axlist = mpf.plot(
            data,
            type="candle",
            style=custom_style,
            volume=True,
            mav=(20, 50),
            returnfig=True,
            figsize=figsize
        )

        # Access the moving average lines manually
        ax_price = axlist[0]  # Main price plot
        # Add legends for moving averages
        moving_avg_lines = ax_price.get_lines()[-2:]  # Get the last two lines (20 and 50 moving averages)
        
        font_properties = font_manager.FontProperties(family='DejaVu Sans', size=9)
       
        ax_price.legend(
            moving_avg_lines,
            ["Short-term (20d M_Avg)", "Long-term (50d M_Avg)"],
            loc="best",
            prop=font_properties, 
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="black"
        )

        # Add the title to the bottom plot (volume)
        ax_price.set_title(f"{stock_name} ({ticker}) - Candle and volumes for {period_str}", fontsize=12, loc="center")

        # Show the plot
        plt.show(block=False)

    except ValueError as e:
        print(f"Error: {e}")


def plot_price_pivots(data, stock_name, ticker, period_str):
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot closing price
    ax.plot(data['Close'], label="Closing Price", color='blue')
    ax.set_title(f"{stock_name} ({ticker}) - Closing Price ({period_str})")
    ax.set_ylabel("Closing Price (Currency)")

    # Find support and resistance levels
    supports, resistances = find_support_resistance(data['Close'])

    # Plot support levels
    ax.scatter(supports.index, supports.values, color='green', label='Support Levels', marker='o', alpha=0.6)
    # Plot dashed lines at each support level
    for y in supports.values:
        ax.axhline(y=y, color='green', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot resistance levels
    ax.scatter(resistances.index, resistances.values, color='red', label='Resistance Levels', marker='o', alpha=0.6)
    # Plot dashed lines at each resistance level
    for y in resistances.values:
        ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    # Find and plot support zones
    support_zones = find_consecutive_zones(supports)
    for zone in support_zones:
        ax.fill_between(data.index[zone[0]:zone[1]+1], zone[2], zone[3], color='green', alpha=0.3, label="Support Zone")

    # Find and plot resistance zones
    resistance_zones = find_consecutive_zones(resistances)
    for zone in resistance_zones:
        ax.fill_between(data.index[zone[0]:zone[1]+1], zone[2], zone[3], color='red', alpha=0.3, label="Resistance Zone")

    # Apply x-axis formatting
    format_xaxis_ticks(ax)

    # Move y-axis to the right
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.show(block=False)


def plot_price_accum_dist_rsi(data, stock_name, ticker, period_str, show_vertical_lines):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))  # Added a third axis

    # Plot closing price
    ax1.plot(data['Close'], label="Closing Price", color='blue')
    ax1.set_title(f"{stock_name} ({ticker}) - Price and Accumulation/Distribution Zones ({period_str})")
    ax1.set_ylabel("Closing Price (Currency)")

    # Move y-axis to the right for ax1
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    # Apply x-axis formatting
    format_xaxis_ticks(ax1)

    # On-Balance Volume (OBV)
    data['OBV'] = (np.where(data['Close'] > data['Close'].shift(1), data['Volume'],
                            np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0))).cumsum()
    ax2.plot(data['OBV'], label="On-Balance Volume (OBV)", color='purple')
    ax2.set_ylabel("OBV (Volume)")

    # Move y-axis to the right for ax2
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax2.axhline(0, color='black', linestyle='--')
    ax2.fill_between(data.index, data['OBV'], where=(data['OBV'] > 0), color='green', alpha=0.3)
    ax2.fill_between(data.index, data['OBV'], where=(data['OBV'] < 0), color='red', alpha=0.3)

    # Show vertical lines if requested
    if show_vertical_lines == 'y':
        for period in data[data['OBV'] > 0].index:
            ax2.axvline(x=period, color='green', linewidth=0.5)
        for period in data[data['OBV'] < 0].index:
            ax2.axvline(x=period, color='red', linewidth=0.5)

    # Plot RSI
    data['RSI'] = calculate_rsi(data)
    ax3.plot(data['RSI'], label="RSI (14)", color='orange')
    ax3.axhline(70, color='red', linestyle='--', label="Overbought (70)")  # Overbought line
    ax3.axhline(30, color='green', linestyle='--', label="Oversold (30)")  # Oversold line
    ax3.set_ylabel("RSI")
    ax3.set_title(f"RSI for {stock_name} ({ticker})")
    ax3.legend()
    
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()

    # Synchronize x-axis format and thickness with the price plot
    ax2.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))
    ax2.tick_params(axis='x', rotation=45)

    ax3.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show(block=False)
    
    

def plot_price_volume(data, stock_name, ticker, period_choice):
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot closing price
    ax1.plot(data.index, data['Close'], label="Closing Price", color='blue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Closing Price (Currency)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Apply x-axis formatting
    format_xaxis_ticks(ax1)

    # Create a second axis for volume
    ax2 = ax1.twinx()
    ax2.bar(data.index, data['Volume'], label="Trading Volume", color='gray', alpha=0.3)
    ax2.set_ylabel("Volume", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Highlight days with volume increase when price rises
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1] and data['Volume'].iloc[i] > data['Volume'].iloc[i-1]:
            ax2.bar(data.index[i], data['Volume'].iloc[i], color='green', alpha=0.5)
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1] and data['Volume'].iloc[i] > data['Volume'].iloc[i-1]:
            ax2.bar(data.index[i], data['Volume'].iloc[i], color='red', alpha=0.5)
        else:
            ax2.bar(data.index[i], data['Volume'].iloc[i], color='gray', alpha=0.3)

    # Add labels to indicate meaning of colors
    ax2.text(0.95, 0.95, "Green: Price up, Volume up", color='green', ha='right', va='top', transform=ax2.transAxes)
    ax2.text(0.95, 0.90, "Red: Price down, Volume up", color='red', ha='right', va='top', transform=ax2.transAxes)
    ax2.text(0.95, 0.85, "Gray: No price change or volume change", color='gray', ha='right', va='top', transform=ax2.transAxes)

    ax1.set_title(f"{stock_name} ({ticker}) - Price and Trading Volume - {period_choice}")
    plt.tight_layout()
    plt.show(block=False)
    
    
def main():
    print("Welcome to Stock Data Analysis!")

    while True:  # Loop to allow restarting with a different period
        ticker, stock_name = get_valid_ticker()

        period_str = input(
            f"{Fore.CYAN}Enter the period for analysis (e.g., '1w', '1m', '1y', 'ytd', 'all'): {Style.RESET_ALL}"
        ) or "ytd"
        try:
            start_date = parse_period(period_str)
            if start_date:
                data = yf.Ticker(ticker).history(start=start_date)
            else:
                data = yf.Ticker(ticker).history(period="max")

            plot_options = [
                ("Plot Closing Price with Pivot points", plot_price_pivots),
                ("Plot Price and Accumulation/Distribution with RSI", plot_price_accum_dist_rsi),
                ("Plot Price and Volume", plot_price_volume),
                ("Plot Candlestick", plot_candlestick_chart),
            ]

            while True:  # Inner loop to allow multiple plots for the same period
                print("\nSelect a plot to generate:")
                for i, (description, _) in enumerate(plot_options, 1):
                    print(f"{i}. {description}")
                print("q. Quit")

                user_choice = input(f"{Fore.CYAN}Enter the number of your choice: {Style.RESET_ALL}").strip().lower()

                if user_choice == 'q':
                    break

                try:
                    plot_choice = int(user_choice) - 1
                    if 0 <= plot_choice < len(plot_options):
                        plot_func = plot_options[plot_choice][1]

                        # Handle special plot options
                        if plot_choice == 1:  # plot_price_accum_dist_rsi requires show_vertical_lines
                            show_vertical_lines = input(
                                f"{Fore.YELLOW}Do you want Accum / Distri Vertical lines? (y/n): {Style.RESET_ALL}"
                            ).strip().lower()
                            plot_func(data, stock_name, ticker, period_str, show_vertical_lines)
                        else:
                            plot_func(data, stock_name, ticker, period_str)
                    else:
                        print(f"{Fore.RED}Invalid choice! Please select a valid option.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid input! Please enter a valid number.{Style.RESET_ALL}")

                continue_choice = input(
                    f"{Fore.YELLOW}Do you want to plot another? (y/n): {Style.RESET_ALL}"
                ).strip().lower()
                if continue_choice != 'y':
                    break

        except ValueError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

        restart_choice = input(
            f"{Fore.YELLOW}Do you want to start over with a different period? (y/n): {Style.RESET_ALL}"
        ).strip().lower()
        if restart_choice != 'y':
            break

    print(f"{Fore.GREEN}Thank you for using Stock Data Analysis. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
