#
# strategy.py
#
# Purpose: Contains functions to determine strategy for signals, opening trades, updating profit targets/stop loss
#
# Revision History
# When      Who         What
# 20200711  Marios C    Created
#

import util
import os

# Define fee applied upon open to cover trading expenses / cover bid ask spread etc (as % of trade size)
open_fee = 0
# Path to input and output csv files
# Note dates must be in format yyyy-mm-dd (excel can sometimes change this if opened manually so be careful)
dir_root = os.getcwd()
results_file = dir_root + "\\walkForwardResults.csv"
param_file = dir_root + "\\walkForwardParams.csv"
trade_file = dir_root + "\\trades.csv"
port_file = dir_root + "\\portfolio.csv"


# Get training parameters for optimise.py
def get_training_data():
    train_yrs = 1  # No. yrs used for training
    test_yrs = 1  # No. yrs between re-optimisation of parameters
    to_maximise = "Total"  # Output to optimise, "Total" portfolio value or "Sharpe" sharpe ratio

    # parameters to test, array of arrays
    # To test [p0, p1] use [[start p0 from, end p0 at, increment], [start p1 from, end p1 at, increment]]
    params_range = [[3, 4, 0.5], [0.05, 0.15, 0.01]]
    return [train_yrs, test_yrs, params_range, to_maximise]


# Define data required when running single simulation
def get_static_data():
    tickers = "LR.PA ACA.PA ATO.PA SGO.PA DG.PA GLE.PA VIV.PA SU.PA MC.PA WLN.PA AI.PA ORA.PA OR.PA BNP.PA SAN.PA " \
              "CAP.PA RI.PA FP.PA EN.PA AC.PA VIE.PA ML.PA ENGI.PA KER.PA HO.PA CA.PA BN.PA AIR.PA UG.PA SW.PA "
    start = "2015-01-01"
    end = "2020-06-25"
    initial_capital = 30000
    trade_size = 1000
    sample_params = [3.5, 0.1]
    indicators = {
        "sma": [200],
        "ATR": [14, 22]
    }
    return [tickers, indicators, start, end, initial_capital, trade_size, sample_params]


# Function to open trades conditional on strategy implemented to define close conditions
# Open long position if indicators suggest to buy, no open long position already exists on that ticker and
# there is sufficient cash to enter trade
def open_trades(port_df, trade_df, hist_data, date, prev_date, trade_size, params):
    open_trades = trade_df[trade_df.Close_Date == 0]
    for sym in hist_data.columns.levels[0]:
        if port_df.loc[date, "Cash"] <= trade_size * (1 + (open_fee / 100)):
            # If we have insufficient cash to enter a new trade, can break loop and not check for any new trades
            return [port_df, trade_df]
        if hist_data[(sym, "Long_Buy")][prev_date] and open_trades[open_trades.Instrument == sym].empty:
            mkt_open = hist_data[(sym, "Open")][date]
            stop_loss = mkt_open - params[0] * hist_data[(sym, "ATR_14d")][date]
            profit_target = mkt_open * (1 + params[1])
            fixed_life = False
            [port_df, trade_df] = util.enter_trade(trade_df, port_df, date, sym, mkt_open, stop_loss, profit_target,
                                                   fixed_life, trade_size)

    return [port_df, trade_df]


# Update stop loss, trade life or profit target in life of trade based on trade history
def update_targets(row, hist_data, prev_date, params):
    row.stop_loss = max(row.stop_loss, hist_data[(row.Instrument, "Close")][prev_date]
                        - params[0] * hist_data[(row.Instrument, "ATR_14d")][prev_date])
    return row


# Determine required trade signals which are independent of trade history which define strategy
def add_signals(df):
    for sym in df.columns.levels[0]:
        if sym == "Date":
            continue
        # Add long condition here customised based on chosen indicators
        df[(sym, "Long_Buy")] = (df.rolling(7).min()[(sym, "Close")] == df[(sym, "Close")]) & (
                df[(sym, "Close")] > df[(sym, "sma_200d")])
        df[(sym, "Long_Sell")] = (df.rolling(7).max()[(sym, "Close")] == df[(sym, "Close")])
    return df
