import util

open_fee = 0
param_file = "C:/Users/mario/python/trading/walkForwardParams_Total.csv"


def get_training_data():
    train_yrs = 3
    test_yrs = 1
    to_maximise = "Total"
    # parameters to test [start from, end at, increment]
    params_range = [[1, 4, 0.5]]
    return [train_yrs, test_yrs, params_range, to_maximise]


def get_static_data():
    tickers = "LR.PA ACA.PA ATO.PA SGO.PA DG.PA GLE.PA VIV.PA SU.PA MC.PA WLN.PA AI.PA ORA.PA OR.PA BNP.PA SAN.PA " \
              "CAP.PA RI.PA FP.PA EN.PA AC.PA VIE.PA ML.PA ENGI.PA KER.PA HO.PA CA.PA BN.PA AIR.PA UG.PA SW.PA "
    start = "2015-01-01"
    end = "2020-06-25"
    initial_capital = 40000
    trade_size = 1000
    params = [1]
    indicators = {
        "sma": [200],
        "ATR": [14]
    }
    return [tickers, indicators, start, end, initial_capital, trade_size, params]


def open_trades(port_df, trade_df, hist_data, date, prev_date, trade_size, params):
    open_trades = trade_df[trade_df.Close_Date == 0]
    for sym in hist_data.columns.levels[0]:
        if port_df.loc[date, "Cash"] <= trade_size:
            return [port_df, trade_df]
        if hist_data[(sym, "Long_Buy")][prev_date] and open_trades[open_trades.Instrument == sym].empty:
            mkt_open = hist_data[(sym, "Open")][date]
            stop_loss = mkt_open - params[0] * hist_data[(sym, "ATR_14d")][date]
            profit_target = False
            fixed_life = False
            [port_df, trade_df] = util.enter_trade(trade_df, port_df, date, sym, mkt_open, stop_loss, profit_target,
                                                   fixed_life, trade_size)

    return [port_df, trade_df]


def update_targets(row, hist_data, prev_date, params):
    sym = row.Instrument
    # Adjust stop loss based on yesterday's sma
    # row.stop_loss = max(hist_data[(sym, "High")][prev_date] - params[0] * hist_data[(sym, "ATR_14d")][prev_date], row.stop_loss)
    return row


def add_signals(df):
    for sym in df.columns.levels[0]:
        if sym == "Date":
            continue
        # Add long condition here customised based on chosen indicators
        df[(sym, "Long_Buy")] = (df.rolling(7).min()[(sym, "Close")] == df[(sym, "Close")]) & (
                    df[(sym, "Close")] > df[(sym, "sma_200d")])
        df[(sym, "Long_Sell")] = (df.rolling(7).max()[(sym, "Close")] == df[(sym, "Close")])
    return df
