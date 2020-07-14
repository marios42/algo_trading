#
# util.py
#
# Purpose: Contains utility functions for this repo which are used across multiple apps
#
# Revision History
# When      Who         What
# 20200711  Marios C    Created
#

import pandas as pd
import numpy as np
import yfinance as yf
import strategy as strat


def get_path_analytics(portfolio_df, port_name):
    all_tests = ["Annual_Return", "Annual_Sharpe", "Max_Drawdown", "Return/Max_Drawdown", "Annual_Std"]
    output_df = pd.DataFrame(columns=[port_name], index=all_tests)
    ann_return = portfolio_df["Return"].mean() * 252
    ann_std = (252**0.5) * portfolio_df["Return"].std()
    max_drawdown = (portfolio_df["Total"] / portfolio_df["High"]).min() - 1.0
    result_obj = {
        "Annual_Return": ann_return,
        "Annual_Std": ann_std,
        "Max_Drawdown": abs(max_drawdown),
        "Annual_Sharpe": ann_return / ann_std,
        "Return/Max_Drawdown": ann_return / abs(max_drawdown)
    }

    for test in all_tests:
        if test not in result_obj:
            print("WARNING - " + test + " not defined in test result object")
            continue
        output_df.loc[test, port_name] = result_obj[test]
    return output_df


def test_params(params, all_data, trade_size, initial_capital, queue):
    [trade_df, port_df] = initialise_dfs(all_data, initial_capital)

    [trade_df, port_df, total, sharpe_ratio] = run_simulation(all_data, trade_df, port_df, trade_size, initial_capital,
                                                              params)
    queue.put([params, total, sharpe_ratio])


def run_baseline(hist_data, trade_df, port_df, trade_size):
    for sym in hist_data.columns.levels[0]:
        date = hist_data.index[0]
        mkt_open = hist_data.loc[date, (sym, "Open")]
        stop_loss = False
        profit_target = False
        fixed_life = False
        [port_df, trade_df] = enter_trade(trade_df, port_df, date, sym, mkt_open, stop_loss, profit_target,
                                          fixed_life, trade_size)

    for date_index in range(1, len(hist_data.index)):
        [trade_df, port_df] = base_trades(hist_data, port_df, trade_df, date_index, trade_size)
    port_df["Return"] = np.log(1 + port_df["Total"].pct_change(1))
    sharpe_ratio = (252 ** 0.5) * port_df["Return"].mean() / port_df["Return"].std()
    total = port_df["Total"][port_df.index[-1]]
    return [trade_df, port_df, total, sharpe_ratio]


def run_simulation(hist_data, trade_df, port_df, trade_size, initial_capital, params):
    for date_index in range(1, len(hist_data.index)):
        [trade_df, port_df] = determine_trades(hist_data, port_df, trade_df, date_index, trade_size, initial_capital,
                                               params)
    port_df["Return"] = port_df["Total"].pct_change(1)
    sharpe_ratio = port_df["Return"].mean() / port_df["Return"].std() * (252 ** 0.5)
    total = port_df["Total"][port_df.index[-1]]
    return [trade_df, port_df, total, sharpe_ratio]


def get_data(tickers, indicators, start, end, drop_index=True):
    print("Extracting Raw Data")
    or_data = yf.download(tickers, end=end, group_by="ticker")

    print("Rescaling Raw Data")
    adj_hist_data = adjust_data(or_data)

    print("Adding indicators to DataFrame")
    data = add_indicators(adj_hist_data, indicators)
    data = data[data.index >= start]
    all_data = data.reset_index(drop=drop_index).copy()
    return all_data


# adjust historical data, scaling so all fields are adjusted
def adjust_data(df):
    adj_df = df.copy()
    adj_df = adj_df.apply(adjust_prices, args=[df])
    drop_cols = []
    for symbol in df.columns.levels[0]:
        drop_cols.append((symbol, "Adj Close"))
    adj_df = adj_df.drop(columns=drop_cols)
    return adj_df


def adjust_prices(col, raw_data):
    sym_name = col.name[0]
    col = col * raw_data[(sym_name, "Adj Close")] / raw_data[(sym_name, "Close")]
    return col


def add_indicators(df, indicators):
    indicator_df = df.copy()
    max_rows_used = 0
    if "ema" in indicators:
        for n in indicators["ema"]:
            add_ewma(indicator_df, n)
            max_rows_used = max(max_rows_used, n)
    if "sma" in indicators:
        for n in indicators["sma"]:
            add_sma(indicator_df, n)
            max_rows_used = max(max_rows_used, n)
    if "RSI" in indicators:
        for n in indicators["RSI"]:
            add_rsi(indicator_df, n)
            max_rows_used = max(max_rows_used, n)
    if "max_close" in indicators:
        for n in indicators["max_close"]:
            add_max_close(indicator_df, n)
            max_rows_used = max(max_rows_used, n)
    if "min_close" in indicators:
        for n in indicators["min_close"]:
            add_min_close(indicator_df, n)
            max_rows_used = max(max_rows_used, n)
    if "ATR" in indicators:
        for n in indicators["ATR"]:
            add_atr(indicator_df, n)
            max_rows_used = max(max_rows_used, n)
    if "bol" in indicators:
        for arr in indicators["bol"]:
            add_bollinger(indicator_df, arr[0], arr[1])
            max_rows_used = max(max_rows_used, arr[0])
    if "MACD" in indicators:
        for arr in indicators["MACD"]:
            add_macd(indicator_df, arr[0], arr[1], arr[2])
            max_rows_used = max(max_rows_used, arr[1])

    # Set default Long_Buy and Long_Sell columns to false
    for sym in df.columns.levels[0]:
        df[(sym, "Long_Sell")] = False

    # Remove all rows before our indicators are properly set update
    indicator_df = indicator_df.tail(-max_rows_used)
    return indicator_df


def add_sma(df, n):
    for sym in df.columns.levels[0]:
        df[(sym, "sma_" + str(n) + "d")] = df.rolling(n, min_periods=1).mean()[(sym, "Close")]
    return df


def add_ewma(df, n):
    for sym in df.columns.levels[0]:
        df[(sym, "ema_" + str(n) + "d")] = df.ewm(span=n, adjust=False).mean()[(sym, "Close")]
    return df


def add_macd(df, short_n, long_n, sig_n):
    for sym in df.columns.levels[0]:
        short = df.ewm(span=short_n, adjust=False).mean()[(sym, "Close")]
        long = df.ewm(span=long_n, adjust=False).mean()[(sym, "Close")]
        df[(sym, "MACD")] = short - long
        df[(sym, "MACD_signal")] = df.ewm(span=sig_n, adjust=False).mean()[(sym, "MACD")]
    return df


def add_rsi(df, n):
    for sym in df.columns.levels[0]:
        delta = df[(sym, "Close")].diff()
        delta = delta.fillna(0)
        delta_up, delta_down = delta.copy(), delta.copy()
        delta_up[delta_up < 0] = 0
        delta_down[delta_down > 0] = 0
        r_u = delta_up.rolling(n, min_periods=1).mean()
        r_d = delta_down.rolling(n, min_periods=1).mean()
        df[(sym, "RSI_" + str(n) + "d")] = 100 * (r_u / (r_u - r_d))
    return df


def add_max_close(df, n):
    for sym in df.columns.levels[0]:
        df[(sym, "max_close_" + str(n) + "d")] = df.rolling(n, min_periods=1).max()[(sym, "Close")]
    return df


def add_min_close(df, n):
    for sym in df.columns.levels[0]:
        df[(sym, "min_close_" + str(n) + "d")] = df.rolling(n, min_periods=1).min()[(sym, "Close")]
    return df


def calc_true_range(row):
    sym = row.index.levels[0][0]
    tr = max(row[(sym, "High")] - row[(sym, "Low")], abs(row[(sym, "High")] - row[(sym, "prev_close")]),
             abs(row[(sym, "Low")] - row[(sym, "prev_close")]))
    return tr


def add_atr(df, n):
    for sym in df.columns.levels[0]:
        data = df[[(sym, "Close"), (sym, "High"), (sym, "Low")]].copy()
        high = data[(sym, "High")]
        low = data[(sym, "Low")]
        close = data[(sym, "Close")]
        data[(sym, "tr0")] = abs(high - low)
        data[(sym, "tr1")] = abs(high - close.shift(1))
        data[(sym, "tr2")] = abs(low - close.shift(1))
        tr = data[[(sym, "tr0"), (sym, "tr1"), (sym, "tr2")]].max(axis=1)
        df[(sym, "ATR_" + str(n) + "d")] = tr.rolling(n, min_periods=1).mean()
    return df


def add_bollinger(df, sma_n, sd_n):
    for sym in df.columns.levels[0]:
        df[(sym, "sma_" + str(sma_n) + "d")] = df.rolling(sma_n, min_periods=1).mean()[(sym, "Close")]
        df[(sym, "upper_band_" + str(sma_n) + "d_" + str(sd_n) + "sd")] = df.rolling(sma_n, min_periods=1).mean()[
                                                                              (sym, "Close")] + sd_n * \
                                                                          df.rolling(sma_n, min_periods=1).std()[
                                                                              (sym, "Close")]
        df[(sym, "lower_band_" + str(sma_n) + "d_" + str(sd_n) + "sd")] = df.rolling(sma_n, min_periods=1).mean()[
                                                                              (sym, "Close")] - sd_n * \
                                                                          df.rolling(sma_n, min_periods=1).std()[
                                                                              (sym, "Close")]

    return df


def initialise_dfs(hist_data, initial_capital):
    print("Initiate Trade Book")
    trade_df = pd.DataFrame(
        columns=["Instrument", "Direction", "Mkt_at_Close", "Open_Date", "Open_Price", "Trade_Low", "Trade_High",
                 "Close_Date", "Close_Price", "NAV", "Profit", "stop_loss", "profit_target", "trade_life"])
    print("Initiate Portfolio DataFrame")
    port_df = pd.DataFrame(columns=["Cash", "Equity", "Total", "Low", "High", "Return"], index=hist_data.index)
    port_df.iloc[0, :] = [initial_capital, 0, initial_capital, initial_capital, initial_capital, 0]
    return [trade_df, port_df]


def close_trades(row, hist_data, date, init_pos):
    sym = row.Instrument
    # Always assume negative would happen first before positive close

    if hist_data[(sym, "Long_Sell")][date]:
        # print("SUCCESS - Close trade on " + sym + " as of " + str(date) + " at success criteria " + str(
        #       hist_data[(sym, "Close")][date]))
        row.Close_Date = date
        row.Close_Price = hist_data[(sym, "Close")][date]

    elif row.stop_loss is not False and hist_data[(sym, "Open")][date] <= row.stop_loss:
        # FAIL - Close trade on symbol as of today at open below stop loss
        row.Close_Date = date
        row.Close_Price = hist_data[(sym, "Open")][date]
    elif row.profit_target is not False and hist_data[(sym, "Open")][date] >= row.profit_target:
        # SUCCESS - Close trade on symbol as of today at open above prof target
        row.Close_Date = date
        row.Close_Price = hist_data[(sym, "Open")][date]
    elif row.stop_loss is not False and hist_data[(sym, "Low")][date] <= row.stop_loss:
        # FAIL - Close trade on symbol as of today at stop loss
        row.Close_Date = date
        row.Close_Price = row.stop_loss
    # If we pass profit target, close trade at target
    elif row.profit_target is not False and hist_data[(sym, "High")][date] >= row.profit_target:
        # SUCCESS - Close trade on symbol as of today at prof target
        row.Close_Date = date
        row.Close_Price = row.profit_target
    elif row.trade_life is not False and row.trade_life == 0:
        # TIME OUT - Close trade on symbol as of today at close
        row.Close_Date = date
        row.Close_Price = row.Mkt_at_Close

    if row.Close_Date != 0:
        row.NAV = row.Close_Price * init_pos / row.Open_Price
        row.Profit = row.NAV - init_pos

    return row


def enter_trade(trade_df, port_df, date, sym, mkt_open, stop_loss, profit_target, fixed_life, trade_size):
    trade_df = trade_df.append(pd.DataFrame([[sym, "Long", mkt_open, date, mkt_open, mkt_open, mkt_open, 0, 0,
                                              trade_size, 0, stop_loss, profit_target, fixed_life]],
                                            columns=trade_df.columns), ignore_index=True)
    change_df = pd.DataFrame([[trade_size * (-1 - (strat.open_fee / 100)), trade_size, 0, 0, 0, 0]],
                             columns=port_df.columns, index=[date])
    port_df.loc[date] = port_df.loc[date] + change_df.loc[date]
    return [port_df, trade_df]


def update_trade(row, hist_data, date, init_pos):
    instrument = row.Instrument
    close = hist_data.loc[date, (instrument, "Close")]
    low = hist_data.loc[date, (instrument, "Low")]
    high = hist_data.loc[date, (instrument, "High")]
    row.Mkt_at_Close = close
    row.Trade_Low = min(row.Trade_Low, low)
    row.Trade_High = max(row.Trade_High, high)
    row.NAV = row.Mkt_at_Close * init_pos / row.Open_Price
    row.Profit = row.NAV - init_pos
    if row.trade_life:
        row.trade_life = row.trade_life - 1
    return row


def eod(port_df, trade_df, date):
    equity = trade_df[trade_df.Close_Date == 0].sum().NAV
    # Previous cash + sum of NAV of all positions closed today
    cash = port_df.loc[date, "Cash"] + trade_df[trade_df.Close_Date == date].sum().NAV
    total = cash + equity
    low = min(port_df.loc[date, "Low"], total)
    high = max(port_df.loc[date, "High"], total)
    return [cash, equity, total, low, high, 0]


def open_trades(port_df, trade_df, hist_data, signal_df, sym_arr, date, prev_date, trade_size=1000):
    present_trades = trade_df[trade_df.Close_Date == 0]
    for sym in sym_arr:
        # Determine whether to open any new trades at open of bar
        # If no open trade and yesterday long signal is = True then open long trade at open
        if signal_df[(sym, "Long")][prev_date] and present_trades[present_trades.Instrument == sym].empty:
            # if present_trades[present_trades.Instrument == sym].empty:
            mkt_open = hist_data[(sym, "Open")][date]
            print("Open trade on " + sym + " as of " + str(date) + " at open with val " + str(mkt_open))
            trade_df = trade_df.append(
                pd.DataFrame([[sym, "Long", mkt_open, date, mkt_open, mkt_open, mkt_open, 0, 0, trade_size, 0]],
                             columns=trade_df.columns), ignore_index=True)
            change_df = pd.DataFrame([[trade_size * (-1 - (strat.open_fee / 100)), trade_size, 0, 0, 0, 0]],
                                     columns=port_df.columns, index=[date])
            print(change_df)
            port_df.loc[date] = port_df.loc[date] + change_df.loc[date]

    return [port_df, trade_df]


def open_random(port_df, trade_df, hist_data, date_index, trade_size, rand_arr):
    date = hist_data.index[date_index]

    for it in range(len(hist_data.columns.levels[0])):
        sym = hist_data.columns.levels[0][it]
        if date_index in rand_arr[it] and not np.isnan(hist_data[(sym, "Open")][date]):
            mkt_open = hist_data[(sym, "Open")][date]
            # Open trade on symbol as of today at open
            stop_loss = False
            profit_target = False
            fixed_life = 5
            [port_df, trade_df] = enter_trade(trade_df, port_df, date, sym, mkt_open, stop_loss, profit_target,
                                              fixed_life, trade_size)

    return [port_df, trade_df]


def base_trades(hist_data, port_df, trade_df, date_index, trade_size):
    date = hist_data.index[date_index]
    prev_date = hist_data.index[date_index - 1]
    port_df.loc[date, :] = port_df.loc[prev_date, :]
    trade_df = trade_df.apply(update_trade, args=[hist_data, date, trade_size], axis=1)

    port_df.loc[date, :] = eod(port_df, trade_df, date)
    return [trade_df, port_df]


# noinspection PyUnusedLocal
def determine_trades(hist_data, port_df, trade_df, date_index, trade_size, initial_capital, params):
    # Get open trades from previous bar
    date = hist_data.index[date_index]
    prev_date = hist_data.index[date_index - 1]
    port_df.loc[date, :] = port_df.loc[prev_date, :]
    [port_df, trade_df] = strat.open_trades(port_df, trade_df, hist_data, date, prev_date, trade_size, params)
    # Do some bookkeeping
    trade_df[trade_df.Close_Date == 0] = trade_df[trade_df.Close_Date == 0].apply(strat.update_targets,
                                                                                  args=[hist_data, prev_date, params],
                                                                                  axis=1)

    trade_df[trade_df.Close_Date == 0] = trade_df[trade_df.Close_Date == 0].apply(update_trade,
                                                                                  args=[hist_data, date, trade_size],
                                                                                  axis=1)

    trade_df[trade_df.Close_Date == 0] = trade_df[trade_df.Close_Date == 0].apply(close_trades,
                                                                                  args=[hist_data, date, trade_size],
                                                                                  axis=1)

    port_df.loc[date, :] = eod(port_df, trade_df, date)
    return [trade_df, port_df]


def add_yrs(date, n):
    date_arr = date.split(sep="-")
    date_arr[0] = str(int(date_arr[0]) + n)
    return "-".join(date_arr)
