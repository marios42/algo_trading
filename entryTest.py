# entryTest.py
#
# Purpose: To test how well good the entry condition
# Entry will be tested against the following
# TEST                              SUCCESS CRITERIA
# Fixed Stop Loss and Target        > 50% trades successful
# Fixed Life Exit                   > 50% trades profitable
# compare to random entry           Determined much better than random
#
# Revision History
# When      Who         What
# 20200711  Marios C    Created
#

import matplotlib.pyplot as plt
import numpy as np
import util
import strategy as strat


def override_life(row):
    row.stop_loss = False
    row.profit_target = False
    row.trade_life = 4
    return row


def override_target(row):
    row.stop_loss = row.Open_Price * 0.9
    row.profit_target = row.Open_Price * 1.1
    row.trade_life = False
    return row


def determine_test_trades(hist_data, port_df, trade_df, date_index, trade_size, params, test_type, rand_arr):
    # Get open trades from previous bar
    date = hist_data.index[date_index]
    prev_date = hist_data.index[date_index - 1]
    port_df.loc[date, :] = port_df.loc[prev_date, :]

    # Determine what trades need to open at start of bar
    if test_type == "Random_Life" or test_type == "Random_Target":
        [port_df, trade_df] = util.open_random(port_df, trade_df, hist_data, date_index, trade_size, rand_arr)
    else:
        [port_df, trade_df] = strat.open_trades(port_df, trade_df, hist_data, date, prev_date, trade_size, params)

    # Do some bookkeeping
    trade_df[trade_df.Close_Date == 0] = trade_df[trade_df.Close_Date == 0].apply(util.update_trade,
                                                                                  args=[hist_data, date, trade_size],
                                                                                  axis=1)

    if test_type == "Fixed_Life" or test_type == "Random_Life":
        trade_df[trade_df.Open_Date == date] = trade_df[trade_df.Open_Date == date].apply(override_life, axis=1)
    elif test_type == "Fixed_Target" or test_type == "Random_Target":
        trade_df[trade_df.Open_Date == date] = trade_df[trade_df.Open_Date == date].apply(override_target, axis=1)

    trade_df[trade_df.Close_Date == 0] = trade_df[trade_df.Close_Date == 0].apply(util.close_trades,
                                                                                  args=[hist_data, date, trade_size],
                                                                                  axis=1)

    port_df.loc[date, :] = util.eod(port_df, trade_df, date)

    return [trade_df, port_df]


def run_test(hist_data, trade_df, port_df, trade_size, params, test_type, trade_cnt):
    if test_type == "Random_Life" or test_type == "Random_Target":
        symbols = hist_data.columns.levels[0]
        rand_arr = np.random.randint(len(hist_data.index), size=(len(symbols), trade_cnt // len(symbols)))
    else:
        rand_arr = []
    for date_index in range(1, len(hist_data.index)):
        [trade_df, port_df] = determine_test_trades(hist_data, port_df, trade_df, date_index, trade_size, params,
                                                    test_type, rand_arr)

    return [trade_df, port_df]


if __name__ == '__main__':
    [tickers, indicators, start, end, initial_capital, trade_size, params] = strat.get_static_data()

    all_data = util.get_data(tickers, indicators, start, end)
    all_data = strat.add_signals(all_data)

    [trade_df_target, port_df_target] = util.initialise_dfs(all_data, initial_capital)
    [trade_df_life, port_df_life] = util.initialise_dfs(all_data, initial_capital)
    [trade_df_rand_life, port_df_rand_life] = util.initialise_dfs(all_data, initial_capital)
    [trade_df_rand_target, port_df_rand_target] = util.initialise_dfs(all_data, initial_capital)

    # Run Fixed Stop Loss / Target Test
    print("Run Fixed Stop Loss / Target Test")
    [trade_df_target, port_df_target] = run_test(all_data, trade_df_target, port_df_target, trade_size, params,
                                                 "Fixed_Target", 0)

    print("Run Fixed Time Test")
    [trade_df_life, port_df_life] = run_test(all_data, trade_df_life, port_df_life, trade_size, params, "Fixed_Life", 0)

    print("Run Random Entry Life Test")
    [trade_df_rand_life, port_df_rand_life] = run_test(all_data, trade_df_rand_life, port_df_rand_life, trade_size,
                                                       params, "Random_Life", len(trade_df_life.index))
    print("Run Random Entry Target Test")
    [trade_df_rand_target, port_df_rand_target] = run_test(all_data, trade_df_rand_target, port_df_rand_target,
                                                           trade_size, params, "Random_Target",
                                                           len(trade_df_target.index))

    target_success_prop = trade_df_target[trade_df_target.Close_Date != 0]["Profit"].gt(0).sum() / \
                          len(trade_df_target[trade_df_target.Close_Date != 0].index)
    rand_target_success_prop = trade_df_rand_target[trade_df_rand_target.Close_Date != 0]["Profit"].gt(0).sum() / \
                               len(trade_df_rand_target[trade_df_rand_target.Close_Date != 0].index)
    life_success_prop = trade_df_life[trade_df_life.Close_Date != 0]["Profit"].gt(0).sum() / \
                        len(trade_df_life[trade_df_life.Close_Date != 0].index)
    rand_life_success_prop = trade_df_rand_life[trade_df_rand_life.Close_Date != 0]["Profit"].gt(0).sum() / \
                             len(trade_df_rand_life[trade_df_rand_life.Close_Date != 0].index)

    print("Proportion of successful trades with fixed target: " + str(target_success_prop))
    print("Proportion of successful random trades with fixed target: " + str(rand_target_success_prop))
    print("Proportion of successful trades with fixed life: " + str(life_success_prop))
    print("Proportion of successful random trades with fixed life: " + str(rand_life_success_prop))

    print("Fixed Life Profit rate: " + str(life_success_prop))
    print("Fixed Life Rate Better than random? " + str(life_success_prop > rand_life_success_prop))
    print("Fixed Target Profit Rate: " + str(target_success_prop))
    print("Fixed Life Rate Better than random? " + str(target_success_prop > rand_target_success_prop))

    plt.plot((port_df_target.Total - initial_capital) / initial_capital, color="blue")
    plt.plot((port_df_life.Total - initial_capital) / initial_capital, color="red")
    plt.plot((port_df_rand_life.Total - initial_capital) / initial_capital, color="black")
    plt.plot((port_df_rand_target.Total - initial_capital) / initial_capital, color="green")
    plt.show()
