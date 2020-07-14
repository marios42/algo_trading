#
# walkForward.py
#
# Purpose: Allows a walk forward test where parameters have been pre optimised using optimise.py
# Test compares results of simulation against holding all possible tickers over simulated period
# Success criterion: walk forward sharpe ratio > baseline sharpe ratio
#
#
# Revision History
# When      Who         What
# 20200711  Marios C    Created
#


import util
import strategy as strat
import matplotlib.pyplot as plt
import pandas as pd


# Find indices where to split the hist data
def add_break_points(row, hist_data):
    row.Test_Start_Index = hist_data[hist_data.Date < row.Test_Start_Date].index.tolist()[-1]
    row.Test_End_Index = hist_data[hist_data.Date < row.Test_End_Date].index.tolist()[-1]
    return row


if __name__ == '__main__':
    params_df = pd.read_csv(strat.param_file)

    [tickers, indicators, start, end, initial_capital, trade_size, params] = strat.get_static_data()
    data = util.get_data(tickers, indicators, util.add_yrs(start, -1), end, False)
    all_data = strat.add_signals(data)

    # Determine how to split historical data to for tests
    params_df["Test_Start_Index"] = ""
    params_df["Test_End_Index"] = ""
    params_df = params_df.apply(add_break_points, args=[all_data], axis=1)
    all_data = all_data.drop(columns=["Date"], axis=1, level=0)
    all_data.columns = all_data.columns.remove_unused_levels()
    [trade_df, port_df] = util.initialise_dfs(all_data.loc[params_df.loc[0, "Test_Start_Index"]:], initial_capital)
    [trade_df_1, port_df_1] = util.initialise_dfs(all_data.loc[params_df.loc[0, "Test_Start_Index"]:], initial_capital)
    for index in range(len(params_df.index)):
        test_params = []
        for col in params_df.columns:
            if col.startswith("Param_"):
                test_params.append(params_df.loc[index, col])

        test_data = all_data.loc[params_df.loc[index, "Test_Start_Index"]:params_df.loc[index, "Test_End_Index"]]
        [trade_df, port_df, total, sharpe_ratio] = util.run_simulation(test_data, trade_df, port_df, trade_size,
                                                                       initial_capital, test_params)

    [trade_df_reg, port_df_reg] = util.initialise_dfs(all_data.loc[params_df.loc[0, "Test_Start_Index"]:],
                                                      initial_capital)

    [trade_df_reg, port_df_reg, total_reg, sharpe_ratio_reg] = util.run_baseline(
        all_data.loc[params_df.loc[0, "Test_Start_Index"]:], trade_df_reg, port_df_reg, trade_size)

    plt.plot((port_df.Total - initial_capital) / initial_capital, color="blue")
    plt.plot((port_df_reg.Total - initial_capital) / initial_capital, color="red")

    test_results = util.get_path_analytics(port_df, "Strategy")
    test_results = test_results.join(util.get_path_analytics(port_df_reg, "Baseline"))
    test_results.to_csv(strat.results_file, index=False)
    trade_df.to_csv(strat.trade_file)
    port_df.to_csv(strat.port_file)
    plt.show()
