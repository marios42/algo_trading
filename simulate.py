import matplotlib.pyplot as plt
import util
import strategy as strat

if __name__ == '__main__':
    [tickers, indicators, start, end, initial_capital, trade_size, params] = strat.get_static_data()

    data = util.get_data(tickers, indicators, start, end)
    all_data = strat.add_signals(data)

    [trade_df, port_df] = util.initialise_dfs(all_data, initial_capital)
    [trade_df, port_df, total, sharpe_ratio] = util.run_simulation(all_data, trade_df, port_df, trade_size, initial_capital, params)

    plt.plot((port_df.Total - 40000) / 1000, color="blue")
    print(port_df["Total"][port_df.index[-1]])
    print(port_df["Return"].mean() / port_df["Return"].std() * (252**0.5))
    trade_df.to_csv("C:/Users/mario/python/trading/trades.csv")
    port_df.to_csv("C:/Users/mario/python/trading/port.csv")

    plt.show()