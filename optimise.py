#
# optimise.py
#
# Purpose: Optimise the parameters in the strategy and output parameters in csv file for use in walk forward test
# Uses multiprocessing and brute force to check all solutions and optimises either Total Return or Sharpe Ratio
#
#
# Revision History
# When      Who         What
# 20200711  Marios C    Created
#

from multiprocessing import Queue, Process
import util
import strategy as strat
import itertools
import numpy as np
import pandas as pd

if __name__ == '__main__':
    queue = Queue()
    [tickers, indicators, test_start, test_end, initial_capital, trade_size, params_static] = strat.get_static_data()
    [train_yrs, test_yrs, params, to_maximise] = strat.get_training_data()

    params_list = []
    for [param_start, param_end, param_step] in params:
        print([param_start, param_end, param_step])
        iterator = itertools.count(param_start, param_step)
        params_list.append(list(next(iterator) for _ in np.arange(1 + (param_end - param_start) / param_step)))

    print(params_list)
    all_params = list(itertools.product(*params_list))
    cols = ["Test_Start_Date", "Test_End_Date"]
    print(all_params)
    for param_n in range(len(all_params[0])):
        cols.append("Param_" + str(param_n))

    out_params_df = pd.DataFrame(columns=cols)
    all_start = util.add_yrs(test_start, -train_yrs)
    all_data = util.get_data(tickers, indicators, all_start, test_end, False)
    all_data = strat.add_signals(all_data)
    train_end = test_start
    train_start = util.add_yrs(train_end, -train_yrs)
    while train_end < test_end:
        print("Optimising parameters for period " + str(train_start) + " to " + str(train_end))
        train_data = all_data[(all_data.Date >= train_start) & (all_data.Date < train_end)].copy().drop(
            columns=["Date"], axis=1, level=0
        )
        train_data.columns = train_data.columns.remove_unused_levels()
        train_data = strat.add_signals(train_data)
        for params in all_params:
            print("Testing " + str(params))
            p = Process(target=util.test_params, args=(params, train_data, trade_size, initial_capital, queue))
            p.start()

        p.join()

        [maxParams, maxValue] = [0, -100]
        while not queue.empty():
            [params, total, sharpe] = queue.get()
            test_out = total if to_maximise == "Total" else sharpe
            if test_out > maxValue:
                maxValue = test_out
                maxParams = params

        print("Optimised parameters are: " + str(maxParams) + " with returned " + to_maximise + " of " + maxValue)
        out_params_df = out_params_df.append(
            pd.DataFrame([[train_end, util.add_yrs(train_end, test_yrs)] + list(maxParams)],
                         columns=out_params_df.columns), ignore_index=True)
        train_end = util.add_yrs(train_end, test_yrs)
        train_start = util.add_yrs(train_end, -train_yrs)

    out_params_df.to_csv(strat.param_file, index=False)
