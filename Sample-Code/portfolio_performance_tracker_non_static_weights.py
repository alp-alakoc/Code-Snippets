import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#The following is a snap-shot of code written to aggregate portfolio cumulative performances
#based off evolving optimal allocations and stock constituents.

#The performance_tracker function below assumes a buy and hold strategy until new optimal allocations
#are computed during the investment period...As a results the underlying weights of the portfolio are
#dynamic and move with day to day market fluctuations during a given back-test scenario

def performance_tracker_non_stat(opt_weights, stock_returns, stock_prices):

    prices_list = []
    for idx, i in enumerate(range(0, len(stock_prices), 80)):
        port_prices = stock_prices[(i + 60):(i + 80)]
        prices_list.append(port_prices)

    units_list = []
    units_list.append(np.array(opt_weights[0]) * 100000 / np.array(prices_list[0].iloc[0, :]))

    evolved_list = []

    for idx, i in enumerate(range(1, 21)):
        evolved = units_list[idx] * prices_list[idx]
        evolved['NAV'] = evolved.sum(axis=1)
        df = evolved[['NAV']].pct_change()
        df = df.fillna(df.mean())
        evolved_list.append(df)
        if (idx != 19):
            units_list.append(np.array(opt_weights[i]) * float(np.array(evolved[['NAV']].iloc[-1, -1])) / np.array(
                prices_list[i].iloc[0, :]))
        if (idx == 19):
            break

    daily_rets = pd.concat(evolved_list, axis=0)

    cum_performance = []
    for i in range(len(evolved_list)):
        cum = (1 + evolved_list[i]).cumprod() - 1
        cum_performance.append(cum)

    # set first cumulative performance to zero#
    for i in range(len(cum_performance)):
        cum_performance[i][:1] = 0

        ##Accumulate average monthly performanes#
    average_monthly_performance = []
    for i in range(len(cum_performance)):
        average_monthly_performance.append(np.mean(cum_performance[i].values))

    average_monthly_performance = np.array(average_monthly_performance)

    ##Acumulate monthly_sharpe#
    sharpe_monthly = []
    for i in range(len(cum_performance)):
        sharpe_monthly.append((np.mean(cum_performance[i].values) - 0.002) / np.std(cum_performance[i].values))

    sharpe_monthly = np.array(sharpe_monthly)

    # stack cumulative performances#
    for i in range(len(cum_performance) - 1):
        cum_performance[i + 1] = cum_performance[i + 1] + float(cum_performance[i][-1:].values)

    cumulative_portfolio_performance = pd.concat(cum_performance, axis=0)
    cumulative_portfolio_performance.reset_index(inplace=True)
    cumulative_portfolio_performance = pd.DataFrame(cumulative_portfolio_performance.iloc[:, 1])
    ################################################################################################

    # compute market performance#
    sp500 = []
    for i in range(2000, 2400, 20):
        sp500.append(stock_returns[['SPX']][i:(i + 20)])

    # compute cumulative market performance#
    cum_sp500 = []
    for i in range(len(sp500)):
        cum = ((1 + sp500[i]).cumprod() - 1)
        cum_sp500.append(cum)

    # set first cumulative performance to zero#
    for i in range(len(cum_sp500)):
        cum_sp500[i][:1] = 0

        # Accumulate market monthly average performance
    mm_average = []
    for i in range(len(cum_sp500)):
        mm_average.append(np.mean(cum_sp500[i].values))

    mm_average = np.array(mm_average)

    # Accumulate monthly sharpe#
    sharpe_mm = []
    for i in range(len(cum_sp500)):
        sharpe_mm.append((np.mean(cum_sp500[i].values) - 0.002) / np.std(cum_sp500[i].values))

    sharpe_mm = np.array(sharpe_mm)

    # stack cumulative market performances#
    for i in range(len(cum_sp500) - 1):
        cum_sp500[i + 1] = cum_sp500[i + 1] + float(cum_sp500[i][-1:].values)

    cumulative_market_performance = pd.concat(cum_sp500, axis=0)
    cumulative_market_performance.reset_index(inplace=True)
    cumulative_market_performance = cumulative_market_performance.iloc[:, 1]
    cumulative_market_performance = pd.DataFrame(cumulative_market_performance)

    ################################################################################################

    fig1, ax1 = plt.subplots(figsize=(20, 7))
    fig1.suptitle('Model Vs Market Cumulative Performance', fontsize=16)
    ax1.plot(cumulative_portfolio_performance, label='Model')
    ax1.plot(cumulative_market_performance, label='Market')

    plt.legend()
    plt.grid()

    sharpe_model = (np.mean(cumulative_portfolio_performance.values) - 0.02) / float(cumulative_portfolio_performance.std())
    sharpe_market = (np.mean(cumulative_market_performance.values) - 0.02) / float(cumulative_market_performance.std())

    print('\n', 'Model\'s Average Return: {}'.format(np.mean(cumulative_portfolio_performance.values)))
    print('\n', 'Market\'s Average Return: {}'.format(np.mean(cumulative_market_performance.values)))

    print('\n', 'Model\'s Sharpe Ratio: {}'.format(sharpe_model))
    print('\n', 'Market\'s Sharpe Ratio: {}'.format(sharpe_market))

    return cumulative_portfolio_performance, average_monthly_performance, sharpe_monthly, cumulative_market_performance, mm_average, sharpe_mm, daily_rets