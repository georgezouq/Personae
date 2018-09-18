import rqalpha

from rqalpha.api import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from strategy import config
from algorithm.RL.DoubleDQN import Algorithm
from base.env.trader import ActionCode
from sklearn.preprocessing import StandardScaler


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # context.s1 = '600036.XSHG'
    context.codes_sort = ["600036", "601328", "601998", "601398"]
    context.codes = ["600036.XSHG", "601328.XSHG", "601998.XSHG", "601398.XSHG"]
    scale = StandardScaler
    context.has_save_data = False
    context.account_amount = config.get('base').get('accounts').get('stock')

    base = config.get('base')

    context.bar_list_origin = []
    context.bar_list = []
    # context.scales = [scale() for _ in context.codes]
    context.scale = scale()

    context.algorithm = Algorithm.generator(context.codes_sort, base.get('start_date'), base.get('end_date'))
    context.algorithm.restore()

    subscribe(context.codes)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


def _get_portfolio_state(context):
    portfolio = [context.portfolio.unit_net_value / context.account_amount,
                 context.portfolio.market_value / context.account_amount]
    for code in context.codes:
        if not context.portfolio.accounts.get('STOCK').positions.get(code):
            portfolio.append(0.0)
            continue

        portfolio.append(context.portfolio.accounts.get('STOCK').positions.get(code).quantity / context.account_amount)

    return portfolio


def process_data(context, bar_dict):
    data = []

    for code in context.codes:
        s1 = bar_dict[code]
        data.extend([s1.open, s1.high, s1.low, s1.close, s1.volume])

    scale = context.scale.fit(np.array([data]).T)
    data_scaled = scale.transform([data])[0]

    data_scaled = np.insert(data_scaled, -1, _get_portfolio_state(context)).reshape((1, -1))

    context.bar_list_origin.append(data)
    context.bar_list.append(data_scaled)

    return context.bar_list[len(context.bar_list) - 1]


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    s = process_data(context, bar_dict)

    if s is None:
        return

    c, a, _ = context.algorithm.predict(s)
    context.algorithm.train()
    # s_next, r, status, info = context.algorithm.env.forward(c, a)

    code = c + '.XSHG'

    if a == ActionCode.Buy.value:
        # order(code, 0.10)
        order_percent(code, 0.2)
        print('Buy Signal:', code)

    elif a == ActionCode.Sell.value:
        order_percent(code, 0)
        print('Sell Signal:', code)


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


rqalpha.run_func(init=init,
                 before_trading=before_trading,
                 handle_bar=handle_bar,
                 after_trading=after_trading,
                 config=config)
