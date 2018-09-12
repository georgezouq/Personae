import rqalpha

from rqalpha.api import *
from sklearn.preprocessing import MinMaxScaler

from strategy import config
from algorithm.RL.DoubleDQN import Algorithm
from deprecated.stock_market import ActionCode

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.s1 = '600036.XSHG'
    update_universe(context.s1)
    context.has_save_data = False

    mode = 'run'
    market = 'stock'
    training_data_ratio = 0.9
    train_steps = 10000
    episode = 3000

    base = config.get('base')

    context.s1 = '600036'

    context.bar_list_origin = []
    context.bar_list = []
    context.scale = MinMaxScaler()

    context.algorithm = Algorithm.generator(context.s, base.get('start_date'), base.get('end_date'))
    context.algorithm.run()

    subscribe(context.s1)


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


def process_data(context, bar_dict):
    s1 = bar_dict[context.s1]
    price = [s1.open, s1.high, s1.low, s1.close, s1.volume]
    context.bar_list_origin.append(price)

    scale = context.scale.fit(context.bar_list_origin)
    price_scaled = scale.transform([price])
    context.bar_list.append(price_scaled[0])

    # if not enough bar
    if len(context.bar_list) < context.algorithm.seq_length * 2 + 2:
        # TODO Use history_bars function
        # pre_data = history_bars
        return

    return context.bar_list[len(context.bar_list) - 1]


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    s = process_data(context, bar_dict)

    if s is None:
        return

    c, a, _ = context.algorithm.predict(s)
    s_next, r, status, info = context.algorithm.env.forward(c, a)

    if a == ActionCode.Buy:
        order(context.s1, 0.10)
        buy_open(context.s1, 100)

    elif a == ActionCode.Sell:
        buy_open(context.s1, 0)


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


rqalpha.run_func(init=init,
                 before_trading=before_trading,
                 handle_bar=handle_bar,
                 after_trading=after_trading,
                 config=config)
