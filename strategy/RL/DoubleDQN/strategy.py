import rqalpha
import os
import tensorflow as tf
import numpy as np

from rqalpha.api import *
from sklearn.preprocessing import MinMaxScaler

from strategy import config
from base.env.market import Market
from algorithm import config as alg_config
from algorithm.RL.DoubleDQN import Algorithm
from checkpoints import CHECKPOINTS_DIR
from helper.data_logger import generate_algorithm_logger, generate_market_logger


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

    codes = ['600036']
    env = Market(codes, start_date=base.get('start_date'), end_date=base.get('end_date'), **{
        "market": market,
        "use_sequence": True,
        "scaler": MinMaxScaler,
        "mix_index_state": True,
        "training_data_ratio": training_data_ratio
    })

    model_name = os.path.basename(__file__).split('.')[0]

    context.bar_list_origin = []
    context.bar_list = []
    context.scale = MinMaxScaler()

    context.algorithm = Algorithm(
        tf.Session(config=alg_config), env, env.seq_length, env.data_dim, env.code_count, **{
            "mode": mode,
            "episodes": episode,
            "enable_saver": True,
            "learning_rate": 0.003,
            "enable_summary_writer": True,
            "logger": generate_algorithm_logger(model_name),
            "train_steps": train_steps,
            "save_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "RL", model_name, market, "summary"),
        }
    )


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    c, a, _ = context.algorithm.predict(s)
    s_next, r, status, info = context.algorithm.env.forward(c, a)

    # s1 = bar_dict[context.s1]
    # price = [s1.open, s1.high, s1.low, s1.close, s1.volume]
    # context.bar_list_origin.append(price)
    #
    # scale = context.scale.fit(context.bar_list_origin)
    # price_scaled = scale.transform([price])
    # context.bar_list.append(price_scaled[0])
    #
    # # if not enough bar
    # if len(context.bar_list) < context.algorithm.seq_length * 2 + 2:
    #     return
    #
    # # frm = len(context.bar_list)-context.algorithm.seq_length
    # x1 = context.bar_list[-context.algorithm.seq_length*2:-context.algorithm.seq_length]
    # x2 = context.bar_list[-context.algorithm.seq_length:]
    #
    # x = [x1, x2]
    # c, a, _ = context.algorithm.predict(x)
    # res = np.append(_, [0, 0, 0, 0])
    #
    # predict = scale.inverse_transform([res])
    # print('predict', predict)
    # pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


rqalpha.run_func(init=init,
                 before_trading=before_trading,
                 handle_bar=handle_bar,
                 after_trading=after_trading,
                 config=config)
