import rqalpha
import os
import tensorflow as tf

from rqalpha.api import *
from sklearn.preprocessing import MinMaxScaler

from strategy import config
from base.env.market import Market
from algorithm import config as alg_config
from algorithm.SL.DualAttnRNN import Algorithm
from checkpoints import CHECKPOINTS_DIR


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.s1 = '600036'
    update_universe(context.s1)
    context.has_save_data = False

    mode = 'run'
    market = 'stock'
    training_data_ratio = 0.9
    train_steps = 30000

    base = config.get('base')

    codes = [context.s1]
    env = Market(codes, start_date=base.get('start_date'), end_date=base.get('end_date'), **{
        "market": market,
        "use_sequence": True,
        "scaler": MinMaxScaler,
        "mix_index_state": True,
        "training_data_ratio": training_data_ratio
    })

    model_name = os.path.basename(__file__).split('.')[0]

    context.algorithm = Algorithm(
        tf.Session(config=alg_config), env, env.seq_length, env.data_dim, env.code_count, **{
            "mode": mode,
            "hidden_size": 5,
            "enable_saver": True,
            "train_steps": train_steps,
            "enable_summary_writer": True,
            "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
        }
    )


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    print('bar_dict:', bar_dict)
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


rqalpha.run_func(init=init,
                 before_trading=before_trading,
                 handle_bar=handle_bar,
                 after_trading=after_trading,
                 config=config)
