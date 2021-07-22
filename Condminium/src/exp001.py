import os
import warnings
import sys
import joblib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from time import time
from contextlib import contextmanager

sys.path.append("../")

from mypipe.config import Config
from mypipe.utils import reduce_mem_usage
from mypipe.experiment import exp_env
from mypipe.experiment.runner import Runner
from mypipe.models.model_lgbm import MyLGBMModel
from mypipe.models.model_xgb import MyXGBModel
from mypipe.Block_fetures import BaseBlock, ContinuousBlock, CountEncodingBlock, WrapperBlock, \
    LabelEncodingBlock, ArthmetricOperationBlock, AggregationBlock


# ---------------------------------------------------------------------- #
exp = "exp001"
config = Config(EXP_NAME=exp, TARGET="PRICE")
exp_env.make_env(config)

address_df = pd.read_csv(config.INPUT + "/raw_address_points.csv")
census_df = pd.read_csv(config.INPUT + "/raw_census_tracts_in_2010.csv")
# ---------------------------------------------------------------------- #


# 評価関数
def root_mean_squared_error(true, pred):
    return mean_squared_error(true, pred)**.5


# 前処理関数
def preprocess_train(input_df):
    input_df = input_df[input_df["SOURCE"] == "Condominium"]
    input_df = input_df[input_df["PRICE"] != 1]
    input_df = input_df[np.log1p(input_df["PRICE"]) >= 10]
    input_df = input_df[np.log1p(input_df["PRICE"]) <= 16]

    # キッチン数の異常値処理
    input_df.loc[input_df["KITCHENS"] > 40, "KITCHENS"] = 4

    # 暖炉数の異常値処理
    input_df = input_df[input_df["FIREPLACES"] < 11]

    # HF_BATHRMの異常値削除（1件）
    input_df = input_df[input_df["HF_BATHRM"] < 10]

    # YR_RMDLの異常値除外
    input_df["YR_RMDL"].replace(20, 2000, inplace=True)
    input_df['AYB'].fillna(input_df['AYB'].mean(), inplace=True)
    input_df["ROOMS"] = input_df[input_df["ROOMS"] < 25]

    # AC修正
    input_df["AC"] = input_df["AC"].apply(lambda x: 1 if x == "Y" else 0)

    input_df = input_df[input_df["GBA"].fillna(0) < 15000]
    input_df = input_df[input_df["LIVING_GBA"].fillna(0) < 6000]

    input_df = input_df[input_df["LANDAREA"] < 75000]

    # convert dtypes
    input_df["ZIPCODE"] = input_df["ZIPCODE"].astype(str)
    input_df["CENSUS_TRACT"] = input_df["CENSUS_TRACT"].astype(str)
    input_df["USECODE"] = input_df["USECODE"].astype(str)

    # "QUADRANT"の欠損値補間
    fillna_list_tr = input_df[input_df["QUADRANT"].isnull()].Id.tolist()

    for i in fillna_list_tr:
        fill_qued = input_df[input_df["Id"] == i]["FULLADDRESS"].str[-2:]
        input_df.loc[input_df["Id"] == i, "QUADRANT"] = fill_qued

    output_df = input_df.copy()

    return output_df.reset_index(drop=True)


def preprocess_test(input_df):
    input_df = input_df[input_df["SOURCE"] == "Condominium"]
    # 暖炉数の異常値処理
    input_df.loc[input_df["FIREPLACES"] > 10, "FIREPLACES"] = 1

    input_df['AYB'].fillna(input_df['AYB'].mean(), inplace=True)
    input_df.loc[input_df["STORIES"]>10, "STORIES"] = 8

    # AC修正
    input_df["AC"] = input_df["AC"].apply(lambda x: 1 if x == "Y" else 0)
    input_df.loc[input_df["LANDAREA"] > 30000, "LANDAREA"] = 23000

    # convert dtypes
    input_df["ZIPCODE"] = input_df["ZIPCODE"].astype(str)
    input_df["CENSUS_TRACT"] = input_df["CENSUS_TRACT"].astype(str)
    input_df["USECODE"] = input_df["USECODE"].astype(str)

    # "QUADRANT"の欠損値補間
    fillna_list_tr = input_df[input_df["QUADRANT"].isnull()].Id.tolist()

    for i in fillna_list_tr:
        fill_qued = input_df[input_df["Id"] == i]["FULLADDRESS"].str[-2:]
        input_df.loc[input_df["Id"] == i, "QUADRANT"] = fill_qued

    output_df = input_df.copy()

    return output_df.reset_index(drop=True)


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def get_function(block, is_train):
    s = mapping = {
        True: 'fit',
        False: 'transform'
    }.get(is_train)
    return getattr(block, s)


def to_feature(input_df,
               blocks,
               is_train=False):
    out_df = pd.DataFrame()

    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)

        with timer(prefix='create ' + str(block) + ' '):
            _df = func(input_df)
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
    return reduce_mem_usage(out_df)


# get age
class AgeBlock(BaseBlock):
    def __init__(self, columns: str):
        self.columns = columns

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df_col_name = f'Age@{self.columns}'

        output_df[output_df_col_name] = 2018 - input_df[self.columns]

        return output_df[output_df_col_name]


def get_sales_date(input_df):
  _df = input_df.copy()
  _df['SALEDATE'] = pd.to_datetime(_df['SALEDATE'])
  output_df = pd.DataFrame()
  output_df["Sale_Year"] = _df['SALEDATE'].dt.year
  output_df["Sale_Month"] = _df['SALEDATE'].dt.month

  return output_df


# make KFold
def make_kf(train_x, train_y, n_splits, random_state=71):
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    s = 5
    _y = pd.cut(train_y, s, labels=range(s))
    return list(skf.split(train_x, _y))


# plot result
def result_plot(train_y, oof):
    name = "result"
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(train_y, label='train_y', color='orange')
    sns.distplot(oof, label='oof')
    ax.legend()
    ax.grid()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORTS, f'{name}.png'), dpi=120)  # save figure
    plt.show()


# create submission
def create_submission(preds):
    sample_sub = pd.read_csv(os.path.join(config.INPUT, "submission.csv"))
    post_preds = [0 if x < 0 else x for x in preds]
    sample_sub["PRICE"] = post_preds
    sample_sub.to_csv(os.path.join(config.SUBMISSION, f'{config.EXP_NAME}.csv'), index=False)


# ---------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # DtypeWaringが出る場合がある
    # https://qiita.com/wwacky/items/6ab49a9fccc59b81411e
    train_df = pd.read_csv(os.path.join(config.INPUT, "DC_train.csv"))
    test_df = pd.read_csv(os.path.join(config.INPUT, "DC_test.csv"))

    # 異常値・欠損値補間の実施
    train = preprocess_train(train_df)
    test = preprocess_test(test_df)
    print(train)

    whole_df = pd.concat([train, test], axis=0)

    process_blocks = [
        *[CountEncodingBlock(c, whole_df=whole_df) for c in ['HEAT',
                                                             'QUALIFIED',
                                                             'ASSESSMENT_NBHD',
                                                             'WARD',
                                                             'QUADRANT',
                                                             'ZIPCODE',
                                                             'CENSUS_TRACT',
                                                             'USECODE']],
        *[LabelEncodingBlock(c, whole_df=whole_df) for c in ['HEAT',
                                                             'QUALIFIED',
                                                             'ASSESSMENT_NBHD',
                                                             'WARD',
                                                             'QUADRANT',
                                                             'ZIPCODE',
                                                             'CENSUS_TRACT',
                                                             'USECODE']],
        *[ContinuousBlock(c) for c in ['AC',
                                       'BATHRM',
                                       'HF_BATHRM',
                                       'ROOMS',
                                       'BEDRM',
                                       'AYB',
                                       'YR_RMDL',
                                       'EYB',
                                       'SALE_NUM',
                                       'FIREPLACES',
                                       'LANDAREA',
                                       'CMPLX_NUM',
                                       'LIVING_GBA',
                                       'LATITUDE',
                                       'LONGITUDE',
                                       ]],
        *[AgeBlock(c) for c in ['AYB', 'YR_RMDL', 'EYB']],
        ArthmetricOperationBlock(target_column1='BATHRM', target_column2='HF_BATHRM', operation="+"),
        ArthmetricOperationBlock(target_column1='ROOMS', target_column2='BEDRM', operation="-"),
        WrapperBlock(get_sales_date),
        AggregationBlock(whole_df=whole_df, key="ZIPCODE",agg_column="BATHRM",agg_funcs=["mean", "std"]),
        AggregationBlock(whole_df=whole_df, key="WARD",agg_column="BATHRM",agg_funcs=["mean", "std"])
    ]

    # create train_x, y, test_x
    train_y = train['PRICE']
    train_x = to_feature(train, process_blocks, is_train=True)
    test_x = to_feature(test, process_blocks)
    print(train_y)
    print(test_x.shape)

    # dump feature
    joblib.dump(train_x, os.path.join("../output/" + exp + "/feature", 'train_feat.pkl'))
    joblib.dump(test_x, os.path.join("../output/" + exp + "/feature", 'test_feat.pkl'))

    # set model
    model = MyLGBMModel

    # set run params
    run_params = {
        "metrics": root_mean_squared_error,
        "cv": make_kf,
        "feature_select_method": "tree_importance",
        "feature_select_fold": 5,
        "feature_select_num": 300,
        "folds": 5,
        "seeds": [0, 1, 2],
    }

    # set model params
    model_params = {
        "n_estimators": 20000,
        "objective": 'rmse',
        "learning_rate": 0.01,
        "num_leaves": 36,
        "random_state": 71,
        "n_jobs": -1,
        "importance_type": "gain",
        'colsample_bytree': .5,
        "reg_lambda": 5,
        "max_depth": 5,
    }

    # fit params
    fit_params = {
        "early_stopping_rounds": 50,
        "verbose": 1000
    }

    # features
    features = {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": np.log1p(train_y)
    }

    # run!
    config.RUN_NAME = f"_{config.TARGET}"
    runner = Runner(config=config,
                    run_params=run_params,
                    model_params=model_params,
                    fit_params=fit_params,
                    model=model,
                    features=features,
                    use_mlflow=False
                    )
    runner.run_train_cv()
    runner.run_predict_cv()

    # make submission
    #create_submission(preds=np.expm1(runner.preds))

    # plot result
    result_plot(train_y=np.log1p(train_y), oof=runner.oof)


if __name__ == "__main__":
    main()

