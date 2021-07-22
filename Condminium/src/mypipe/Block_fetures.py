import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class BaseBlock(object):
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()


class ContinuousBlock(BaseBlock):
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        return input_df[self.column].copy()


class CountEncodingBlock(BaseBlock):
    def __init__(self, column, whole_df:pd.DataFrame):
        self.column = column
        self.whole_df = whole_df

    def transform(self, input_df):
        output_df = pd.DataFrame()
        c = self.column

        vc = self.whole_df[c].value_counts()
        output_df[c] = input_df[c].map(vc)
        return output_df.add_prefix("CE_")


class OneHotEncodingBlock(BaseBlock):
    def __init__(self, column, count_limit: int):
        self.column = column
        self.count_limit = count_limit

    def fit(self, input_df, y=None):
        vc = input_df[self.column].dropna().value_counts()
        cats_ = vc[vc > self.count_limit].index
        self.cats_ = cats_
        return self.transform(input_df)

    def transform(self, input_df):
        x = pd.Categorical(input_df[self.column], categories=self.cats_)
        output_df = pd.get_dummies(x, dummy_na=False)
        output_df.columns = output_df.columns.tolist()
        return output_df.add_prefix(f'OHE_{self.column}=')


class LabelEncodingBlock(BaseBlock):
    def __init__(self, column: str, whole_df: pd.DataFrame):
        self.column = column
        self.le = LabelEncoder()
        self.whole_df = whole_df

    def fit(self, input_df, y=None):
        self.le.fit(self.whole_df[self.column].fillna("nan"))
        return self.transform(input_df)

    def transform(self, input_df):
        c = self.column
        output_df = input_df.copy()
        output_df[c] = self.le.transform(input_df[c].fillna("nan")).astype("int")
        output_df = output_df[[c]]
        return output_df.add_prefix(f'LE_')


class WrapperBlock(BaseBlock):
    def __init__(self, function):
        self.function = function

    def transform(self, input_df):
        return self.function(input_df)


# 特定カラムの四則演算
class ArthmetricOperationBlock(BaseBlock):
    def __init__(self, target_column1: str, target_column2: str, operation: str):
        self.target_column1 = target_column1
        self.target_column2 = target_column2
        self.operation = operation

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = input_df.copy()
        output_df_columns_name = f'{self.target_column1}{self.operation}{self.target_column2}'

        if self.operation == "+":
            output_df[output_df_columns_name] = output_df[self.target_column1] + output_df[self.target_column2]

        elif self.operation == "-":
            output_df[output_df_columns_name] = output_df[self.target_column1] - output_df[self.target_column2]

        elif self.operation == "*":
            output_df[output_df_columns_name] = output_df[self.target_columns1] * output_df[self.target_columns2]

        elif self.operation == "/":
            output_df[output_df_columns_name] = output_df[self.target_columns1] / output_df[self.target_columns2]

        return output_df[output_df_columns_name]


# Aggregation
# なんかおかしい
class AggregationBlock(BaseBlock):
    def __init__(self, whole_df: pd.DataFrame, key: str, agg_column: str, agg_funcs: ["mean"], fillna=None):
        self.whole_df = whole_df
        self.key = key
        self. agg_column = agg_column
        self.agg_funcs = agg_funcs
        self.fillna = fillna

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):

        if self.fillna:
            self.whole_df[self.agg_column] = self.whole_df[self.agg_column].fillna(self.fillna)

        self.gp_df = self.whole_df.groupby(self.key).agg({self.agg_column: self.agg_funcs}).reset_index()
        column_names = [f'GP_{self.agg_column}@{self.key}_{agg_func}' for agg_func in self.agg_funcs]

        self.gp_df.columns = [self.key] + column_names
        output_df = pd.merge(input_df[self.key], self.gp_df, on=self.key, how="left").drop(columns=[self.key])
        print(output_df)
        return output_df


