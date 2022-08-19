from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer
from PrivateUtils import util
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from PrivateUtils import global_var, send_email



IMAGESPATH = '../images/nn model without non scored[old cv]'


def read_csv_data():
    base_path = 'E:\AAAMyCodes\myjupyter\kaggle\Mechanisms of Action (MoA) Prediction\data'
    train_features = pd.read_csv(base_path + '/train_features.csv')
    train_targets_scored = pd.read_csv(base_path + '/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(base_path + '/train_targets_nonscored.csv')

    test_features = pd.read_csv(base_path + '/test_features.csv')
    sample_submission = pd.read_csv(base_path + '/sample_submission.csv')

    return train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission


def data_preprocessing(train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission):
    def my_plot(original=True):
        if original:
            global_var.get_value('email_log').print_add(f'观察原始数据图像并保存'.center(100, '*'))
        else:
            global_var.get_value('email_log').print_add(f'观察处理后数据图像并保存'.center(100, '*'))
        """观察原始数据"""
        plot_example_list = ['g-0', 'g-1', 'g-2', 'c-0', 'c-1', 'c-2']
        plt.figure(figsize=(10, 10))
        for i, plot_example in enumerate(plot_example_list):
            plt.subplot(4, 3, (i + 1))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            plt.title('train_features_' + plot_example)
            plt.hist(train_features[plot_example])

            plt.subplot(4, 3, (i + 1 + 6))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            plt.title('test_features_' + plot_example)
            plt.hist(test_features[plot_example])

    def transform_data(train_features, test_features):
        global_var.get_value('email_log').print_add(f'将特征转换成正态分布'.center(100, '*'))
        """将前四个特征之外的特征转换为正态分布（train、test分别转换为正态分布）"""
        # 这里面有些操作应该有更简洁的写法，但是这种属于同一种功能的不同写法，就不改了，浪费时间，大致理解其意思即可
        for col in (GENES + CELLS):
            # 设置train_features与test_features同列不同行
            # 很多机器学习算法偏好正态分布
            transformer = QuantileTransformer(n_quantiles=100, output_distribution="normal")
            # 设置train_features
            vec_len = len(train_features[col].values)
            vec_len_test = len(test_features[col].values)
            # train_features[col]是series类型，下面这个将其转换为(vec_len, 1)维（不是(vec_len,)维）的ndarray
            raw_vec = train_features[col].values.reshape(vec_len, 1)
            transformer.fit(raw_vec)
            train_features[col] = transformer.transform(raw_vec)[:, 0]  # transformer.transform(raw_vec)是转化为二维列向量

            # 设置test_features，不过为什么test_features也用train_features的fit，难道是前者的数据比后者的少？
            raw_vec2 = test_features[col].values.reshape(vec_len_test, 1)
            test_features[col] = transformer.transform(raw_vec2)[:, 0]

        return train_features, test_features

    def fa_adequacy_test_and_confirm_num(train_features, test_features):
        """
        原ipynb文件用的是因子分析降维，这里用了两个检验来确定因子分析降维的合理性
        至于选择因子数多少，sklearn的因子分析并没有给可解释方差这一属性，但是pca给了可解释方差，画的可解释方差累积图实际上是根据pca来画的
        """
        global_var.get_value('email_log').print_add(f'因子分析充分性检验，并确定因子分析数目'.center(100, '*'))
        global_var.get_value('email_log').print_add('GENES属性'.center(75, '*'))
        data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

        _, p_value = calculate_bartlett_sphericity(data)
        global_var.get_value('email_log').print_add(f'若巴特利特P值<0.01；则适合因子分析，实际P值为{p_value}')
        _, kmo_model = calculate_kmo(data)
        global_var.get_value('email_log').print_add(f'若KMO值>0.6，则适合因子分析；实际P值为{kmo_model}')

        util.plot_pca_var(data, [0, 120])
        util.save_fig(IMAGESPATH, 'GENE特征可解释方差图')
        plt.show()

        global_var.get_value('email_log').print_add('CELLS属性'.center(75, '*'))
        data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

        _, p_value = calculate_bartlett_sphericity(data)
        global_var.get_value('email_log').print_add(f'若巴特利特P值<0.01；则适合因子分析，实际P值为{p_value}')
        _, kmo_model = calculate_kmo(data)
        global_var.get_value('email_log').print_add(f'若KMO值>0.6，则适合因子分析；实际P值为{kmo_model}')

        util.plot_pca_var(data, [0, 80])
        util.save_fig(IMAGESPATH, 'CELLS特征可解释方差图')
        plt.show()

    def genes_factor_analysis(train_features, test_features):
        """原文件用的是因子分析，这里选用了更为
        因子分析, 因子分析是一种共线性分析方法，用于在大量变量中寻找和描述潜在因子
        https://www.displayr.com/factor-analysis-and-principal-component-analysis-a-simple-explanation/

        Factor analysis explicitly assumes the existence of latent factors underlying the observed data.
        PCA instead seeks to identify variables that are composites of the observed variables.
        """
        n_comp = 90  # <--Update # 因子分析的因子数（降低为90列）

        # 纵向拼接（仅是为了统一处理列，后面又分开），为啥要合在一起
        data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
        data2 = FactorAnalysis(n_components=n_comp).fit_transform(data[GENES])
        train2 = data2[:train_features.shape[0]]
        test2 = data2[-test_features.shape[0]:]

        # 为新获得的dataframe添加列名
        train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
        test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

        # 将pca后的特征与原特征横向拼接
        # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
        train_features = pd.concat((train_features, train2), axis=1)
        test_features = pd.concat((test_features, test2), axis=1)

        return train_features, test_features

    # 由数据描述，很自然分离出GENES与CELLS两大类特征，注意这里返回的是list类型
    pd.set_option('display.max_columns', 15)
    global_var.get_value('email_log').print_add('train_features.head()')
    global_var.get_value('email_log').print_add_df(train_features.head())
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    my_plot()
    util.save_fig(IMAGESPATH, 'some train and test features')
    plt.show()

    train_features, test_features = transform_data(train_features, test_features)
    my_plot(False)
    util.save_fig(IMAGESPATH, 'some train and test features after QuantileTransformer')
    plt.show()

    global_var.get_value('email_log').print_add(f'genes维数：{train_features[GENES].shape}')
    global_var.get_value('email_log').print_add(f'cells维数：{train_features[CELLS].shape}')

    fa_adequacy_test_and_confirm_num(train_features, test_features)
    train_features, test_features = genes_factor_analysis(train_features, test_features)


if __name__ == '__main__':
    global_var._init()
    global_var.set_value('email_log', send_email.Mylog(header='header', subject='subject', 
                                                       name='nn model without non scored[old cv]'))
    util.set_output_width()
    util.seed_everything()

    train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission = read_csv_data()
    data_preprocessing(train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission)

