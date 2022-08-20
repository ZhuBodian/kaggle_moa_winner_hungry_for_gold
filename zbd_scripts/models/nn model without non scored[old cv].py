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
from sklearn.cluster import KMeans


IMAGESPATH = './images/nn model without non scored[old cv]/'
LOGPATH = './log/'
PICKLEPATH = './pickle_data/nn model without non scored[old cv]/'


def read_csv_data():
    base_path = 'E:\AAAMyCodes\myjupyter\kaggle\Mechanisms of Action (MoA) Prediction\data'
    train_features = pd.read_csv(base_path + '/train_features.csv')
    train_targets_scored = pd.read_csv(base_path + '/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(base_path + '/train_targets_nonscored.csv')

    test_features = pd.read_csv(base_path + '/test_features.csv')
    sample_submission = pd.read_csv(base_path + '/sample_submission.csv')

    return train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission


def data_preprocessing(train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission):
    def my_plot(fig_name):
        print_str = 'plot_' + fig_name
        global_var.get_value('email_log').print_add(print_str.center(100, '*'))
        """观察数据"""
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

    def factor_analysis(train_features, test_features):
        """原文件用的是因子分析，这里选用了更为
        因子分析, 因子分析是一种共线性分析方法，用于在大量变量中寻找和描述潜在因子
        https://www.displayr.com/factor-analysis-and-principal-component-analysis-a-simple-explanation/

        Factor analysis explicitly assumes the existence of latent factors underlying the observed data.
        PCA instead seeks to identify variables that are composites of the observed variables.
        """
        global_var.get_value('email_log').print_add(f'因子分析'.center(100, '*'))
        global_var.get_value('email_log').print_add(f'GENES特征因子分析'.center(75, '*'))
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

        global_var.get_value('email_log').print_add(f'CELLS特征因子分析'.center(75, '*'))
        # CELLS
        n_comp = 50  # <--Update

        data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
        data2 = FactorAnalysis(n_components=n_comp).fit_transform(data[CELLS])
        train2 = data2[:train_features.shape[0]]
        test2 = data2[-test_features.shape[0]:]

        train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
        test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

        # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
        train_features = pd.concat((train_features, train2), axis=1)
        test_features = pd.concat((test_features, test2), axis=1)

        global_var.get_value('email_log').print_add('factor_analysis添加新列后,train_features的特征：')
        global_var.get_value('email_log').print_add(str(list(train_features.columns.values)))
        global_var.get_value('email_log').print_add(f'维数：{train_features.shape}')

        return train_features, test_features

    def transform_data2(train_features, test_features):
        global_var.get_value('email_log').print_add('将前四个特征之外的特征转换为正态分布'.center(100, '*'))
        # var_thresh = VarianceThreshold(0.8)  #<-- Update
        var_thresh = QuantileTransformer(n_quantiles=100, output_distribution="normal")

        data = train_features.append(test_features)  # 纵向拼接
        data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])  # 仅处理第四列之外的特征

        # 将处理过之后的数据拆分
        train_features_transformed = data_transformed[: train_features.shape[0]]
        test_features_transformed = data_transformed[-test_features.shape[0]:]

        column_names = train_features.columns
        # 重组train_features
        train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4),
                                      columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

        train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
        train_features.columns = column_names

        # 重组test_features
        test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4),
                                     columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

        test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)
        test_features.columns = column_names

        return train_features, test_features

    def decide_prpoer_K(train, test):
        """
        太费时间，运行一次即可
        """
        global_var.get_value('email_log').print_add('找寻K均值合适的聚类数目'.center(100, '*'))
        features_g = list(train.columns[4:776])  # 4:776这个要自己观察dataframe数据去找
        features_c = list(train.columns[776:876])

        global_var.get_value('email_log').print_add('GENES特征'.center(75, '*'))
        model = KMeans()
        util.find_proper_k(model, train[features_g], (30, 100 + 1), IMAGESPATH, '基因特征聚类手肘图')

        global_var.get_value('email_log').print_add('CELLS特征'.center(75, '*'))
        model = KMeans()
        util.find_proper_k(model, train[features_c], (10, 50 + 1), IMAGESPATH, '细胞特征聚类手肘图')
        util.program_done_sound()

    def fe_cluster(train, test, n_clusters_g = 45, n_clusters_c=15):
        global_var.get_value('email_log').print_add('聚类'.center(100, '*'))
        # 返回的是int值列表
        features_g = list(train.columns[4:776])  # 4:776这个要自己观察dataframe数据去找
        features_c = list(train.columns[776:876])

        def create_cluster(train, test, features, kind='g', n_clusters=n_clusters_g):
            # features是int列表，包含了要进行聚类的gene/cell的列索引，注意用来聚类的特征并不包括降维后的特征
            train_ = train[features].copy()
            test_ = test[features].copy()
            data = pd.concat([train_, test_], axis=0)
            kmeans = KMeans(n_clusters=n_clusters).fit(data)  # 这里为什么要将train与test的数据一同fit？（为了保证类别划分的标准一致？）
            train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]  # 添加一个新列，这个列标记了样本按gene/cell进行的分类结果
            test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
            train = pd.get_dummies(train, columns=[f'clusters_{kind}'])  # 将类别节点（原来为[0, n_clusters)的整数）转换为哑结点(1,0,0,0……0这种形式)
            test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
            return train, test

        global_var.get_value('email_log').print_add('GENES特征聚类'.center(75, '*'))
        # 注意运行create_cluster是对train与test添加新列，并不会覆盖老数据，所以这种写法没关系
        train, test = create_cluster(train, test, features_g, kind='g', n_clusters=n_clusters_g)
        global_var.get_value('email_log').print_add('CELLS特征聚类'.center(75, '*'))
        train, test = create_cluster(train, test, features_c, kind='c', n_clusters=n_clusters_c)

        global_var.get_value('email_log').print_add('KMeans后添加新列后，train_features的特征：')
        global_var.get_value('email_log').print_add(str(list(train.columns.values)))
        global_var.get_value('email_log').print_add(f'维数：{train.shape}')
        return train, test

    def transform_data3(train_features, train_targets_scored, test_features):
        global_var.get_value('email_log').print_add('transform_data3'.center(100, '*'))

        # merge 函数通过一个或多个键将数据集的行连接起来。
        # 场景：针对同一个主键存在的两张包含不同特征的表，通过主键的链接，将两张表进行合并。合并之后，两张表的行数不增加，列数是两张表的列数之和。
        train = train_features.merge(train_targets_scored, on='sig_id')
        # 这个拼接的目的是一同丢弃cp_type'=='ctl_vehicle的行（出于代码书写简洁考虑）
        train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)  # drop=True为重新添加连续索引之后，将老索引删除
        test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

        target = train[train_targets_scored.columns]  # 一同丢弃相应行后，再分离出target（注意这里的train中仍然包含着target）

        # cp_type现在没用了，删除
        train = train.drop('cp_type', axis=1)
        test = test.drop('cp_type', axis=1)

        # 获得target的各列名
        target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

        folds = train.copy()

        # MultilabelStratifiedKFold是用于多标签分层的K折交叉验证
        mskf = MultilabelStratifiedKFold(n_splits=5)

        # 不清楚为啥非要用类型转换
        # v_idx返回的是第f+1次k折交叉验证的验证集索引
        # 至于这个x其实对函数计算并没有帮助，只不过根据官网介绍sklearn的非分层抽样要x，分层抽样算法上不用x，但为了兼容，还是要了x，本库为了与sklearn兼容，也要x了
        for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
            folds.loc[v_idx, 'kfold'] = int(f)  # 添加一个新的列来记录下第f+1轮的验证集索引（与此对应的，第f+1轮，该列数值不是f的样本，即是训练集样本）

        folds['kfold'] = folds['kfold'].astype(int)

        # folds仅仅比train多了一列k-fold相关的列
        global_var.get_value('email_log').print_add('transform_data3后添加新列后，folds的特征：')
        global_var.get_value('email_log').print_add(str(list(folds)))
        global_var.get_value('email_log').print_add(f'维数：{folds.shape}')

        return train, folds, test, target, target_cols



    # 由数据描述，很自然分离出GENES与CELLS两大类特征，注意这里返回的是list类型
    pd.set_option('display.max_columns', 15)
    global_var.get_value('email_log').print_add('train_features.head()：')
    global_var.get_value('email_log').print_add_not_str(train_features.head())
    global_var.get_value('email_log').print_add('train_features.columns：')
    global_var.get_value('email_log').print_add_not_str(train_features.columns.values)

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    my_plot('some train and test features')
    util.save_fig(IMAGESPATH, 'some train and test features')
    plt.show()

    train_features, test_features = transform_data(train_features, test_features)
    my_plot('some train and test features after transform_data')
    util.save_fig(IMAGESPATH, 'some train and test features after transform_data')
    plt.show()

    global_var.get_value('email_log').print_add(f'genes维数：{train_features[GENES].shape}')
    global_var.get_value('email_log').print_add(f'cells维数：{train_features[CELLS].shape}')

    fa_adequacy_test_and_confirm_num(train_features, test_features)
    train_features, test_features = factor_analysis(train_features, test_features)
    train_features, test_features = transform_data2(train_features, test_features)

    my_plot('some train and test features after transform_data2')
    util.save_fig(IMAGESPATH, 'some train and test features after transform_data2')
    plt.show()

    # decide_prpoer_K(train_features, test_features)
    train_features, test_features = fe_cluster(train_features, test_features)

    train_features, test_features, test, target, target_cols = transform_data3(train_features, train_targets_scored, test_features)

    util.save_as_pickle(PICKLEPATH, 'train_features', train_features)
    util.save_as_pickle(PICKLEPATH, 'test_features', test_features)
    util.save_as_pickle(PICKLEPATH, 'test', test)
    util.save_as_pickle(PICKLEPATH, 'target', target)
    util.save_as_pickle(PICKLEPATH, 'target_cols', target_cols)

    return train_features, test_features, test, target, target_cols




if __name__ == '__main__':
    First_train = False
    global_var._init()
    global_var.set_value('email_log', send_email.Mylog(header='header', subject='subject', 
                                                       name='nn model without non scored[old cv]', folder_path=LOGPATH))
    util.set_output_width()
    util.seed_everything()

    if First_train:
        global_var.get_value('email_log').print_add('首次运行数据预处理'.center(100, '*'))

        train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission = read_csv_data()
        train_features, test_features, test, target, target_cols = data_preprocessing(
            train_features, train_targets_scored, train_targets_nonscored, test_features, sample_submission)
    else:
        global_var.get_value('email_log').print_add('非首次运行数据预处理，读取pickle文件'.center(100, '*'))

        train_features = util.load_from_pickle(PICKLEPATH, 'train_features')
        test_features = util.load_from_pickle(PICKLEPATH, 'test_features')
        test = util.load_from_pickle(PICKLEPATH, 'test')
        target = util.load_from_pickle(PICKLEPATH, 'target')
        target_cols = util.load_from_pickle(PICKLEPATH, 'target_cols')

