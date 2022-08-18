import sys
'''
sys.path.append('../input/iterativestratification')
'''
from sklearn.cluster import KMeans
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
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer
from utils import util


'''
os.listdir('../input/lish-moa')
'''
'''
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
'''


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data(base_path):
    train_features = pd.read_csv(base_path + '/train_features.csv')
    train_targets_scored = pd.read_csv(base_path + '/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(base_path + '/train_targets_nonscored.csv')
    test_features = pd.read_csv(base_path + '/test_features.csv')
    sample_submission = pd.read_csv(base_path + '/sample_submission.csv')

    return train_features, test_features, train_targets_scored, train_targets_nonscored, sample_submission


def preprocess(train_features, test_features):
    print('对原始数据preprocess'.center(100, '*'))

    # 由数据描述，很自然分离出GENES与CELLS两大类特征，注意这里返回的是list类型
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    # 将前四个特征之外的特征转换为正态分布（train、test分别转换为正态分布）
    # RankGauss
    # str的list相加其实就是拼接
    # 这里面有些操作应该有更简洁的写法，但是这种属于同一种功能的不同写法，就不改了，浪费时间，大致理解其意思即可
    for col in (GENES + CELLS):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        # 设置train_features
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)  # 可是train_features[col].values本身就是这个形状啊
        transformer.fit(raw_vec)

        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        # 设置test_features
        test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


    print('genes维数', train_features[GENES].shape)
    print('cells维数', train_features[CELLS].shape)

    return train_features, test_features


def plot_features(train_features, test_features, tiltle):
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
    plt.suptitle(tiltle, fontsize=30)
    plt.show()


def dim_reduction(train_features, test_features, gene_comp, cell_comp):
    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    print('对数据进行降维'.center(100, '*'))
    print(f'gene特征降低为{gene_comp}维')
    print(f'cell特征降低为{cell_comp}维')

    # GENES
    # 因子分析
    n_comp = gene_comp  # <--Update # 因子分析的因子数（降低为90列）

    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])  # 纵向拼接（仅是为了统一处理列，后面又分开）
    data2 = (FactorAnalysis(n_components=n_comp, random_state=42).fit_transform(data[GENES]))
    train2 = data2[:train_features.shape[0]];
    test2 = data2[-test_features.shape[0]:]

    # 为新获得的dataframe添加列名
    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

    # 将pca后的特征与原特征横向拼接
    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    # CELLS
    n_comp = cell_comp  # <--Update

    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    data2 = (FactorAnalysis(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))
    train2 = data2[:train_features.shape[0]];
    test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)

    print('降维并拼接原特征后的特征维数', train_features.shape)

    return train_features, test_features


def preprocess2(train_features, test_features):
    print('对拼接后的数据preprocess'.center(100, '*'))
    var_thresh = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")

    data = train_features.append(test_features)  # 纵向拼接
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])  # 仅处理第四列之外的特征

    # 将处理过之后的数据拆分
    train_features_transformed = data_transformed[: train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0]:]

    # 重组train_features
    train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                  columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

    # 重组test_features
    test_features = pd.DataFrame(test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                 columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    print('拼接并处理后的特征维数',train_features.shape)

    return train_features, test_features


def fe_cluster(train, test, n_clusters_g=45, n_clusters_c=15, SEED=123):
    print('根据gene与cell特征对样本进行聚类'.center(100, '*'))
    # 返回的是int值列表
    features_g = list(train.columns[4:776])  # 4:776这个要自己观察dataframe数据去找
    features_c = list(train.columns[776:876])

    def create_cluster(train, test, features, kind='g', n_clusters=n_clusters_g):
        # features是int列表，包含了要进行聚类的gene/cell的列索引
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(
            data)  # 这里为什么要将train与test的数据一同fit？（为了保证类别划分的标准一致？）
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]  # 添加一个新列，这个列标记了样本按gene/cell进行的分类结果
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train,
                               columns=[f'clusters_{kind}'])  # 将类别节点（原来为[0, n_clusters)的整数）转换为哑结点(1,0,0,0……0这种形式)
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(train, test, features_g, kind='g', n_clusters=n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind='c',
                                 n_clusters=n_clusters_c)  # 注意运行create_cluster是对train与test添加新列，并不会覆盖老数据，所以这种写法没关系
    return train, test


def fe_stats(train, test):
    print('dataframe添加新的数据特征的列，删除无用的行'.center(100, '*'))
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])

    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis=1)
        df['g_mean'] = df[features_g].mean(axis=1)
        df['g_std'] = df[features_g].std(axis=1)
        df['g_kurt'] = df[features_g].kurtosis(axis=1)  # 峰度
        df['g_skew'] = df[features_g].skew(axis=1)  # 偏斜度

        df['c_sum'] = df[features_c].sum(axis=1)
        df['c_mean'] = df[features_c].mean(axis=1)
        df['c_std'] = df[features_c].std(axis=1)
        df['c_kurt'] = df[features_c].kurtosis(axis=1)
        df['c_skew'] = df[features_c].skew(axis=1)

        df['gc_sum'] = df[features_g + features_c].sum(axis=1)
        df['gc_mean'] = df[features_g + features_c].mean(axis=1)
        df['gc_std'] = df[features_g + features_c].std(axis=1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis=1)
        df['gc_skew'] = df[features_g + features_c].skew(axis=1)

    return train, test


def preprocess3(train_features, test_features, train_targets_scored):
    print('删除’cp_type’==’ctl_vehicle’的行'.center(100, '*'))
    # 删除’cp_type’==’ctl_vehicle’的行，并重置索引以使得索引连续化 因为这样的行没有moa，也就是对分类没有帮助，所以要删除

    # merge 函数通过一个或多个键将数据集的行连接起来。
    # 场景：针对同一个主键存在的两张包含不同特征的表，通过主键的链接，将两张表进行合并。合并之后，两张表的行数不增加，列数是两张表的列数之和。
    train = train_features.merge(train_targets_scored, on='sig_id')  # 这个拼接的目的是一同丢弃cp_type'=='ctl_vehicle的行（出于代码书写简洁考虑）
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)  # drop=True为重新添加连续索引之后，将老索引删除
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    target = train[train_targets_scored.columns]  # 一同丢弃相应行后，再分离出target（注意这里的train中仍然包含着target）

    # cp_type现在没用了，删除
    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)
    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

    print('train_features describe：')
    print(train.describe())

    return train, test, target, target_cols


def k_fold(train, target, n_splits):
    print(f'计算{n_splits}折交叉验证索引'.center(100, '*'))
    folds = train.copy()

    # MultilabelStratifiedKFold是用于多标签分层的K折交叉验证
    mskf = MultilabelStratifiedKFold(n_splits=n_splits)

    # 不清楚为啥非要用类型转换
    # v_idx返回的是第f+1次k折交叉验证的验证集索引
    # 至于这个x其实对函数计算并没有帮助，只不过根据官网介绍sklearn的非分层抽样要x，分层抽样算法上不用x，但为了兼容，还是要了x，本库为了与sklearn兼容，也要x了
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)  # 添加一个新的列来记录下第f+1轮的验证集索引（与此对应的，第f+1轮，该列数值不是f的样本，即是训练集样本）

    folds['kfold'] = folds['kfold'].astype(int)

    return folds


def main(data_base_path):
    if not os.path.exists(os.path.join(os.getcwd(), 'train_features.pickle')):  # 首次运行
        print('首次运行，首先将读取数据并处理'.center(100, '*'))
        train_features, test_features, train_targets_scored, train_targets_nonscored, sample_submission = get_data(data_base_path)
        plot_features(train_features, test_features, 'features_before_preprocess')

        train_features, test_features = preprocess(train_features, test_features)
        plot_features(train_features, test_features, 'features_after_preprocess')

        seed_everything()

        dim_reduction(train_features, test_features, gene_comp=90, cell_comp=50)

        train_features, test_features = preprocess2(train_features, test_features)

        train_features, test_features = fe_cluster(train_features, test_features)

        train_features, test_features = fe_stats(train_features, test_features)

        train_features, test_features, target, target_cols = preprocess3(train_features, test_features, train_targets_scored)

        folds = k_fold(train_features, target, n_splits=5)

        util.save_as_pickle(os.path.join(os.getcwd(), 'train_features.pickle'), train_features)
        util.save_as_pickle(os.path.join(os.getcwd(), 'test_features.pickle'), test_features)
        util.save_as_pickle(os.path.join(os.getcwd(), 'folds.pickle'), folds)
        util.save_as_pickle(os.path.join(os.getcwd(), 'target.pickle'), target)
        util.save_as_pickle(os.path.join(os.getcwd(), 'target_cols.pickle'), target_cols)
    else:
        print('非首次运行，直接从pickle中读取数据'.center(100, '*'))

        train_features = util.load_from_pickle(os.path.join(os.getcwd(), 'train_features.pickle'))
        test_features = util.load_from_pickle(os.path.join(os.getcwd(), 'test_features.pickle'))
        folds = util.load_from_pickle(os.path.join(os.getcwd(), 'folds.pickle'))
        target = util.load_from_pickle(os.path.join(os.getcwd(), 'target.pickle'))
        target_cols = util.load_from_pickle(os.path.join(os.getcwd(), 'target_cols.pickle'))




if __name__ == '__main__':
    data_base_path = 'E:\AAAMyCodes\myjupyter\kaggle\Mechanisms of Action (MoA) Prediction\data'
    main(data_base_path)



