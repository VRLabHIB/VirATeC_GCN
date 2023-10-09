import matplotlib
import numpy as np
import pandas as pd
import os
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import shap
import seaborn as sns
from pathlib import Path

import os
# from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from utils.S4_data_visualization import S401_visualize_features as vis

def GBDT(source, target, n_iter=50, n_estimators=100):
    # method could also be binary
    print('Target variable', target.columns.tolist())
    print('Source variables', source.columns.tolist())
    target = target.values.ravel()

    sc_X = StandardScaler()
    X = sc_X.fit_transform(source)

    y = np.where(target == 1, int(1), int(-1))

    acc_lst = list()
    f1_lst = list()
    quickprec_lst = list()
    slowprec_lst = list()

    best = 0

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    shap_values = np.zeros(X_test.shape)

    for i in range(n_iter):
        print(i)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

        gbm = ensemble.GradientBoostingClassifier(n_estimators=n_estimators)
        gbm.fit(X_train, Y_train)

        prediction = gbm.predict(X_test)
        accuracy = accuracy_score(Y_test, prediction)
        acc_lst.append(accuracy)

        f1 = f1_score(Y_test, prediction, labels=["2D", "3D"])
        f1_lst.append(f1)

        qprecision = precision_score(Y_test, prediction, pos_label=-1)
        quickprec_lst.append(qprecision)
        sprecision = precision_score(Y_test, prediction, pos_label=1)
        slowprec_lst.append(sprecision)

        shap_values = shap_values + shap.TreeExplainer(gbm).shap_values(X_test)

        if accuracy > best:
            gbmb = gbm
            Y_testb = Y_test
            predictionb = prediction
            X_testb = X_test
            best = accuracy

    shap_value = shap.TreeExplainer(gbmb).shap_values(X_testb)
    shap_values = shap_values / n_iter
    contingency = ((np.abs(Y_testb - predictionb) * 0.5) + 1) % 2

    return [X_testb, Y_testb, predictionb, best, acc_lst, shap_values, quickprec_lst, slowprec_lst, f1_lst, contingency,
            shap_value]


def model_writer(results_path, model):
    os.chdir(results_path)

    with open(results_path + '/model.npy', 'wb') as f:
        np.save(f, np.asarray(model[0]))  # X_test
        np.save(f, np.asarray(model[1]))  # Y_test
        np.save(f, np.asarray(model[2]))  # prediction
        np.save(f, np.asarray([model[3]]))  # best_acc
        np.save(f, np.asarray(model[4]))  # acc_lst
        np.save(f, np.asarray(model[5]))  # shap_values
        np.save(f, np.asarray(model[6]))  # quickprec_lst
        np.save(f, np.asarray(model[7]))  # slowprec_lst
        np.save(f, np.asarray(model[8]))  # f1s
        np.save(f, np.asarray(model[9]))  # contingency
        np.save(f, np.asarray(model[10]))  # shap_value


def model_reader(results_path):
    os.chdir(results_path)
    with open(results_path + '/model.npy', 'rb') as f:
        X_test = np.load(f)
        Y_test = np.load(f)
        prediction = np.load(f)
        best_acc = np.load(f)
        acc_lst = np.load(f)
        shap_values = np.load(f)
        quickprec_lst = np.load(f)
        slowprec_lst = np.load(f)
        f1s = np.load(f)
        contingency = np.load(f)
        shap_value = np.load(f)

    return [X_test, Y_test, prediction, best_acc, acc_lst, shap_values, quickprec_lst, slowprec_lst, f1s, contingency,
            shap_value]


if __name__ == '__main__':
    project_path = os.path.abspath(os.getcwd()).rsplit('\\', 2)[0]
    save_path = project_path + '\\data\\nodes_and_transitions\\'
    result_path = project_path + '\\data\\results\\'

    df_max, df_min, df_mean, df_std = vis.load_statistic_dataframes(project_path)

    df = df_mean.copy()
    df = df.dropna(axis='index', how='any', ignore_index=True)
    print(len(df))

    import random

    random.seed(10092022)

    df = df.reset_index(drop=True)
    source = df.iloc[:, 4:]
    source = source.drop(columns=['seating_row_aoi', 'seating_loc_aoi', 'active_disruption', 'passive_disruption'])
    target = df[['Complexity']].astype(int)

    model = GBDT(source, target, n_iter=100, n_estimators=100)
    model_writer(result_path, model)

    model = model_reader(result_path)

    columns = df.columns[7:-3]
    X_test = model[0]
    Y_test = model[1]
    prediction = model[2]
    best_acc = model[3]
    acc_lst = model[4]
    shap_values = model[5]
    quickprec_lst = model[6]
    slowprec_lst = model[7]
    f1s = model[8]
    contingency = list(map(bool, model[9]))
    shap_value = model[10]

    print('Mean Acc: ', np.round(np.nanmean(acc_lst), 3), 'SD: ', np.round(np.nanstd(acc_lst), 3))
    print('Best Model accuracy ', np.round(best_acc, 3))
    print('Quick Precision: {}'.format(np.round(np.nanmean(quickprec_lst), 3)))
    print('Slow Precision: {}'.format(np.round(np.nanmean(slowprec_lst), 3)))

    # Plot results
    plt.rcParams.update({'font.size': 30})
    font = {'size': 30}
    matplotlib.rc('font', **font)

    # fig = plt.figure(figsize=(20, 15))
    # mat = confusion_matrix(Y_test, prediction)
    # sns.heatmap(data=mat.T, square=True, annot=True, fmt='d', cbar=False,
    #            xticklabels=["2D", "3D"],
    #            yticklabels=["2D", "3D"])
    # plt.xlabel('true label',fontsize=30)
    # plt.ylabel('predicted label',fontsize=30)
    # plt.plot()
    # plt.savefig(result_path + 'confusion_matrix.jpg', dpi=500)  # bbox_inches='tight
    # plt.show()

    fig = plt.figure(figsize=(12, 8))
    shap.initjs()
    shap.summary_plot(shap_value, X_test, feature_names=columns, plot_type='dot', plot_size=None, show=False)
    # w, _ = plt.gcf().get_size_inches()
    # plt.gcf().set_size_inches(w, w * 2.5 / 4)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=15)
    # cbar.ax.tick_params(direction='out', length=6, width=2, grid_alpha=0.5)

    # ax_list = shap_figure.axes  # see https://stackoverflow.com/a/24107230/11148296
    # ax = ax_list[0]
    # ax.set_xlabel('local shap values (left side= 2D; right side = 3D)', fontsize=20)

    # plt.tight_layout()
    plt.show()
    plt.savefig(result_path + 'shap_summary.jpg', dpi=100)  # bbox_inches='tight
