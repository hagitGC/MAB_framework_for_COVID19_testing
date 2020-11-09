import pandas as pd
import numpy as np
import sys
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

filename = 'C:/Users/hagit/pycharmprojects/covid_gov_data/corona_tested_individuals_ver_009.csv'

df = pd.read_csv(filename, low_memory=False )

headers = [ 'test_date', 'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'corona_result', 'age_60_and_above', 'gender', 'test_indication']

df.info()
categories = df['test_indication'].unique()

def data_cleaning(df1):
    df1 = df1.replace('None', 0.)
    df1 = df1.replace("NULL", 0.)
    df1 = df1.replace(np.nan, 0.)
    df1['gender'] = df1['gender'].replace("נקבה", 1.)
    df1['gender'] = df1['gender'].replace("זכר", 0.)
    df1['corona_result'] = df1['corona_result'].replace('חיובי', 1.)
    df1['corona_result'] = df1['corona_result'].replace('שלילי', -1.) 
    df1['corona_result'] = df1['corona_result'].replace('אחר',  np.nan)
    df1['age_60_and_above'] = df1['age_60_and_above'].replace('Yes', 1.)
    df1['age_60_and_above'] = df1['age_60_and_above'].replace('No', 0.) # 0)
    df1['test_date_date'] = pd.to_datetime(df1['test_date'])
    df1['Week_Number'] = df1['test_date_date'].dt.strftime('%U')
    return df1


def create_dummies_for_test_indication(df1, categories):
    df_test_indication = pd.get_dummies(df1['test_indication'], columns=categories)
    newdf = pd.concat([df1, df_test_indication.reindex(df1.index)], axis=1)
    data = newdf.drop(['test_indication',  'age_60_and_above', 'test_date'], axis=1)  #'_id','test_date',
    return data




def data_preprocess(data):
    data = data.drop(['test_date_date', 'Week_Number'], axis=1)
    X = data.loc[:, data.columns != 'corona_result']
    y = data.loc[:, data.columns == 'corona_result']
    floatX = []
    for row in X.values:
        temp = [float(obj) for obj in row]
        floatX.append(temp)
    floaty = np.asarray([item[0] for item in y.values])
    return data, floatX, floaty



data1 = data_cleaning(df)
data1['gender'] = pd.to_numeric(data1['gender'])
data2 = create_dummies_for_test_indication(data1, categories)


nonull_df = data2.dropna(axis=0)  

my_dict = dict()
for col in nonull_df.keys().values:
    my_dict[col] = []
weeks = []

train_weeks = ['10', '11', '12']
test_weeks = ['13', '14', '15', '16']

''''
train model 
'''
print ('training model with weeks {}'.format(train_weeks))
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF




train_weeks_df = nonull_df[nonull_df['Week_Number'].isin(train_weeks)]
train_data, trainX, trainy = data_preprocess(train_weeks_df)
from sklearn import linear_model
clf = SVC(kernel = 'poly', gamma='auto', probability=True)
clf.fit(trainX, trainy)
y_hat_train = clf.predict(trainX)
probas = clf.predict_proba(trainX)
pos_probas = [x[1] for x in probas]




to_file_precision = r'C:/Users/hagit/pycharmprojects/covid_gov_data/figs/svm_kern_precision_per_week.txt'
file_P = open(to_file_precision,"w+")
to_file_recall = r'C:/Users/hagit/pycharmprojects/covid_gov_data/figs/svm_kern_recall_per_week.txt'
file_r = open(to_file_recall,"w+")
'''
eval
'''

file_P.write('week, positives @ total, possitive @1000, possitive @2000, possitive @3000, possitive @4000, possitive @5000, num of tests \n')
file_r.write('week, recall @1000,  recall @2000, recall @3000, recall @4000, recall @5000 \n')
for week in test_weeks:
    test_week_df = nonull_df[nonull_df['Week_Number'] == week]
    test_data, testX, week_y = data_preprocess(test_week_df)
    probas  = clf.predict_proba(testX) 
    pos_probas = [x[1] for x in probas]
    positives = defaultdict(int)

    week_positives = [x for x in week_y if x==1.0]
    week_positives_precision = len(week_positives)/len(week_y)
    results1 = np.column_stack((pos_probas, week_y))
    sorted_r1 = sorted(results1, key=lambda x: -x[0])
    file_r.write("{}, ".format(week))
    file_P.write("{}, ".format(week))
    file_P.write("{}, ".format(week_positives_precision))
    sizes_list =  np.arange(999, 5999, 1000).tolist()
    for i in sizes_list:
        week__r1_positives_at_i = [x for x in sorted_r1[0:i] if x[1]==1.0]
        week_positives_precision_r1_at_i = len(week__r1_positives_at_i)/(i+1)
        w_recall_at_i = len(week__r1_positives_at_i)/ len(week_positives)
        file_r.write("{}, ".format(w_recall_at_i))
        file_P.write("{}, ".format(week_positives_precision_r1_at_i))
    file_P.write("{}, ".format(len(week_y)))

    file_P.write('\n')
    file_r.write('\n')
file_P.close()
file_r.close()

