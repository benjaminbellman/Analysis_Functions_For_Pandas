## We start by importing the appropriate packages which we will want to use. 
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

from datetime import timedelta
import seaborn as sns
import warnings
import statsmodels.api as sm

from statsmodels.graphics.api import abline_plot 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model, preprocessing

def get_datetimes(df,date_cols, date_col): 
    '''Add Details to function later'''
    df[date_cols] =df[date_cols].apply(pd.to_datetime)
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    df['Day_of_Week'] = df[date_col].dt.weekday
    df['Year'] = df[date_col].dt.year
    if len(date_cols)>1:
        df['Days_Elapsed'] =  (df[date_cols[1]].dt.day - df[date_cols[0]].dt.day)
    print('Range of Dates in this DataFrame are between {} & {}'.format(df[date_cols[0]].min(),
                                                                        df[date_cols[0]].max()))

def preview_data(df): 
    print('First Five Rows of Data: \n')
    display(df.head())
    print('\n Shape: \n')
    print(df.shape)
    print('\n Info: \n')
    print(df.info())



## Get percentage of missing values for each column
def get_missing_counts(df):
    '''Function that retrieves percentage of missing values in each column of the dataframe'''
    data_dict = dict(df.isna().sum())
    print('Missing Value Percentages by Column: \n')
    for k,v in data_dict.items():
        print('{} -----> {} -----> {}{}'.format(k, ## Gets the name of the column
                                                v,
                                                str(round((v /len(df))*100,2)), ## Retrieves percentage of missing values. 
                                                '%')) 


def fill_missing_values(df,cols_mean,cols_median,cols_mode,cols_zero):
    '''Function that fills missing values with respective strategy to the column. : '''
    ## Fill NAs with mean of respective column. 
    df[cols_mean] = df[cols_mean].apply(lambda x: x.fillna(x.mean())) 

    ## Fill NAs with median of the respective column. 
    df[cols_median] = df[cols_median].apply(lambda x: x.fillna(x.median())) 
    
    ## Fill NAs with mode of the respective column. 
    df[cols_mode] = df[cols_mode].apply(lambda x: x.fillna(x.mode())) 

    ## Fill NAs with zeros. 
    df[cols_zero] = df[cols_zero].apply(lambda x: x.fillna(0)) 


def get_unique_column_count(df):
    ''' Function that returns object columns and their distinct count'''
    print('Unique values in each object column: \n')
    for column in df.select_dtypes('object').columns:
        print('{}{}{}'.format(column,': ',df[column].nunique()))

def get_value_counts(df):
    ''' Function that returns object columns and their distinct count'''
    print('Unique values in each object column: \n')
    for column in df.select_dtypes('object').columns:
        print('{}{}{}{{}}'.format(column,'\n',df[column].value_counts(),'\n'))
        print('\n')

def convert_bool(df): 
    ''' Converts our Should be Boolean Columns to Boolean. 1 indicates yes, 0 is Y or U'''
    cols = ['TotalLossInd','CollisionROFlagYN','dimDRPFlagYN','PaintFlagYN','TowingFlagYN']
    df[cols] = df[cols].apply(lambda x: np.where(x =='Y',1,0))


def get_netamounts(df):
    '''Get Net Amount Aggregates for each car brand'''
    makes = df.groupby('VehicleMake').sum().sort_values('NetAmount', ascending=False).round()
    makes['NetAmount%'] = round(makes['NetAmount'] / makes['NetAmount'].sum() * 100,2)
    makes['NetAmount%total'] = makes['NetAmount%'].cumsum().round(1)
    makes['Margins'] = 1 - (makes['RepairCost'] /makes['NetAmount']).round(2)
    return makes[['NetAmount%','NetAmount%total','NetAmount','Margins']].sort_values('Margins',ascending=False)


def shop_aggregates(df):
    '''Get Net Amounts by Shop'''
    shops = df.groupby('Shop').sum().sort_values('NetAmount', ascending=False).round()
    shops['NetAmount%'] = round(shops['NetAmount'] / shops['NetAmount'].sum() * 100,2)
    shops['NetAmount%total'] = shops['NetAmount%'].cumsum().round(1)
    shops['Margins'] = 1 - (shops['RepairCost'] /shops['NetAmount']).round(2)
    return shops[['NetAmount%','NetAmount%total','NetAmount','Margins']]


def get_repairscosts(df):
    '''Function gets the Repair Costs Distribution by Make'''
    makes = df.groupby('VehicleMake').sum().sort_values('RepairCost', ascending=False).round()
    makes['RepairCost%'] = round(makes['RepairCost'] / makes['RepairCost'].sum() * 100,2)
    makes['RepairCost%total'] = makes['RepairCost%'].cumsum().round(1)
    return makes[['RepairCost%','RepairCost%total','RepairCost']]

def get_model_metrics(names,models):
    counter = 0
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
        print('{} Model Metrics:'.format(names[counter]))
        print('ROC_AUC_SCORE: ' + str(round(roc_auc,3)))
        print('Accuracy: ' + str(round(accuracy_score(y_test,y_pred),3)))
        print('Precision: '+ str(round(precision_score(y_test,y_pred),3)))
        print('Recall: ' + str(round(recall_score(y_test,y_pred),3)))
        print('F1-Score: ' + str(round(f1_score(y_test,y_pred),3)))
        print('Log Loss Score: ' + str(round(log_loss(y_test,y_pred),3)))
        print('MCC: ' + str(round(matthews_corrcoef(y_test,y_pred),3)))
        print("\n")
        counter +=1
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    for model, ax in zip(models, axes.flatten()):
        plot_confusion_matrix(model, 
                                X_test, 
                                y_test, 
                                ax=ax, 
                                cmap='Blues',
                                display_labels=['Non-Repeat','Repeat'])
        ax.title.set_text(type(model).__name__)

    plt.tight_layout()  
    plt.show()