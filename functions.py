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


def compare_regression_models(models):
    for i in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        errors = abs(np.array(y_test) - y_pred)
        r2 = round(r2_score(y_test,y_pred),3)
        MSE = round(mse(y_test,y_pred),3)
        RMSE = round(mse(y_test, y_pred, squared =False),3)
        MAE = round(mae(y_test, y_pred),3)
        print('{} Performance: '.format(model))
        print('R Squared : ' + str(r2))
        print('MSE: '+ str(MSE))
        print('RMSE: ' +str(RMSE))
        print('MAE: ' +str(MAE))
        print(('Average Errors: ' + str(np.mean(errors))))
        print("\n")
        return RMSE


def scale_columns(cols_to_scale):
    '''Input a list of columns to scale'''
    scaler = StandardScaler()
    scaler.fit(X[cols_to_scale])
    X_scaled = scaler.transform(X[cols_to_scale])
    for ind, col in enumerate(cols_to_scale):
        X[col] = X_scaled[:,ind]

def plot_roc_curves(names,models):
    linestyles =['-',':','--',':','-','--',':','-','-']
    colors = ['r','m','dodgerblue','g','darkorange','limegreen', 'deeppink','navy','y']

    plt.figure(figsize=(15,10))
    counter = 0
    for name,clf in zip(names,models):
        clf.fit(X_train,y_train)
        y_proba = clf.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=name+ ' (auc: %0.3f)' %roc_auc_score(y_test, clf.predict_proba(X_test)[:,1],average='macro'),
             linestyle=linestyles[counter], c=colors[counter])
        counter += 1
    lims = [np.min([0.0, 0.0]),  np.max([1.0, 1.0])]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0, c='black', linestyle ='--')

    plt.xlabel('False positive rate -->', fontsize = 15)
    plt.ylabel('True positive rate -->', fontsize = 15)
    plt.title('ROC curve for the different Classification Models', pad =15, fontsize = 25)
    plt.legend(loc='best')
    plt.annotate('<--- No skill (0.5)',xy=(0.6,0.55))
    plt.show()

def plot_precision_recall_curve(names,models):
    '''Function plots the precision-recall curve.'''
    linestyles =['-',':','--',':','-','--',':','-','-']
    colors = ['r','m','dodgerblue','g','darkorange','limegreen', 'deeppink','navy','y']

    plt.figure(figsize=(15,10))

    ## Plotting the no-skill line. 
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill', c='black')

    counter = 0
    for name, model in zip(names,models):
        precision,recall,thresholds = precision_recall_curve(y_test,model.predict_proba(X_test)[:,1])
        plt.plot(recall,precision,color=colors[counter], 
                linestyle=linestyles[counter], 
                label = name  +' AUC: %.3f'%auc(recall,precision))
        counter +=1


    plt.title('Precision Recall Tradeoff', pad=20, fontsize = 20)
    plt.xlabel('Recall', fontsize = 15)
    plt.ylabel('Precision', fontsize = 15)
    plt.annotate('No Skill Line: '+str(round(no_skill,3)),(0.2,0.28), fontsize =10)
    plt.legend()
    plt.savefig('precision_recall_4_models.png')
    plt.show()