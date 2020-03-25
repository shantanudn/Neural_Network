# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
startup50 = pd.read_csv("D:/Training/ExcelR_2/Multi_Linear_Regression/50_Startups/50_Startups.csv")

# to get top 40 rows
startup50.head(40) 


##### EDA ###################
startup50.columns

startup50.dtypes

# Correlation matrix 
startup50.corr()

np.mean(startup50)
startup50['Profit'].mean() 
startup50['Profit'].median()
startup50['Profit'].mode()
startup50['Profit'].var()
startup50['Profit'].std()

print(startup50.describe())
descriptive = startup50.describe()

from tabulate import tabulate as tb
print(tb(descriptive,startup50.columns))

######### boxplots ###########

import seaborn as sns

plt.boxplot(startup50.Profit)
plt.xticks([1,], ['Profit'])
plt.boxplot(startup50.RandDSpend)
plt.xticks([1,], ['R&D Spend'])
plt.boxplot(startup50.Administration)
plt.xticks([1,], ['Administration'])
plt.boxplot(startup50.MarketingSpend)
plt.xticks([1,], ['MarketingSpend'])
plt.boxplot(startup50.State)
plt.xticks([1,], ['State'])


######### Histogram ###########
plt.hist(startup50.Profit)
plt.xlabel('Profit')
plt.hist(startup50.RandDSpend)
plt.xlabel('RandDSpend')
plt.hist(startup50.Administration)
plt.xlabel('Administration')
plt.hist(startup50.MarketingSpend)
plt.xlabel('MarketingSpend')
plt.hist(startup50.State)
plt.xlabel('State')

#Scatter Plots

plt.plot(startup50.RandDSpend,startup50.Profit,"ro");plt.xlabel("RandDSpend");plt.ylabel("Profit")
plt.plot(startup50.Administration,startup50.Profit,"ro");plt.xlabel("Administration");plt.ylabel("Profit")
plt.plot(startup50.MarketingSpend,startup50.Profit,"ro");plt.xlabel("MarketingSpend");plt.ylabel("Profit")
plt.plot(startup50.State,startup50.Profit,"ro");plt.xlabel("State");plt.ylabel("Profit")


# Correlation matrix 
startup50.corr()

plt.matshow(startup50.corr())
plt.show()


###### Creating dummy varibales for State ########
state_dummies = pd.get_dummies(startup50.State)

####### concatinating or combining the dummy variables column to the startup50 dataset
startup50 = pd.concat([startup50,state_dummies], axis=1)

######## dropping the variables "State" and "New York" ##############
startup50 = startup50.drop(["State","New York"],axis=1)

# getting boxplot of Profit with respect to each category of gears 
import seaborn as sns 
sns.boxplot(x="RandDSpend",y="Profit",data=startup50)

heat1 = startup50.corr()
sns.heatmap(heat1, xticklabels=startup50.columns, yticklabels=startup50.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(startup50)


# columns names
startup50.columns

# pd.tools.plotting.scatter_matrix(startup50); -> also used for plotting all in one graph
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Profit~RandDSpend+Administration+MarketingSpend+California+Florida',data=startup50).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()

#The variables States, Administration and MarketingSpend have p-values greater than 0.05, thus these variables insignificant to the outcome


# Preparing model based only on MarketingSpend
ml_MarketingSpend=smf.ols('Profit~MarketingSpend',data = startup50).fit()  
ml_MarketingSpend.summary() 

# Preparing model based only on Administration
ml_Administration=smf.ols('Profit~Administration',data = startup50).fit()  
ml_Administration.summary() 

# Preparing model based only on the states
ml_states=smf.ols('Profit~California+Florida',data = startup50).fit()  
ml_states.summary() 

# Preparing model based only on R&D Spend
ml_RandDSpend=smf.ols('Profit~RandDSpend',data = startup50).fit()  
ml_RandDSpend.summary() 


# Preparing model based only on RandDSpend & MarketingSpend
ml_RandDSpend_MarketingSpend=smf.ols('Profit~RandDSpend+MarketingSpend',data = startup50).fit()  
ml_RandDSpend_MarketingSpend.summary() 

# Preparing model based only on Administration & MarketingSpend
ml_Administration_MarketingSpend=smf.ols('Profit~Administration+MarketingSpend',data = startup50).fit()  
ml_Administration_MarketingSpend.summary() 

# Preparing model based only on Administration & RandDSpend
ml_Administration_RandDSpend=smf.ols('Profit~Administration+RandDSpend',data = startup50).fit()  
ml_Administration_RandDSpend.summary() 



#Dropping variables "California", "Florida" and "Administration" since they are insignificant to the outcome
ml1_v3 = smf.ols('Profit~RandDSpend+MarketingSpend',data = startup50).fit()  # regression model
ml1_v3.summary()
ml1_v3.params

#The model ml1_v3 produces a model of R-sqaured value of 0.950

#Implementing the above model in the prediction model of regression

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1_v3)

# data in rows 19, 45,46 & 49  are found to be influential to the dataframe, hence we drop these variable to improve the model

# Delete the rows with labels 47, 48,19 & 49 
startup50 = startup50.drop([19,47,48,49], axis=0)


# final model
final_model = smf.ols('Profit~RandDSpend+MarketingSpend',data=startup50).fit() 

sm.graphics.influence_plot(final_model)

final_model.params
final_model.summary() 
Profit_pred = final_model.predict(startup50)


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) #
plt.xticks([1,], ['Residuals'])
#Transformation models 

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(startup50.Profit,Profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(Profit_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) #


#Transformation models 

############################################## Log Transformation ######################

ml1_log = smf.ols('Profit~np.log(RandDSpend)+np.log(MarketingSpend)',data=startup50).fit()
ml1_log.summary()


ml1_log.params
profit_pred_log = ml1_log.predict(startup50)


# histogram
plt.hist(ml1_log.resid_pearson) #

###### Log Linearity #########
# Observed values VS Fitted values
plt.scatter(startup50.Profit,profit_pred_log,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(profit_pred_log,ml1_log.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



############################################## Exponential Model ####################

exp_model = smf.ols('np.log(Profit)~RandDSpend+MarketingSpend',data=startup50).fit() 
exp_model.params
exp_model.summary() 


print(exp_model.conf_int(0.05)) # 95% confidence level
pred_log = exp_model.predict(startup50)
pred_log
exp_pred=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
exp_pred
exp_pred.corr(startup50.Profit)
resid_3 = exp_pred-startup50.Profit


######  Linearity for Exponential model #########
# Observed values VS Fitted values
plt.scatter(startup50.Profit,exp_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(exp_pred,exp_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals in Exponential model ######
# histogram
plt.hist(exp_model.resid_pearson) #





### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
startup50_train,startup50_test  = train_test_split(startup50,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols('Profit~RandDSpend+MarketingSpend',data=startup50).fit() 

# train_data prediction
train_pred = model_train.predict(startup50_train)

# train residual values 
train_resid  = train_pred - startup50_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(startup50_test)

# test residual values 
test_resid  = test_pred - startup50_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
