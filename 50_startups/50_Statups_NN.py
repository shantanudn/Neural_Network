import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# loading the data
startup50 = pd.read_csv("C:/Training/Analytics/Neural_Network/50_startups/50_Startups.csv")
startup50_ori = startup50
startup50 = startup50.rename(columns={"R&D Spend": "RandDSpend", "Marketing Spend": "MarketingSpend"})

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

sns.boxplot(x="RandDSpend",y="Profit",data=startup50)

heat1 = startup50.corr()
sns.heatmap(heat1, xticklabels=startup50.columns, yticklabels=startup50.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(startup50)


# columns names
startup50.columns

########################## Neural Network for predicting continuous values ###############################

## Reading data 
#startup50 = pd.read_csv("C:/Training/Analytics/Neural_Network/Solved/startup50.csv")
#startup50.head()
#startup50_back = startup50

from keras.utils import plot_model

startup50 = startup50.iloc[:,[3,0,1,2,4,5]]


def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(startup50.iloc[:,0:])


def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

column_names = list(startup50.columns)
predictors = column_names[1:6]
target = column_names[0]

first_model = prep_model([5,50,1])
first_model.fit(np.array(startup50[predictors]),np.array(startup50[target]),epochs=900)
pred_train = first_model.predict(np.array(startup50[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startup50[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,startup50[target],"bo")
np.corrcoef(pred_train,startup50[target]) # we got high correlation 



# =============================================================================
# MLP Regressor
# =============================================================================


from sklearn.neural_network import MLPRegressor

train,test = train_test_split(startup50,test_size = 0.3,random_state=42)
trainX = train.drop(["Profit"],axis=1)
trainY = train["Profit"]
testX = test.drop(["Profit"],axis=1)
testY = test["Profit"]

mlp = MLPRegressor(hidden_layer_sizes=(5,26))

mlp.fit(trainX,trainY)
prediction_train=mlp.predict(trainX)
prediction_test = mlp.predict(testX)

rmse_value = np.sqrt(np.mean((prediction_train-trainY)**2))
plt.plot(prediction_train,trainY,"bo")
np.corrcoef(prediction_train,trainY) 


rmse_value_test = np.sqrt(np.mean((prediction_test-testY)**2))
plt.plot(prediction_test,testY,"bo")
np.corrcoef(prediction_test,testY) 
