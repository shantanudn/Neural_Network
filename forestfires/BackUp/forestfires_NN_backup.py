import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


########################## Neural Network for predicting continuous values ###############################

# Reading data 
forestfires = pd.read_csv("C:/Training/Analytics/Neural_Network/forestfires/forestfires.csv")
forestfires.head()

forestfires_ori = forestfires


##Creating a seperate dataframe which has only continuoMarital.Status variables
#forestfires_cat = forestfires[['month','day','size_category']].copy()
#
##Creating a seperate dataframe which has only categorical variables
#forestfires_con = forestfires.drop(['month','day','size_category'],axis=1)

####### Creating dummy varibales for State ########
#forestfires_dummies = pd.get_dummies(forestfires_cat[['size_category']])
#forestfires_dummies.columns
##forestfires_dummies = pd.to_numeric(forestfires_dummies)
#forestfires_dummies = forestfires_dummies.drop(['size_category_small'],axis=1)
#
#forestfires_dummies = int(forestfires_dummies)

####### concatinating or combining the dummy variables column to the startup50 dataset
#forestfires = pd.concat([forestfires,forestfires_dummies], axis=1)

######## dropping the variables ##############
forestfires.columns
forestfires = forestfires.drop(['month','day','size_category'],axis=1)



##### EDA ###################3
forestfires.columns

forestfires.dtypes

# Correlation matrix 
cor_matrix = forestfires.corr()

def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
forestfires = norm_func(forestfires.iloc[:])

print(forestfires.describe())
descriptive = forestfires.describe()

from tabulate import tabulate as tb
print(tb(descriptive,forestfires.columns))

######### boxplots ###########
forestfires.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
#plt.boxplot(forestfires.cement)
#plt.xticks([1,], ['cement'])
#plt.boxplot(forestfires.slag)
#plt.xticks([1,], ['slag'])
#plt.boxplot(forestfires.ash)
#plt.xticks([1,], ['ash'])
#plt.boxplot(forestfires.water)
#plt.xticks([1,], ['water'])
#plt.boxplot(forestfires.superplastic)
#plt.xticks([1,], ['superplastic'])
#plt.boxplot(forestfires.coarseagg)
#plt.xticks([1,], ['coarseagg'])
#plt.boxplot(forestfires.fineagg)
#plt.xticks([1,], ['fineagg'])
#plt.boxplot(forestfires.age)
#plt.xticks([1,], ['age'])
#plt.boxplot(forestfires.strength)
#plt.xticks([1,], ['strength'])

########## Histogram ###########
plt.hist((forestfires.area))
forestfires.hist()
forestfires.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)

#plt.hist(forestfires.cement)
#plt.xlabel('cement')
#plt.hist(forestfires.slag)
#plt.xlabel('slag')
#plt.hist(forestfires.ash)
#plt.xlabel('ash')
#plt.hist(forestfires.water)
#plt.xlabel('water')
#plt.hist(forestfires.superplastic)
#plt.xlabel('superplastic')
#plt.hist(forestfires.coarseagg)
#plt.xlabel('coarseagg')
#plt.hist(forestfires.fineagg)
#plt.xlabel('fineagg')
#plt.hist(forestfires.age)
#plt.xlabel('age')
#plt.hist(forestfires.strength)
#plt.xlabel('strength')
#
##Scatter Plots
#plt.plot(forestfires.slag,forestfires.cement,"ro");plt.xlabel("slag");plt.ylabel("cement")
#plt.plot(forestfires.ash,forestfires.cement,"ro");plt.xlabel("ash");plt.ylabel("cement")
#plt.plot(forestfires.water,forestfires.cement,"ro");plt.xlabel("water");plt.ylabel("cement")
#plt.plot(forestfires.superplastic,forestfires.cement,"ro");plt.xlabel("superplastic");plt.ylabel("cement")
#plt.plot(forestfires.coarseagg,forestfires.cement,"ro");plt.xlabel("coarseagg");plt.ylabel("cement")
#plt.plot(forestfires.fineagg,forestfires.cement,"ro");plt.xlabel("fineagg");plt.ylabel("cement")
#plt.plot(forestfires.age,forestfires.cement,"ro");plt.xlabel("age");plt.ylabel("cement")
#plt.plot(forestfires.strength,forestfires.cement,"ro");plt.xlabel("strength");plt.ylabel("cement")


# Correlation matrix 
forestfires.corr()
cor_matrix = forestfires.corr()
plt.matshow(forestfires.corr())
plt.show()


# getting boxplot of cement with respect to each category of fineagg 
import seaborn as sns 


heat1 = forestfires.corr()
sns.heatmap(heat1, xticklabels=forestfires.columns, yticklabels=forestfires.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(forestfires)


# columns names
forestfires.columns


# =============================================================================
# #Preparing NN model
# =============================================================================

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["asuperplasticuracy"])
    return (model)

forestfires = forestfires.iloc[:,[8,0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]

column_names = list(forestfires.columns)
predictors = column_names[1:]
target = column_names[0]

first_model = prep_model([28,50,1])
first_model.fit(np.array(forestfires[predictors]),np.array(forestfires[target]),epochs=900)
pred_train = first_model.predict(np.array(forestfires[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-forestfires[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,forestfires[target],"bo")
np.corrcoef(pred_train,forestfires[target]) # we got high correlation 


# =============================================================================
# MLP Regressor
# =============================================================================


from sklearn.neural_network import MLPRegressor

train,test = train_test_split(forestfires,test_size = 0.3,random_state=42)
trainX = train.drop(["area"],axis=1)
trainY = train["area"]
testX = test.drop(["area"],axis=1)
testY = test["area"]

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
