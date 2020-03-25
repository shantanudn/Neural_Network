import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


########################## Neural Network for predicting continuous values ###############################

# Reading data 
Concrete = pd.read_csv("C:/Training/Analytics/Neural_Network/Solved/concrete.csv")
Concrete.head()

##### EDA ###################3
Concrete.columns

# Correlation matrix 
cor_matrix = Concrete.corr()


print(Concrete.describe())
descriptive = Concrete.describe()

from tabulate import tabulate as tb
print(tb(descriptive,Concrete.columns))

######### boxplots ###########

plt.boxplot(Concrete.cement)
plt.xticks([1,], ['cement'])
plt.boxplot(Concrete.slag)
plt.xticks([1,], ['slag'])
plt.boxplot(Concrete.ash)
plt.xticks([1,], ['ash'])
plt.boxplot(Concrete.water)
plt.xticks([1,], ['water'])
plt.boxplot(Concrete.superplastic)
plt.xticks([1,], ['superplastic'])
plt.boxplot(Concrete.coarseagg)
plt.xticks([1,], ['coarseagg'])
plt.boxplot(Concrete.fineagg)
plt.xticks([1,], ['fineagg'])
plt.boxplot(Concrete.age)
plt.xticks([1,], ['age'])
plt.boxplot(Concrete.strength)
plt.xticks([1,], ['strength'])

######### Histogram ###########
plt.hist(Concrete.cement)
plt.xlabel('cement')
plt.hist(Concrete.slag)
plt.xlabel('slag')
plt.hist(Concrete.ash)
plt.xlabel('ash')
plt.hist(Concrete.water)
plt.xlabel('water')
plt.hist(Concrete.superplastic)
plt.xlabel('superplastic')
plt.hist(Concrete.coarseagg)
plt.xlabel('coarseagg')
plt.hist(Concrete.fineagg)
plt.xlabel('fineagg')
plt.hist(Concrete.age)
plt.xlabel('age')
plt.hist(Concrete.strength)
plt.xlabel('strength')

#Scatter Plots
plt.plot(Concrete.slag,Concrete.cement,"ro");plt.xlabel("slag");plt.ylabel("cement")
plt.plot(Concrete.ash,Concrete.cement,"ro");plt.xlabel("ash");plt.ylabel("cement")
plt.plot(Concrete.water,Concrete.cement,"ro");plt.xlabel("water");plt.ylabel("cement")
plt.plot(Concrete.superplastic,Concrete.cement,"ro");plt.xlabel("superplastic");plt.ylabel("cement")
plt.plot(Concrete.coarseagg,Concrete.cement,"ro");plt.xlabel("coarseagg");plt.ylabel("cement")
plt.plot(Concrete.fineagg,Concrete.cement,"ro");plt.xlabel("fineagg");plt.ylabel("cement")
plt.plot(Concrete.age,Concrete.cement,"ro");plt.xlabel("age");plt.ylabel("cement")
plt.plot(Concrete.strength,Concrete.cement,"ro");plt.xlabel("strength");plt.ylabel("cement")


# Correlation matrix 
Concrete.corr()

plt.matshow(Concrete.corr())
plt.show()


# getting boxplot of cement with respect to each category of fineagg 
import seaborn as sns 
sns.boxplot(x="slag",y="cement",data=Concrete)

heat1 = Concrete.corr()
sns.heatmap(heat1, xticklabels=Concrete.columns, yticklabels=Concrete.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(Concrete)


# columns names
Concrete.columns


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

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

first_model = prep_model([8,50,1])
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=900)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target]) # we got high correlation 
