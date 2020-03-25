

import numpy
import pandas as pd

#from sklearn.feature_selection import RFE

import seaborn as sns

import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


#from sklearn.preprocessing import MinMaxScaler
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet
#from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.svm import SVR
#from sklearn.metrics import explained_variance_score
#from sklearn.metrics import mean_absolute_error



from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
#from keras.utils import np_utils
#from keras.constraints import maxnorm
##from sklearn.preprocessing import MinMaxScaler
##from sklearn.metrics import mean_squared_error
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)




########################## Neural Network for predicting continuous values ###############################

# Reading data 
forestfires = pd.read_csv("C:/Training/Analytics/Neural_Network/forestfires/forestfires.csv")
forestfires.head()

forestfires_ori = forestfires

forestfires = forestfires[['month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']].copy()
forestfires.columns


# Encode Data
forestfires.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
forestfires.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

##### EDA ###################



print("Head:", forestfires.head())



describe =  forestfires.describe()

forestfires.columns

forestfires.dtypes



print("Shape:", forestfires.shape)

print("Data Types:", forestfires.dtypes)

print("Correlation:", forestfires.corr(method='pearson'))



dataset = forestfires.values


X = dataset[:,0:11]
Y = dataset[:,10]


#
#
##Feature Selection
#model = ExtraTreesRegressor()
#rfe = RFE(model, 3)
#fit = rfe.fit(X, Y)
#
#print("Number of Features: ", fit.n_features_)
#print("Selected Features: ", fit.support_)
#print("Feature Ranking: ", fit.ranking_) 
#
#

plt.hist((forestfires.area))


# =============================================================================
# Histogram
# =============================================================================
forestfires.hist()


# =============================================================================
# Continous distribution funtion
# =============================================================================
forestfires.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)



forestfires.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)



scatter_matrix(forestfires)

# =============================================================================
# Corrlation Heat Map
# =============================================================================

heat1 = forestfires.corr()
sns.heatmap(heat1, xticklabels=forestfires.columns, yticklabels=forestfires.columns, annot=True)


# =============================================================================
# Heat map without values
# =============================================================================

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(forestfires.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,13,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(forestfires.columns)
ax.set_yticklabels(forestfires.columns)

num_instances = len(X)
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
