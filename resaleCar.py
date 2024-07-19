import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import numpy as np
from tabulate import tabulate

data = pd.read_csv('car data.csv')

data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}}, inplace=True)
data.replace({'Seller_Type':{'Dealer':0,'Individual':1}}, inplace=True)
data.replace({'Transmission':{'Manual':0,'Automatic':1}}, inplace=True)

X = data.drop(['Car_Name','Selling_Price'], axis=1)
Y = data['Selling_Price']
print(data.head())
null_counts = data.isnull().sum()
total_nulls = null_counts.sum()
print("Total number of null values in the dataset:", total_nulls)

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
all_variance = pca.explained_variance_ratio_

print("PCA VAriance",pca.explained_variance_ratio_)
print(np.sum(all_variance[:1])/(sum(all_variance)))
x = pd.DataFrame(X_pca, columns=['PC1','PC2'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.1, random_state=2)

input_data = {}
input_data['Year'] = int(input("Enter the year of the car: "))
input_data['Present_Price'] = float(input("Enter the present price of the car: "))
input_data['Kms_Driven'] = int(input("Enter the kilometers driven by the car: "))
input_data['Fuel_Type'] = int(input("Enter the fuel type of the car (0 for Petrol, 1 for Diesel, 2 for CNG): "))
input_data['Seller_Type'] = int(input("Enter the seller type of the car (0 for Dealer, 1 for Individual): "))
input_data['Transmission'] = int(input("Enter the transmission type of the car (0 for Manual, 1 for Automatic): "))
input_data['Owner'] = int(input("Enter the number of previous owners of the car: "))
print("\n")
input_df = pd.DataFrame(input_data, index=[0])

input_pca = pca.transform(input_df)

##############################################################################################################

#LINEAR REGRESSION
lin_reg = LinearRegression()

lin_reg.fit(X_train, Y_train)

pred_lin_pca = lin_reg.predict(X_train)

err_lin_pca = metrics.r2_score(Y_train, pred_lin_pca)
print("Accuracy (Linear Regression) with PCA: ", err_lin_pca)
predicted_selling_price_lin_pca = lin_reg.predict(input_pca)
print("Predicted selling price (Linear Regression) with PCA: ", predicted_selling_price_lin_pca)

plt.scatter(Y_train, pred_lin_pca)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression with PCA")
plt.show()

print("\n")
lin_reg.fit(x_train, y_train)
pred_lin= lin_reg.predict(x_train)

err_lin = metrics.r2_score(y_train, pred_lin)
plt.scatter(y_train, pred_lin)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression without PCA")
plt.show()
print("Accuracy (Linear Regression) without PCA: ", err_lin)
predicted_selling_price_lin = lin_reg.predict(input_df)
print("Predicted selling price (Linear Regression) without PCA: ", predicted_selling_price_lin)

print("\n")
print("\n")
###############################################################################################################

# Train KNN model
length=len(x)
sample=int(math.sqrt(length))
if (sample%2==0):
    k=sample+1
    
knn = KNeighborsRegressor(n_neighbors=sample)
knn.fit(X_train, Y_train)

pred_knn_pca=knn.predict(X_test)
acc_knn_pca =metrics.r2_score(Y_test, pred_knn_pca)
print("Accuracy (KNN) with PCA: ", acc_knn_pca)

predicted_selling_price_knn_pca = knn.predict(input_pca)
print("Predicted selling price (KNN) with PCA: ", predicted_selling_price_knn_pca)

plt.scatter(Y_test, pred_knn_pca)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("KNN with PCA")
plt.show()

print("\n")
knn.fit(x_train, y_train)

pred_knn=knn.predict(x_test)
acc_knn = metrics.r2_score(y_test, pred_knn)
print("Accuracy (KNN) without PCA: ", acc_knn)

predicted_selling_price_knn = knn.predict(input_df)
print("Predicted selling price (KNN) without PCA: ", predicted_selling_price_knn)

plt.scatter(Y_test, pred_knn)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("KNN without PCA")
plt.show()

print("\n")
print("\n")

###############################################################################################################

#Train Random Forest
acc_rf_pca=[]

for i in range (1,100):
    rf = RandomForestRegressor(n_estimators=i, random_state=42)
    rf.fit(X_train, Y_train)
    pred_rf_pca = rf.predict(X_test)
    acc = metrics.r2_score(Y_test, pred_rf_pca)
    acc_rf_pca.append(acc)
max_acc_rf_pca = max(acc_rf_pca)
max_index = acc_rf_pca.index(max_acc_rf_pca)

print('Accuracy for RF with PCA:', max_acc_rf_pca, 'with estimators: ',(max_index+1))
predicted_selling_price_rf_pca = rf.predict(input_pca)
print('Predicted Selling Price (Random Forest) with PCA:', predicted_selling_price_rf_pca)

print("\n")

acc_rf=[]

for i in range (1,100):
    rf = RandomForestRegressor(n_estimators=i, random_state=42)
    rf.fit(x_train, y_train)
    pred_rf = rf.predict(x_test)
    acc_no = metrics.r2_score(y_test, pred_rf)
    acc_rf.append(acc_no)
max_acc_rf = max(acc_rf)
max_ind = acc_rf.index(max_acc_rf)

print('Accuracy for RF without PCA:', max_acc_rf, 'with estimators: ',(max_ind+1))

predicted_selling_price_rf = rf.predict(input_df)
print('Predicted Selling Price (Random Forest) without PCA:', predicted_selling_price_rf)

print("\n")
print("\n")
###############################################################################################################

# SVR
svr = SVR(kernel='rbf')  
svr.fit(X_train, Y_train)

pred_svr_pca = svr.predict(X_test)
acc_svr_pca = metrics.r2_score(Y_test, pred_svr_pca)
print('Accuracy for SVR with PCA:', acc_svr_pca)

predicted_selling_price_svr_pca = svr.predict(input_pca)
print("Predicted selling price (SVR) with PCA: ", predicted_selling_price_svr_pca)

plt.scatter(Y_test, pred_svr_pca)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("SVR with PCA")
plt.show()

print("\n")


svr.fit(x_train, y_train)

pred_svr = svr.predict(x_test)
acc_svr = metrics.r2_score(y_test, pred_svr)
print('Accuracy for SVR without PCA:', acc_svr)

predicted_selling_price_svr = svr.predict(input_df)
print("Predicted selling price (SVR) without PCA: ", predicted_selling_price_svr)

plt.scatter(Y_test, pred_svr)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("SVR with PCA")
plt.show()

print("\n")
print("\n")
###############################################################################################################

#Decision Tree

dt = DecisionTreeRegressor()
dt.fit(X_train, Y_train)
pred_dt_pca = dt.predict(X_test)
acc_dt_pca = metrics.r2_score(Y_test, pred_dt_pca)
print("Accuracy for Decision Tree with PCA:", acc_dt_pca)

predicted_selling_price_dt_pca = dt.predict(input_pca)
print("Predicted selling price (Decision Tree) with PCA: ", predicted_selling_price_dt_pca)
print("\n")

dt.fit(x_train, y_train)
pred_dt = dt.predict(x_test)
acc_dt = metrics.r2_score(y_test, pred_dt)
print("Accuracy for Decision Tree with PCA:", acc_dt)

predicted_selling_price_dt = dt.predict(input_df)
print("Predicted selling price (Decision Tree) without PCA: ", predicted_selling_price_dt)

print("\n")
print("\n")

###############################################################################################################

head=['Model','R2 with PCA','R2 without PCA']
mydata=[['Linear Regression',err_lin_pca,err_lin],['K-Nearest Neighbors',acc_knn_pca,acc_knn],['Random Forest',max_acc_rf_pca,max_acc_rf],['SVR',acc_svr_pca,acc_svr],['Decision Tree',acc_dt_pca,acc_dt]]
print(tabulate(mydata,headers=head,tablefmt='grid'))