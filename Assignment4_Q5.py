x=df_train.drop(["id","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19","remaining_cycle","label"],axis=1)
y=df_train['remaining_cycle']
print('x shape : ',x.shape)
print('y shape : ',y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=3)
print('X_train shape : ',x_train.shape)
print('X_test shape : ',x_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)
df_train
x.head()
y
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import xgboost as xgb
import lightgbm as lgb
## RandomForestRegressor

rf_regressor=RandomForestRegressor(n_jobs=-1,n_estimators=400)
rf_regressor.fit(x_train,y_train)
y_pred_rf=rf_regressor.predict(x_test)


mae=mean_absolute_error(y_test,y_pred_rf)
mse=mean_squared_error(y_test,y_pred_rf)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred_rf)
print(f'Model : RandomForestRegressor \n  MAE : {mae}\n  MSE : {mse}\n RMSE: {rmse}\n R2 score: {r2}\n')
df_test
y_test
df_test
y_pred_rf=rf_regressor.predict(df_test.drop(["label","remaining_cycle","rul_failed"],axis=1))
df_test["remaining_cycle"]
df_excel=pd.DataFrame(y_pred_rf,columns=["RUL-predicted"])
df_excel["TRUE-RUL"]=df_test["remaining_cycle"]
df_excel.to_csv("result.csv")
plt.figure(figsize=(12,6))
plt.plot(df_test["remaining_cycle"],label='TRUE RUL')
plt.plot(y_pred_rf,label="Predicted RUL - Random Forest")
plt.xlabel('Index')
plt.ylabel('RUL')
plt.title('True and Predicted RUL')
plt.legend()
plt.show()
