import os
os.getcwd()
os.chdir("C:/Users/Albin/Documents/GitHub/Telecom-customer-churn-prediction")
os.getcwd()
import pandas as pd
data1=pd.read_csv("Telecom_Data (1).csv")

#anova(continous & categorial)
import pingouin as pp
pp.welch_anova(data1,dv="account length",between="churn")
pp.welch_anova(data1,dv="area code",between="churn")
pp.welch_anova(data1,dv="total day minutes",between="churn")  
pp.welch_anova(data1,dv="total day calls",between="churn")
pp.welch_anova(data1,dv="total day charge",between="churn")
pp.welch_anova(data1,dv="total eve minutes",between="churn")
pp.welch_anova(data1,dv="total eve calls",between="churn")
pp.welch_anova(data1,dv="total eve charge",between="churn")
pp.welch_anova(data1,dv="total night minutes",between="churn")
pp.welch_anova(data1,dv="total night calls",between="churn")
pp.welch_anova(data1,dv="total night charge",between="churn")
pp.welch_anova(data1,dv="number vmail messages",between="churn")
pp.welch_anova(data1,dv="total intl minutes",between="churn")
pp.welch_anova(data1,dv="total intl calls",between="churn")
pp.welch_anova(data1,dv="total intl charge",between="churn")
data1["customer service calls"].unique()
pp.welch_anova(data1,dv="customer service calls",between="churn")

#drop columns with p>0.05
data2=data1.drop(["account length","area code","total day calls","total eve calls","total night calls","phone number"],axis=1)

#chisquare(categorical & categorical)
import scipy.stats as stats
compare=pd.crosstab(data2["state"],data2["churn"])
stats.chi2_contingency(compare)
stats.chi2_contingency(pd.crosstab(data2["international plan"],data2["churn"]))
stats.chi2_contingency(pd.crosstab(data2["voice mail plan"],data2["churn"]))

data2.churn.replace({True: 1,False: 0}, inplace=True)
data2=pd.get_dummies(data2,columns=['international plan','voice mail plan'])

#separate into dependent & independent 
y=data2["churn"]
x=data2.drop(["churn","state"],axis=1)

#train test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#training model
from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#finding accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
confusion_matrix(y_test, y_pred)
