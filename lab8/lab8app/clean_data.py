import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# load DataFrame
df = pd.read_csv("../../data_lab_2/restaurant_data.csv")

y_ =  df["Revenue"]
features = df.columns[1:-1]
x = df[features]
x_ = pd.get_dummies(x)

x_enc,x_enc_test,y,y_test = train_test_split(x_,y_,train_size=0.8,random_state=42)


