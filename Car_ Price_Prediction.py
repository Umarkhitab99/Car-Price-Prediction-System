import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = {
     "Year": [2015, 2016, 2017, 2018, 2019, 2020,2021],
     "KM_Driven": [50000, 40000,35000,30000,20000,15000,10000],
     "Price": [10000,12000,15000,17000,20000,22000,25000]
}
df = pd.DataFrame(data)
print("Dataset : \n")
print(df)
X = df[["Year", "KM_Driven"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,
 test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
prediction = model.predict([[2022,5000]])
print("\n Prediction Car Price: ")
print(prediction)
plt.scatter(df["KM_Driven"], df["Price"])
plt.xlabel("KM_Driven")
plt.ylabel("Price")
plt.title("Car Price Prediction")
plt.show()