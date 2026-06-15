import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\creditcard.csv")
print(df.shape)
print(df.head())

duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)
df =df.drop_duplicates()
print(df.shape)

amount_scaler = StandardScaler()
df['Amount'] = amount_scaler.fit_transform(df[['Amount']])

time_scaler = StandardScaler()
df['Time'] = time_scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis = 1)
y = df['Class']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

X_train.to_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_train.csv", index=False)

X_test.to_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/X_test.csv", index=False)

y_train.to_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/y_train.csv", index=False)

y_test.to_csv(r"C:\Users\win10\OneDrive\Desktop\Data-Projects\Portfolio Project\Credit Card Fraud Detection\01_Data\processed_data/y_test.csv", index=False)