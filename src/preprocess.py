from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df.drop("failure", axis=1)
    y = df["failure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler