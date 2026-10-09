import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(__file__))
from data_processing import create_demo_market_data
from explainability import feature_contributions

os.makedirs("results", exist_ok=True)

df = create_demo_market_data()

features = ["return", "ma_5", "ma_20", "volatility", "momentum"]
X = df[features]
y = df["target"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

df.loc[df.index[split:], "prediction"] = pred
df["signal"] = df["prediction"].map({1: "BUY", 0: "SELL"})

# Basic risk layer
df["risk"] = np.where(df["volatility"] > df["volatility"].median(), "HIGH", "NORMAL")

print(f"Demonstration test accuracy: {accuracy:.4f}")
print("\nLatest signal:")
print(df[["close", "signal", "volatility", "risk"]].tail(1).to_string(index=False))

# Market/signal chart
plt.figure(figsize=(10, 5))
plt.plot(df["close"], label="Synthetic Close Price")
buy = df.index[df["signal"] == "BUY"]
sell = df.index[df["signal"] == "SELL"]
plt.scatter(buy, df.loc[buy, "close"], marker="^", label="BUY", s=25)
plt.scatter(sell, df.loc[sell, "close"], marker="v", label="SELL", s=25)
plt.title("IntelliTrade AI Demonstration Signals")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("results/market_signals.png", dpi=200)
plt.close()

# Explainability chart
imp = feature_contributions(model, features)
plt.figure(figsize=(8, 5))
plt.barh(imp["feature"], imp["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Absolute Model Coefficient")
plt.title("Feature Importance / Explanation")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=200)
plt.close()
