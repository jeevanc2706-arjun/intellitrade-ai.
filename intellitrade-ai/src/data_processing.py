import numpy as np
import pandas as pd

def create_demo_market_data(n=250, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(mean=10, sigma=0.35, size=n)

    df = pd.DataFrame({"close": close, "volume": volume})
    df["return"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["return"].rolling(20).std()
    df["momentum"] = df["close"].pct_change(5)

    # Demonstration target: next-period direction
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)
