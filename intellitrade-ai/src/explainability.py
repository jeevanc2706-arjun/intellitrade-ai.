import pandas as pd

def feature_contributions(model, feature_names):
    """Return absolute linear-model coefficients as a simple explanation."""
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance = abs(model.coef_[0])
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
