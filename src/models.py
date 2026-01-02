try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

def get_model(model_name, **params):
    """Factory function to get a model by name."""
    if model_name == "Ridge":
        return Ridge(**params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "MLP":
        return MLPRegressor(**params)
    elif model_name == "XGBoost":
        if xgb is None:
            raise ImportError("xgboost is not installed")
        return xgb.XGBRegressor(**params)
    elif model_name == "LightGBM":
        if lgb is None:
            raise ImportError("lightgbm is not installed")
        return lgb.LGBMRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
