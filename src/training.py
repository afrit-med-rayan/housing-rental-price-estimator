import time
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

def evaluate_model(name, model, Xtr, Xte, ytr, yte, do_print=True):
    """Fits, predicts, and evaluates a model."""
    t0 = time.time()
    model.fit(Xtr, ytr)
    t_train = time.time() - t0
    preds = model.predict(Xte)

    mse = mean_squared_error(yte, preds)         
    rmse = float(np.sqrt(mse))                  
    mae = mean_absolute_error(yte, preds)
    r2 = r2_score(yte, preds)

    if do_print:
        print(f"--- {name} ---")
        print(f"Train time: {t_train:.1f}s")
        print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.6f}")
    
    return {
        "name": name, 
        "model": model, 
        "rmse": rmse, 
        "mae": mae, 
        "r2": r2, 
        "preds": preds
    }

def tune_hyperparameters(model, param_dist, X_train, y_train, n_iter=8, cv=3, random_state=42):
    """Performs randomized search CV for hyperparameter tuning."""
    rs = RandomizedSearchCV(
        model, 
        param_dist, 
        n_iter=n_iter, 
        scoring="neg_mean_squared_error",
        cv=cv, 
        random_state=random_state, 
        n_jobs=-1, 
        verbose=1
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_

def save_model(model, filename):
    """Saves a model to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Saved model to: {filename}")
