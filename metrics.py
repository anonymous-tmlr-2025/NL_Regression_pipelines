from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
)

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

METRICS_TO_RECORD = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "medae": median_absolute_error,
}