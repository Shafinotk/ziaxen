import numpy as np
from fastapi import HTTPException
from typing import List

def softmax_if_needed(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float).squeeze()
    s = x.sum()
    if s < 0.999 or s > 1.001:
        e = np.exp(x - np.max(x))
        x = e / np.sum(e)
    return x

def ensure_2d(matrix: List[List[float]]) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail=f"'data' must be a 2D array, got shape {X.shape}")
    return X

def reshape_to_model(X: np.ndarray, input_shape) -> np.ndarray:
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    target = input_shape[1:]
    if len(target) == 1:
        return X if X.ndim == 2 else X.reshape((X.shape[0], -1))
    if len(target) == 2:
        return X[None, :, :] if X.ndim == 2 else X
    if len(target) == 3:
        H, W, C = target
        H = H or X.shape[0]
        W = W or X.shape[1]
        C = C or 1
        return X.reshape((1, H, W, C))
    return X

def preprocess_lidar_radar(data: List[List[float]]) -> np.ndarray:
    import pandas as pd
    import numpy as np

    columns = ['X','Y','Z','Intensity','Azimuth','Elevation','Range',
               'Speed','RCS','Power','Noise']
    df = pd.DataFrame(data, columns=columns)

    df['Velocity_Magnitude'] = np.sqrt(df['Speed']**2 + df['Azimuth']**2)
    df['Velocity_Range_Ratio'] = df['Speed'] / (df['Range'] + 1e-5)
    df['Log_Intensity'] = np.log1p(df['Intensity'])
    df['Power_Noise_Ratio'] = df['Power'] / (df['Noise'] + 1e-5)

    ordered_features = [
        'X','Y','Z','Intensity','Azimuth','Elevation','Range','Speed',
        'RCS','Power','Noise',
        'Velocity_Magnitude','Velocity_Range_Ratio','Log_Intensity','Power_Noise_Ratio'
    ]
    return df[ordered_features].values
