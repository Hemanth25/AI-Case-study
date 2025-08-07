# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:05:33 2025

@author: hemanthn
"""

# === model_utils.py ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from logger import setup_logger

logger = setup_logger()

def train_model(data_path):
    data = pd.read_csv(data_path)
    X = data[['income', 'employment_years', 'family_size', 'wealth_index']]
    y = data['eligible']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = xgb.XGBClassifier(learning_rate=0.04, max_depth=4, reg_alpha=0.4)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler
