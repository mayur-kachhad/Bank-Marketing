# config.py

#File Paths
DATA_FILE_PATH = 'data/eda_data.csv'
MODEL_SAVE_PATH = 'saved_model/lgbm_pipeline.pkl'

# Target Variable
TARGET_COLUMN = 'y'

# Feature Lists
NUM_COLS = [
    'age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

CAT_COLS = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome'
]

# Model Parameters
# Best parameters found during RandomizedSearchCV in  notebook
LGBM_BEST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'num_leaves': 63,
    'min_child_samples': 10,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

#Prediction Threshold
#Optimal threshold determined from the profit analysis in notebook
OPTIMAL_THRESHOLD = 0.1438