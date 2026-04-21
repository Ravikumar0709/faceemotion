import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression

# High-Performance Models
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier

# 1. RAMANUJAN SUMS
def ramanujan_sum(n, q):
    def gcd(a, b):
        while b: a, b = b, a % b
        return a
    if q == 0: return 0
    return sum(np.cos(2 * np.pi * a * n / q) for a in range(1, q + 1) if gcd(a, q) == 1)

# 2. GANomaly ARCHITECTURE
class GANomalyTabular(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(GANomalyTabular, self).__init__()
        self.encoder1 = nn.Sequential(nn.Linear(input_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, input_dim))
        self.encoder2 = nn.Sequential(nn.Linear(input_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, latent_dim))
    def forward(self, x):
        z1 = self.encoder1(x)
        x_rec = self.decoder(z1)
        z2 = self.encoder2(x_rec)
        return x_rec, z1, z2

# 3. FEATURE ENGINE
def engineer_features(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Monthly_to_Total'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5)
    services = ['OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'OnlineBackup']
    df['Service_Density'] = (df[services] == 'Yes').sum(axis=1)
    df['Raman_12'] = df['tenure'].apply(lambda x: ramanujan_sum(x, 12))
    df['Raman_4'] = df['tenure'].apply(lambda x: ramanujan_sum(x, 4))
    df['Avg_By_Contract'] = df.groupby('Contract')['MonthlyCharges'].transform('mean')
    return df

# --- EXECUTION ---
print("🚀 Initializing 0.917+ Winning Pipeline...")
train_df = pd.read_csv('train.csv')
train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})
train_df = engineer_features(train_df)

# Target Encoding Initialization
train_df['Target_Enc'] = 0.0 
skf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for t_idx, v_idx in skf_cv.split(train_df, train_df['Churn']):
    train_fold, val_fold = train_df.iloc[t_idx], train_df.iloc[v_idx]
    means = train_fold.groupby('Contract')['Churn'].mean()
    train_df.loc[v_idx, 'Target_Enc'] = train_df.loc[v_idx, 'Contract'].map(means).fillna(train_df['Churn'].mean())

# Encoding Categoricals
encoders = {}
for col in train_df.select_dtypes(include=['object']).columns:
    if col not in ['id', 'Churn']:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le

X = train_df.drop(['id', 'Churn'], axis=1)
y = train_df['Churn']

# GANomaly Feature Generation
scaler = MinMaxScaler()
X_scaled = torch.FloatTensor(scaler.fit_transform(X))
gan = GANomalyTabular(X.shape[1])
optimizer = torch.optim.Adam(gan.parameters(), lr=0.002)

print("🧠 Training GANomaly for Anomaly Scoring...")
X_normal = X_scaled[y == 0]
for epoch in range(100):
    gan.train()
    optimizer.zero_grad()
    x_rec, z1, z2 = gan(X_normal)
    loss = nn.MSELoss()(x_rec, X_normal) + nn.MSELoss()(z1, z2)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    gan.eval()
    x_rec, _, _ = gan(X_scaled)
    train_df['GAN_Score'] = torch.mean((X_scaled - x_rec)**2, dim=1).numpy()

# 4. FINAL STACKING & METRICS
X_final = train_df.drop(['id', 'Churn'], axis=1)
oof_probs = np.zeros(len(X_final))

base_models = [
    ('cat', CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=6, silent=True)),
    ('xgb', XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6, eval_metric='auc')),
    ('lgbm', LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=31))
]

# FIX: final_estimator changed to LogisticRegression
stack_ensemble = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(C=0.1), 
    cv=5, 
    stack_method='predict_proba', 
    n_jobs=-1
)

print("\n--- 📊 STARTING 10-FOLD VALIDATION ---")
for fold, (t_idx, v_idx) in enumerate(skf_cv.split(X_final, y)):
    Xt, Xv = X_final.iloc[t_idx], X_final.iloc[v_idx]
    yt, yv = y.iloc[t_idx], y.iloc[v_idx]
    
    stack_ensemble.fit(Xt, yt)
    
    # LogisticRegression final_estimator allows predict_proba
    fold_preds = stack_ensemble.predict_proba(Xv)[:, 1]
    oof_probs[v_idx] = fold_preds
    
    auc = roc_auc_score(yv, fold_preds)
    print(f"Fold {fold+1:02d} | ROC-AUC: {auc:.5f}")

print("\n" + "="*40)
print(f"OVERALL CV ROC-AUC: {roc_auc_score(y, oof_probs):.5f}")
print("="*40)

# Final fit and save
stack_ensemble.fit(X_final, y)
joblib.dump({
    'model': stack_ensemble, 
    'gan': gan, 
    'scaler': scaler, 
    'encoders': encoders, 
    'features': X_final.columns.tolist(),
    'global_mean': train_df['Churn'].mean()
}, 'best_model.pkl')

print("✅ Model saved. Run submit.py next!")