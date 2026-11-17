"""
Training script for Telco Customer Churn prediction.
Trains 6 classifiers, compares them, and saves the best as a sklearn Pipeline.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Add parent dir to path so we can import features
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features import clean_data, create_features, encode_features


def load_and_prepare_data(data_path: str):
    """Load CSV, clean, engineer features, encode, and split."""
    df_raw = pd.read_csv(data_path)
    print(f"Raw data shape: {df_raw.shape}")
    
    df = clean_data(df_raw)
    df = create_features(df)
    df = encode_features(df)
    
    print(f"\nFinal encoded shape: {df.shape}")
    print(f"Feature count: {df.shape[1] - 1}")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_and_compare(X_train, X_test, y_train, y_test):
    """
    Train 6 classifiers and return comparison results.
    StandardScaler for LR, SVM, KNN. Trees use unscaled data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'SVM': (SVC(probability=True, random_state=42), True),
        'KNN': (KNeighborsClassifier(), True),
        'Naive Bayes': (GaussianNB(), False),
    }
    
    results = []
    trained_models = {}
    
    for name, (model, needs_scaling) in classifiers.items():
        if needs_scaling:
            Xtr, Xte = X_train_scaled, X_test_scaled
        else:
            Xtr, Xte = X_train.values, X_test.values
        
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        
        metrics = {
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        }
        results.append(metrics)
        trained_models[name] = (model, needs_scaling)
        
        print(f"\n{name}:")
        print(f"  Accuracy={metrics['Accuracy']}, Precision={metrics['Precision']}, "
              f"Recall={metrics['Recall']}, F1={metrics['F1']}, AUC={metrics['ROC-AUC']}")
    
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    return results_df, trained_models, scaler


def save_best_model(best_name, trained_models, scaler, feature_names, model_dir):
    """
    Save the best model as a sklearn Pipeline with preprocessing.
    For models that need scaling, include StandardScaler in the pipeline.
    """
    model, needs_scaling = trained_models[best_name]
    
    if needs_scaling:
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('model', model)
        ])
    
    # Attach metadata
    pipeline.feature_names_ = feature_names
    pipeline.needs_scaling_ = needs_scaling
    pipeline.model_name_ = best_name
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'churn_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\nSaved best model ({best_name}) to {model_path}")
    
    return pipeline


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'telco-customer-churn.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(data_path)
    results_df, trained_models, scaler = train_and_compare(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    best_name = results_df.iloc[0]['Model']
    print(f"\nBest model by ROC-AUC: {best_name}")
    
    # Save best model
    pipeline = save_best_model(best_name, trained_models, scaler, feature_names, model_dir)
    
    # Print detailed evaluation for best model
    model, needs_scaling = trained_models[best_name]
    if needs_scaling:
        X_test_eval = scaler.transform(X_test)
    else:
        X_test_eval = X_test.values
    
    y_pred = model.predict(X_test_eval)
    
    print("\n" + "="*70)
    print(f"DETAILED EVALUATION: {best_name}")
    print("="*70)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix (sum should be {len(y_test)}):")
    print(cm)
    print(f"Matrix sum: {cm.sum()}")
    
    # Feature importance for LR and RF
    for name in ['Logistic Regression', 'Random Forest']:
        m, _ = trained_models[name]
        if name == 'Logistic Regression':
            importances = np.abs(m.coef_[0])
        else:
            importances = m.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names)
        feat_imp = feat_imp.sort_values(ascending=False).head(10)
        print(f"\nTop 10 features ({name}):")
        for feat, imp in feat_imp.items():
            print(f"  {feat}: {imp:.4f}")
    
    return results_df, pipeline


if __name__ == '__main__':
    results_df, pipeline = main()
