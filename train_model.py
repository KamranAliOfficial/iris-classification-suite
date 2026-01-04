import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y

# 2. Build Advanced Pipeline (Scaling + Model)
def build_model():
    # Pipeline use karna professional approach hai
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Data ko normalize karega
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline

# 3. Execution
if __name__ == "__main__":
    print("🚀 Training Professional Model...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model()
    model.fit(X_train, y_train)
    
    # Model Save karna
    joblib.dump(model, 'iris_model.pkl')
    print("✅ Model Trained and Saved as 'iris_model.pkl'")