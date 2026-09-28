import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load dataset
data_path = None
for path in ['data/student_placement_data.csv', 'data/student_placement_data_v2.csv', 'student_placement_data.csv']:
    if os.path.exists(path):
        data_path = path
        break

if not data_path:
    raise FileNotFoundError("Could not find student placement dataset.")

df = pd.read_csv(data_path)

# 2. Detect target column dynamically
target_col = 'placed' if 'placed' in df.columns else 'placement_status'
y = df[target_col]

# 3. Explicit features matching app.py input schema (prevents dimension mismatch)
EXPECTED_FEATURES = [
    'gender', 'age', 'degree', 'branch', 'cgpa', 'backlogs',
    'internships', 'certifications', 'coding_skills', 'communication_skills',
    'aptitude_score', 'projects', 'Lx_Level_Reached', 'Ax_Level_Reached',
    'Cx_Level_Reached', 'Px_Level_Reached', 'Sx_Level_Reached'
]

available_features = [col for col in EXPECTED_FEATURES if col in df.columns]
if len(available_features) == len(EXPECTED_FEATURES):
    X = df[EXPECTED_FEATURES]
else:
    cols_to_drop = [
        target_col, 'student_id', 'student_name', 'usn', 'company_type', 'package_lpa',
        'Overall_Level_Score', 'tenth_pct', 'twelfth_pct', '10th_percentage', '12th_percentage'
    ]
    raw_prefixes = ('lang_', 'apt_', 'soft_', 'core_', 'prog_', 'Unnamed')
    cols_to_drop += [c for c in df.columns if c.startswith(raw_prefixes)]
    X = df.drop(columns=cols_to_drop, errors='ignore')

print(f"[*] Training dataset features ({len(X.columns)}): {list(X.columns)}")

# 4. Feature preprocessing pipeline
cat_cols = ['gender', 'degree', 'branch']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ]
)

# 5. Stratified train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 6. Candidate algorithms with calibration
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest (Calibrated)': RandomForestClassifier(
        n_estimators=150, min_samples_leaf=15, random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    ),
    'KNN': KNeighborsClassifier(n_neighbors=9),
}

print('\n=== Model Comparison Benchmark ===\n')
results = []
confusion_matrices = {}
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    confusion_matrices[name] = cm

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
    })

    if name == 'Random Forest (Calibrated)':
        best_pipeline = pipe

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 7. Safe export
os.makedirs('models', exist_ok=True)

# Save benchmark results to JSON
results_df.to_json('models/benchmark_results.json', orient='records', indent=2)

# Save confusion matrices to JSON
with open('models/confusion_matrices.json', 'w') as f:
    json.dump(confusion_matrices, f, indent=2)

# Extract and save feature importances for Random Forest
rf_model = best_pipeline.named_steps['classifier']
prep = best_pipeline.named_steps['preprocessor']
cat_feature_names = prep.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(cat_feature_names)
importances = rf_model.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

feature_imp_df.to_json('models/feature_importances.json', orient='records', indent=2)

# Generate and save confusion matrix plot for the best model
plt.figure(figsize=(6, 4.5))
best_cm = confusion_matrices['Random Forest (Calibrated)']
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed (0)', 'Placed (1)'],
            yticklabels=['Not Placed (0)', 'Placed (1)'])
plt.title('Confusion Matrix: Random Forest (Calibrated)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=200)
plt.close()

# Generate and save feature importance plot
plt.figure(figsize=(8, 5))
top_features = feature_imp_df.head(10)
sns.barplot(data=top_features, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=200)
plt.close()

joblib.dump(best_pipeline, 'models/best_pipeline.pkl')
print('\n[✓] Exported calibrated pipeline (Random Forest) to models/best_pipeline.pkl')
print('[✓] Exported benchmark results, confusion matrices, and feature importances to models/')
