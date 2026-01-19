import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

print('Libraries imported.')

#tsk1----------------------------------------------------------------
path = 'train.csv'
df = pd.read_csv(path)
print('Shape:', df.shape)
df.head()

#tsk2----------------------------------------------------------------


#Drop duplicates
df = df.drop_duplicates()

#check invalid cols
print(df.columns)

#Missing value check
df.isna().sum()

#seperate
X = df.drop(columns=['price_range'],axis=1)
y = df['price_range']

#Class distribution
print(y.value_counts().sort_index())

#tsk3----------------------------------------------------------------

# avoid categorical my dataset is only numerical cols
numeric_features = X.columns
print("All numeric features:", list(numeric_features))

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

print("pipeline created.")
preprocessor

#tsk4----------------------------------------------------------------
#primary model rf
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("primary model: RandomForestClassifier")

# Full pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
#tsk5 ----------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
rf_pipeline.fit(X_train, y_train)
print('Trained.')

#tsk6 ----------------------------------------------------------------

cv_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("Cv accuracy scores:")
print(cv_scores)

print("\n avg:", cv_scores.mean().round(2))
print("standard deviation:", cv_scores.std().round(2))


#tsk7 ----------------------------------------------------------------


param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)


print("best params:",grid_search.best_params_)

print("\n best cv accuracy:",grid_search.best_score_.round(4))

#tsk8 ----------------------------------------------------------------

best_model = grid_search.best_estimator_

print("best model:",best_model)


#tsk9 ----------------------------------------------------------------

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("accuracy:", acc)
print("\nConfusion Matrix:",cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#save file using pkl
#----------------------------------------------------------------

import pickle
with open("mobile_price_rf_pipeline.pkl", "wb") as file:
    pickle.dump(best_model, file)
print("Save done")
