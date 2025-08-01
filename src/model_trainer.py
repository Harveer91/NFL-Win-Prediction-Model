import pandas as pd 
import joblib
from sqlalchemy import text 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import Database_engine

with Database_engine.connect() as conn:
   df = pd.read_sql(text("SELECT * FROM play_by_play_data_labeled"), con=conn)

target = "posteam_win"

columns_to_remove = ["game_id", "play_id", "home_team", "away_team", "posteam", "defteam", "winning_team", "posteam_win"]

cols_feature = []

#removing unwanted coloums for dataset that will be used for training
for col in df.columns:
   if col not in columns_to_remove:
       cols_feature.append(col)

#new dataset with the prefered data 
X = df[cols_feature]
y = df[target]

X = pd.get_dummies(X)

#setting train/test split to 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

joblib.dump(model, "xgb_play_by_play_model.pkl")
loaded_model = joblib.load("xgb_play_by_play_model.pkl")

y_prediction = loaded_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_prediction))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))
print("Classification Report:\n", classification_report(y_test, y_prediction))

