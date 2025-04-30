# train_models.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle

# 載入資料
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pickle.dump(rf, open("rf_model.pkl", "wb"))
print("✅ 儲存 rf_model.pkl 完成")

# SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
pickle.dump(svm, open("svm_model.pkl", "wb"))
print("✅ 儲存 svm_model.pkl 完成")

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)
pickle.dump(xgb, open("xgb_model.pkl", "wb"))
print("✅ 儲存 xgb_model.pkl 完成")

