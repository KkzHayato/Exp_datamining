import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error


df = pd.read_csv("winequality-white.csv")

# 特徴量とターゲットに分離
x = df.drop('quality',axis=1)
y = df['quality']

# トレーニングデータとテストデータに分割（80% トレーニング、20% テスト）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 分割されたデータの形状を表示
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# ランダムフォレストモデルの作成
rf_model = RandomForestClassifier(n_estimators=180, max_depth=None, max_leaf_nodes=None, min_samples_split=4)

# モデルの訓練
rf_model.fit(x_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# テストデータで予測
y_pred = rf_model.predict(x_test)

# モデルの評価
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)

#テスト〜