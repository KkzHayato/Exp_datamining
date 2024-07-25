import pytest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# テスト用のデータを生成する
def generate_test_data():
    np.random.seed(0)
    X = np.random.rand(100, 6)  # 100サンプル、6特徴量
    y = np.random.randint(1, 11, size=100)  # 1から10までの品質スコア
    return X, y

# k_neighbors_learning 関数のテスト
def test_k_neighbors_learning():
    X, y = generate_test_data()
    
    def k_neighbors_learning(a, b):
        X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
        model = KNeighborsClassifier(n_neighbors=22, weights="uniform", algorithm="auto", metric="canberra")
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        return accuracy
    
    accuracy = k_neighbors_learning(X, y)
    
    # 正解率が期待される範囲にあることを確認する
    assert accuracy >= 0.5  # バイナリ分類の場合、0.5以上であれば合格とする

# k_neighbors_gridsearch 関数のテスト
def test_k_neighbors_gridsearch():
    X, y = generate_test_data()
    
    def k_neighbors_gridsearch(a, b):
        K_grid = {KNeighborsClassifier(): {
            "n_neighbors": [i for i in range(20, 50)],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming", "canberra", "braycurtis"],
            "p": [1, 2]
        }}
        
        X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
        
        max_score = 0
        best_param = None
        best_model = None
        
        for model, param in K_grid.items():
            clf = GridSearchCV(model, param)
            clf.fit(X_train, Y_train)
            pred_y = clf.predict(X_test)
            score = f1_score(Y_test, pred_y, average="micro")
            
            if max_score < score:
                max_score = score
                best_param = clf.best_params_
                best_model = model.__class__.__name__

        return max_score, best_param, best_model
    
    max_score, best_param, best_model = k_neighbors_gridsearch(X, y)
    
    # 最大スコアが期待される範囲にあることを確認する
    assert max_score >= 0.5  # スコアの下限を0.5と仮定

# val_curve 関数のテスト（可視化のテストは難しいため、データが生成されるかどうかの確認）
def test_val_curve():
    X, y = generate_test_data()
    
    def val_curve(a, b, model1):
        param_range = [5, 10, 15, 20, 25, 30, 35, 40]
        train_scores, test_scores = validation_curve(
            estimator=model1,
            X=a, y=b,
            param_name="n_neighbors",
            param_range=param_range, cv=10)
        
        return train_scores, test_scores
    
    model = KNeighborsClassifier()
    train_scores, test_scores = val_curve(X, y, model)
    
    # 訓練スコアとテストスコアの形状を確認する
    assert train_scores.shape[1] == len([5, 10, 15, 20, 25, 30, 35, 40])
    assert test_scores.shape[1] == len([5, 10, 15, 20, 25, 30, 35, 40])


print("kneighborlearningのユニットテスト")
#test_k_neighbors_learning()
print("グリッドサーチのユニットテスト")
test_k_neighbors_gridsearch()
print("検証曲線のユニットテスト")
test_val_curve()

