import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 載入資料集
data = load_breast_cancer()
feature_names = data.feature_names

# 中文對照表
feature_name_map = {
    "mean radius": "平均半徑",
    "mean texture": "平均紋理",
    "mean perimeter": "平均周長",
    "mean area": "平均面積",
    "mean smoothness": "平均平滑度",
    "mean compactness": "平均緊密度",
    "mean concavity": "平均凹度",
    "mean concave points": "平均凹點數",
    "mean symmetry": "平均對稱性",
    "mean fractal dimension": "平均分形維度",
    "radius error": "半徑誤差",
    "texture error": "紋理誤差",
    "perimeter error": "周長誤差",
    "area error": "面積誤差",
    "smoothness error": "平滑度誤差",
    "compactness error": "緊密度誤差",
    "concavity error": "凹度誤差",
    "concave points error": "凹點數誤差",
    "symmetry error": "對稱性誤差",
    "fractal dimension error": "分形維度誤差",
    "worst radius": "最大半徑",
    "worst texture": "最大紋理",
    "worst perimeter": "最大周長",
    "worst area": "最大面積",
    "worst smoothness": "最差平滑度",
    "worst compactness": "最差緊密度",
    "worst concavity": "最差凹度",
    "worst concave points": "最差凹點數",
    "worst symmetry": "最差對稱性",
    "worst fractal dimension": "最差分形維度"
}

# 頁面標題
st.title("乳癌預測系統 🩺")
st.caption("請輸入以下特徵，我們將預測是否為惡性腫瘤")

# 選擇模型
model_choice = st.selectbox(
    "選擇使用的預測模型",
    ["Random Forest", "SVM", "XGBoost", "PyTorch MLP"]
)

# 載入模型
if model_choice == "PyTorch MLP":
    import torch
    import torch.nn as nn
    import joblib

    class BreastCancerMLP(nn.Module):
        def __init__(self, input_size=30):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            return self.net(x)

    model = BreastCancerMLP()
    model.load_state_dict(torch.load("pytorch_model.pt"))
    model.eval()
    scaler = joblib.load("pytorch_scaler.pkl")
else:
    model_path = {
        "Random Forest": "rf_model.pkl",
        "SVM": "svm_model.pkl",
        "XGBoost": "xgb_model.pkl"
    }
    model = pickle.load(open(model_path[model_choice], "rb"))

# 使用者輸入特徵
tabs = st.tabs(["第 1 頁", "第 2 頁", "第 3 頁"])
user_input = []
for i, tab in enumerate(tabs):
    with tab:
        for j in range(10):
            idx = i * 10 + j
            name = feature_names[idx]
            chinese_name = feature_name_map.get(name, name)
            val = st.number_input(
                label=f"{idx+1}. {chinese_name}",
                min_value=0.0,
                max_value=10000.0,
                value=1.0
            )
            user_input.append(val)

# 預測按鈕
if st.button("進行預測"):
    input_data = np.array([user_input])

    if model_choice == "PyTorch MLP":
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            proba = probs
            prediction = np.argmax(proba)
    else:
        proba = model.predict_proba(input_data)[0]
        prediction = model.predict(input_data)[0]

    result = "良性 🎉" if prediction == 1 else "惡性 ⚠️"

    st.subheader("🔍 預測結果")
    st.success(f"預測結果為：**{result}**")

    st.subheader("📊 預測機率")
    st.write(f"良性：{proba[1]*100:.2f}%")
    st.write(f"惡性：{proba[0]*100:.2f}%")
    st.progress(float(proba[1]))

    # 特徵重要性
    if hasattr(model, "feature_importances_"):
        st.subheader("🧠 模型特徵重要性")
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(6, 10))
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = np.array([feature_name_map.get(f, f) for f in feature_names])
        ax.barh(top_features[top_idx][::-1], importances[top_idx][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Features")
        st.pyplot(fig)
    elif model_choice == "PyTorch MLP":
        st.info("⚠️ PyTorch 模型目前不支援特徵重要性視覺化")

# 上傳 CSV 批量預測
st.divider()
st.subheader("📁 上傳 CSV 批量預測")

uploaded_file = st.file_uploader("上傳包含 30 欄位的 CSV（需中文標題）", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 30:
            st.error("❌ CSV 欄位數量錯誤，請確認共有 30 欄位！")
        else:
            if model_choice == "PyTorch MLP":
                scaled = scaler.transform(df.values)
                tensor = torch.tensor(scaled, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(tensor)
                    probas = torch.softmax(logits, dim=1).numpy()
                    preds = np.argmax(probas, axis=1)
            else:
                preds = model.predict(df.values)
                probas = model.predict_proba(df.values)

            df["預測結果"] = np.where(preds == 1, "良性", "惡性")
            df["良性機率"] = (probas[:, 1] * 100).round(2)
            df["惡性機率"] = (probas[:, 0] * 100).round(2)

            st.success("✅ 預測完成！以下是結果：")
            st.dataframe(df.head())
            csv_download = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下載預測結果 CSV", data=csv_download, file_name="breast_cancer_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")

# 資料視覺化
st.divider()
st.subheader("📊 資料視覺化")

viz_df = pd.DataFrame(data.data, columns=[feature_name_map.get(f, f) for f in feature_names])
viz_df['target'] = pd.Series(data.target).map({0: "惡性", 1: "良性"})

if st.checkbox("📦 顯示 Box Plot（良性 vs 惡性）", value=True):
    fig, axes = plt.subplots(10, 3, figsize=(18, 30))
    for i, col in enumerate(viz_df.columns[:-1]):
        r, c = divmod(i, 3)
        sns.boxplot(x='target', y=col, data=viz_df, ax=axes[r][c])
        axes[r][c].set_title(col)
        axes[r][c].set_xlabel("")
        axes[r][c].set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

if st.checkbox("📊 顯示直方圖（Histograms）"):
    fig, axes = plt.subplots(10, 3, figsize=(18, 30))
    for i, col in enumerate(viz_df.columns[:-1]):
        r, c = divmod(i, 3)
        sns.histplot(viz_df[col], bins=30, kde=True, ax=axes[r][c])
        axes[r][c].set_title(col)
        axes[r][c].set_xlabel("")
        axes[r][c].set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)

if st.checkbox("🔥 顯示相關性 Heatmap"):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = viz_df.drop(columns="target").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("特徵之間的相關性")
    st.pyplot(fig)

