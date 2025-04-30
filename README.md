# 🩺 Breast Cancer Prediction Web App | 乳癌預測系統

This is a machine learning-powered interactive web application that predicts whether a breast tumor is benign or malignant based on user input or uploaded CSV files. Built with Streamlit and supporting multiple models including Random Forest, SVM, XGBoost, and a PyTorch MLP.

這是一個基於機器學習的互動式網頁應用程式，可根據使用者輸入或上傳的資料，預測乳癌腫瘤為良性或惡性。使用 Streamlit 架設介面，支援多種模型（隨機森林、SVM、XGBoost、PyTorch MLP）。

---

## 🔍 Features | 功能特色

  - 🎛 **Multi-model prediction** 多模型切換：RF / SVM / XGBoost / PyTorch
  - 🖥 **Streamlit interface** 中文化網頁介面
  - 📤 **CSV batch upload** 支援上傳 CSV 批次預測
  - 📈 **Model probability output** 顯示預測機率與信心值
  - 🧠 **Feature importance plot** 特徵重要性圖（支援的模型）
  - 📊 **Data visualization** 資料視覺化（BoxPlot / Histogram / Heatmap）

---

## 📦 Tech Stack | 使用技術

- Python
- Scikit-learn
- XGBoost
- PyTorch
- Streamlit
- Pandas / Matplotlib / Seaborn

---

## ⚙️Usage | 快速使用

### 1. Clone this repo | 複製專案
```bash
git clone https://github.com/YOUR_USERNAME/breast_cancer_prediction.git
cd breast_cancer_prediction
```

### 2. Create virtual environment | 建立虛擬環境
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies | 安裝套件
```bash
pip install -r requirements.txt
```

### 4. Train models | 訓練模型（如果沒有現成的）
```bash
python3 train_models.py
python3 train_pytorch.py
```

### 5. Run the app | 啟動預測系統
```bash
streamlit run app.py
```

---

## 📁 Project Structure | 專案結構
  ```text
  ├── app.py                   # Streamlit Web UI
  ├── train_models.py          # 訓練 scikit-learn 模型
  ├── train_pytorch.py         # 訓練 PyTorch 模型
  ├── rf_model.pkl             # 隨機森林模型
  ├── svm_model.pkl            # SVM 模型
  ├── xgb_model.pkl            # XGBoost 模型
  ├── pytorch_model.pt         # PyTorch 模型
  ├── pytorch_scaler.pkl       # PyTorch 輸入標準化器
  ├── requirements.txt         # 套件需求清單
  └── README.md
  ```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

---

## 🙋‍♂️ Author

If you like this project or want to collaborate, feel free to reach out:
如果你喜歡這個專案或希望合作，歡迎聯絡我：

- GitHub: [https://github.com/D0683160](https://github.com/D0683160)
- Email: antoniolee489@gmail.com

---

## 💡 Credit | 資料來源
Dataset from [UCI Breast Cancer Wisconsin Diagnostic Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

模型設計靈感來自醫療分類應用。

---

## 📜 License
MIT License
