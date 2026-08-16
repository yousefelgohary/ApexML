<div align="center">
  <img src="assets/ApexML_Logo.png" alt="ApexML Logo" width="200" />
  <h1>ApexML</h1>
  <p><em>Elevating Data. Maximizing Metrics.</em></p>
  <p><b>Enterprise-Grade ML Dashboard | Powered by Gemini AI</b></p>

  <p>
    <a href="https://apexml.streamlit.app" target="_blank">
      <img src="https://img.shields.io/badge/🟢_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
    </a>
  </p>

  <p>
    <a href="https://github.com/yousefelgohary/ApexML/issues">Report Bug</a>
    ·
    <a href="https://github.com/yousefelgohary/ApexML/issues">Request Feature</a>
  </p>
</div>

<hr />

## 🚀 About ApexML

**ApexML** is a comprehensive, AI-powered Streamlit dashboard designed for end-to-end Machine Learning workflows. Whether you're a data scientist, analyst, or enthusiast, ApexML streamlines your journey from raw data to robust predictions. 

By integrating **Gemini AI**, ApexML acts as your intelligent co-pilot, providing smart suggestions for data preprocessing, feature engineering, and model selection.

---

## ✨ Features

- **📂 Dataset Overview & Validation**: Instantly upload CSV/Excel files and get comprehensive metadata, data quality reports, and column-level statistics.
- **🧹 Smart Preprocessing**: AI-suggested strategies for handling missing values, duplicates, outliers, and data encoding. 
- **🔬 Feature Engineering & EDA**: Dive deep into your data with interactive Plotly charts. Support for univariate, bivariate, and multivariate analysis.
- **🤖 ML Studio**: Train and compare multiple models effortlessly. Fine-tune hyperparameters or leverage the *Competition Boost* mode to automatically assemble the strongest possible predictive pipeline.
- **🏆 Evaluation Leaderboard**: Compare models using holdout and cross-validation metrics. Analyze overfitting gaps, training speeds, and model sizes to lock in the final winner.
- **🔮 Inference & Prediction**: Generate predictions on new data via single-entry forms, batch file uploads, or a dedicated final-test workflow.
- **🌙 Dark Mode**: Beautiful and seamless light/dark mode UI integration for optimal viewing.

---

## 📊 Supported Algorithms

### Classification
* Logistic Regression
* Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)
* XGBoost
* LightGBM
* AdaBoost
* Artificial Neural Networks (ANN)

### Regression
* Linear / Polynomial
* Ridge / Lasso
* Decision Tree
* Random Forest
* Support Vector Regressor (SVR)
* KNN
* XGBoost
* LightGBM
* AdaBoost
* ANN

### Clustering & Dimensionality Reduction
* K-Means, Hierarchical Clustering, DBSCAN
* PCA, t-SNE

---

## 🛠️ Technology Stack

- **Frontend/Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn
- **Visualizations:** Plotly, Matplotlib, Seaborn, SHAP, mlxtend
- **AI Integration:** Google GenAI (`google-genai`)

---

## 💻 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yousefelgohary/ApexML.git
   cd ApexML
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy the `.env.example` file to `.env`
   - Add your Gemini AI API key to the `.env` file:
     ```env
     GEMINI_API_KEY=your_api_key_here
     ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment (Streamlit Cloud)

ApexML is ready to be deployed on Streamlit Community Cloud:
1. Push your code to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Click "New app", select your repository (`yousefelgohary/ApexML`), branch (`main`), and main file path (`app.py`).
4. **Important**: Add your `GEMINI_API_KEY` to the Streamlit Secrets in the app settings before deploying.
5. Click "Deploy".

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License.

<div align="center">
  <sub>Built with ❤️ by Yousef Elgohary and the open-source community.</sub>
</div>
