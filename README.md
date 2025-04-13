# 📈 Stock Price Prediction Web App (India Focused)

This is a Streamlit-powered web application that predicts stock prices based on historical data of Indian companies. The app uses machine learning models and allows users to choose date ranges for forecasting trends interactively.

---

## 🌟 Features

- 📊 Load and visualize historical stock data
- 🧠 Predict future stock prices using a pre-trained ML model (`.h5`)
- 📈 Interactive charts and tables using Streamlit
- 📂 Analysis based on the top 10 Indian stocks from the dataset

---

## 🧰 Tech Stack

| Category     | Tools Used         |
|--------------|--------------------|
| Language     | Python             |
| App Framework| Streamlit          |
| ML/AI        | TensorFlow, Keras  |
| Data         | Pandas, NumPy      |
| Visualization| Matplotlib, Seaborn|

---

## 📁 Folder Structure

```bash
stock-Prediction-main/
├── app.py                          # Streamlit app entry point
├── requirements.txt               # Python dependencies
├── Stock_Predictions_Model.h5     # Pre-trained ML model
├── top_10_indian_stocks_last_6.csv # Input dataset
├── .gitignore
├── .gitattributes                 # Git LFS tracking
└── README.md
How to Run the App
Follow these steps to run the project locally:

✅ Step 1: Clone the repository
bash
Copy
Edit
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Replace <your-username> and <repo-name> with your GitHub info.

✅ Step 2: (Optional) Create and activate a virtual environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate         # On macOS/Linux
venv\Scripts\activate            # On Windows
✅ Step 3: Install dependencies
pip install -r requirements.txt
✅ Step 4: Run the Streamlit app
streamlit run app.py



