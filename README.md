# 🏙️ Greater Noida Land Price Prediction System
### B.Tech ML Project

> An end-to-end Machine Learning web application that predicts land prices in Greater Noida using a Random Forest Regressor with **94.3% accuracy (R² Score)**.

---

## 📂 Project Structure

```
land_price_prediction/
│
├── app.py                  # Flask web application (routes + prediction logic)
├── model_training.py       # ML model training, EDA, evaluation
├── generate_dataset.py     # Synthetic dataset generation script
├── dataset.csv             # Generated dataset (1,200 rows)
├── model.pkl               # Trained Random Forest model (saved)
├── feature_columns.json    # Feature column order for prediction
├── sector_map.json         # Sector → numeric encoding map
├── metrics.json            # Model evaluation metrics (for UI display)
├── requirements.txt        # Python dependencies
│
├── templates/
│   ├── index.html          # Prediction form page
│   └── result.html         # Prediction result page
│
├── static/
│   ├── style.css           # Complete responsive UI stylesheet
│   └── graphs/
│       ├── price_distribution.png
│       ├── sector_avg_price.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── actual_vs_predicted.png
│
└── README.md               # This file
```

---

## ⚙️ Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## 🚀 How to Run (Step-by-Step)

### Step 1 — Clone / Extract the project
```bash
cd Desktop
# extract the zip or copy the folder
cd land_price_prediction
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate dataset
```bash
python generate_dataset.py
```
✅ This creates `dataset.csv` with 1,200 rows of Greater Noida land data.

### Step 5 — Train the model
```bash
python model_training.py
```
✅ This will:
- Perform EDA
- Train Linear Regression and Random Forest
- Print model comparison table
- Save `model.pkl`, `metrics.json`, and all graphs

### Step 6 — Run the web app
```bash
python app.py
```
✅ Open your browser and visit: **http://127.0.0.1:5000**

---

## 🎯 Features

| Feature | Details |
|---|---|
| **Input Parameters** | 10 features (sector, area, road width, metro distance, etc.) |
| **ML Models** | Linear Regression + Random Forest (compared) |
| **Best Model** | Random Forest (R² = 0.9434) |
| **Dataset Size** | 1,200 rows, 20 sectors |
| **Output** | Price in ₹ with Indian currency formatting |
| **Visualizations** | 4 graphs (distribution, importance, comparison, actual vs predicted) |
| **UI** | Dark luxury theme, fully responsive |

---

## 📊 Model Results

| Metric       | Linear Regression | Random Forest |
|--------------|-------------------|---------------|
|   MAE        |    ₹27,21,933     |   ₹16,74,459  |
| RMSE         |    ₹37,28,353     |   ₹26,88,435  |
| **R² Score** |    **0.8912**     | **0.9434 ✅** |

**Winner: Random Forest** — 94.3% accuracy!

---

## 🔍 Features Used for Prediction

1. **Sector** — Location (Alpha 1, Pari Chowk, etc.)
2. **Area_sqm** — Plot area in square meters
3. **Road_Width_ft** — Adjacent road width in feet
4. **Metro_Dist_km** — Distance from nearest Aqua Line metro
5. **Airport_Dist_km** — Distance from upcoming Jewar Airport
6. **Corner_Plot** — Whether it's a corner plot (Yes/No)
7. **Facing** — Plot facing direction (North/South/East/West)
8. **Nearby_School** — School within 1km (Yes/No)
9. **Nearby_Hospital** — Hospital within 1km (Yes/No)
10. **Commercial_Nearby** — Commercial zone nearby (Yes/No)

---

## 🎓 Viva Presentation Tips

 1. Explain the Problem
> "We built an ML system to predict land prices in Greater Noida. The real estate market is complex — prices depend on location, infrastructure, and amenities. Our model learns these patterns from 1,200 data points."

 2. Explain Data Preprocessing
> "We encoded categorical variables: binary Yes/No → 1/0, Facing direction → one-hot encoding, and Sector → ordinal encoding based on market value."

 3. Why Random Forest Over Linear Regression?
> "Linear Regression assumes a linear relationship between features and price. But real estate pricing is non-linear. Random Forest uses multiple decision trees and captures complex patterns — giving us 94.3% accuracy vs 89.1% for Linear Regression."

 4. Explain Feature Importance
> "According to our model, Area and Metro Distance are the most important features. This makes real-world sense — larger plots cost more, and proximity to the Aqua Metro Line increases land value significantly."

 5. Explain Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average prediction error in ₹
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **R² Score**: % of variance explained (0.94 = 94% accuracy)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask 2.3 |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML5, CSS3 (vanilla) |
| Model Serialization | Joblib |

---

## 👨‍💻 Developed by

**B.Tech 2nd Year Student**
IILM UNIVERSITY Greater Noida
BTECH CSE (AI & ML)  
Greater Noida, Uttar Pradesh  

*name : Jatin Kumar Singh  
*Batch: 2024–2028*

---

*⚠️ Disclaimer: This system uses synthetic training data for academic purposes. Predictions are estimates and should not be used for actual real estate transactions.*
