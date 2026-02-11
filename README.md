# 🤖 AutoML - Orchestrator: Modular ML assistant (v1.1).

**Orchestrator** is a robust, modular, and easy-to-use Python library designed to streamline the End-to-End Machine Learning lifecycle. From Exploratory Data Analysis (EDA) to Model Deployment, it automates the tedious parts of Data Science while retaining full control for manual fine-tuning.

Built with **Scikit-Learn**, **Pandas**, and **XGBoost**, it supports both **Classification** and **Regression** problems using a Facade Pattern architecture.

---

## ✨ Key Features

* **🔍 Automated EDA:** Generates statistical summaries and univariate/multivariate plots automatically.
* **🛠️ Feature Engineering:**
    * Auto-detection of numerical/categorical columns.
    * Missing value imputation (Median/Mode).
    * Categorical Encoding (Factorization with JSON mapping).
    * Outlier detection and capping (IQR Method).
* **⚙️ Data Preparation:**
    * Generates dataset variants (With vs. Without Outliers).
    * Auto-Scaling (MinMax and Standard Scaler).
    * Feature Selection (SelectKBest via ANOVA F-value).
* **🥊 Model Tournament:**
    * Trains multiple algorithms simultaneously (RandomForest, XGBoost, SVM, Linear Models, etc.).
    * **Anti-Overfitting Filter:** Automatically discards models with high variance between Train and Test scores.
* **🔧 Hyperparameter Optimization:** Automatic `GridSearchCV` for the winning model.
* **(NEW) Optimization by Stacking:** Takes the best models and merge them together to try to make an even better one. 
* **💾 Production-Ready Artifacts:** Saves the best model, scalers, and encoders in a structured directory, ready for deployment.

---

## ⚡ Quick Start (AutoML Mode)

The `MLOrchestrator` class provides a `run_pipeline` method that executes the entire flow in a single step. Ideal for overnight runs or quick baselines.

```python
from orchestrator import MLOrchestrator

# 1. Initialize
orchestrator = MLOrchestrator()

# 2. Run everything
orchestrator.run_pipeline(
    df_path="data/diabetes.csv",       # Path to your data
    target="Outcome",                  # Target column name
    problem_type="classification",     # "classification" or "regression"
    save_path="./my_experiment",       # Output folder
    optimize=True,                     # Enable Hyperparameter Tuning
    overfitting_filter=True            # Discard overfitted models
)
```

## 🧪 Advanced Usage (Manual Step-by-Step)

For Data Scientists who need more control, you can access each module individually through the orchestrator.

### 1. Load & Explore
```python
orc = MLOrchestrator()
orc.load_data("data/house_prices.csv", target="Price")

# Generate plots and stats
orc.execute_analysis(mode="all")
```

### 2. Feature Engineering
This step generates clean datasets and handles outliers.
```python
# Saves encoders and intermediate CSVs to './project/datasets'
orc.prepare_features(problem_type="classification", 
                    save_path="./project", save_data=True)
```

### 3. Initialize Trainer & Add Custom Models
You can rely on defaults or inject your own model configurations.

```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize
orc.initialize_trainer()

# Optional: Add a custom model to the roster
knn = KNeighborsClassifier(n_neighbors=5)
orc.trainer.add_model("KNN", knn)
```

### 4. Train & Optimize

```python
# Run the battle (Screening phase)
orc.run_training_cycle(
    optimize=True,           # Optimize the winner
    gap_threshold=0.20,      # Max allowed Train/Test gap (20%)
    trim_models=True         # Remove losing models to save RAM
)
```

### 5. Save Artifacts

```python
orc.save_artifacts()
```

## 📂 Output Structure

After a successful run, the `save_path` directory will look like this:

```text
my_experiment/
├── datasets/                 # Processed CSVs (Train/Test variants)
│   ├── X_train_with_outliers_minmax.csv
│   ├── X_test_without_outliers_standard.csv
│   └── ...
├── encoders/                 # JSON mappings for categorical features
│   ├── Gender_encoder.json
│   ├── City_encoder.json
│   └── ...
├── scalers/                  # Scaler objects (.pkl)
│   ├── scaler_minmax.pkl
│   └── scaler_standard.pkl
└── models/                   # Final Production Artifacts
    └── RandomForest_best.pkl # Contains Model + Scaler + Selector + Features

```

## 🛠️ Architecture
```text
AutoML/
│
├── __init__.py           
├── orchestrator.py       # 👑 The Hub.
├── explorer.py           # 🕵️ The Scout.
├── preparer.py           # 🧹 The Cleaner.
├── feat_engineer.py      # 🏗️ The Architect.
├── trainer.py            # 🏋️ The Muscle.
└── visualizer.py         # 🎨 The Artist.
```

* **`DataExplorer`**: Handles data type inference (Cardinality detection) and statistical summaries.
* **`Visualizer`**: Uses Matplotlib/Seaborn to generate distributions, boxplots, and correlation heatmaps.
* **`FeatureEngineer`**: Applies transformations:
    * Null Imputation (Median/Mode).
    * Categorical Encoding (Factorization).
    * Outlier Capping (Winsorization/IQR).
* **`DataPreparer`**: Splits data and applies Scalers (MinMax/Standard) and Feature Selection (ANOVA).
* **`ModelTrainer`**: Manages the training loop, metric calculation (Accuracy/RMSE), Hyperparameter Optimization (`GridSearchCV`) and Stacking Ensemble.
* **`MLOrchestrator`**: The **Facade** that connects all modules into a unified workflow.

---

## 📋 Requirements

* Python >= 3.8
* Pandas >= 1.3.0
* Scikit-Learn >= 1.0.0
* XGBoost >= 1.6.0
* Matplotlib >= 3.5.0
* Seaborn >= 0.11.0

---

## 📄 License

This project is open-source and available under the **MIT License**.