# ==========================================
# 1. Standard Python Libraries
# ==========================================
import os
import json
import pickle
import warnings
from typing import Optional, List, Dict, Any, Literal

# ==========================================
# 2. Data Manipulation & Core
# ==========================================
import numpy as np
import pandas as pd

# ==========================================
# 3. Visualization & Utilities
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# ==========================================
# 4. Machine Learning: Scikit-Learn Core
# ==========================================
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, f_regression, SelectKBest
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# 5. Machine Learning: Metrics
# ==========================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, mean_absolute_error, root_mean_squared_error, r2_score, confusion_matrix)

# ==========================================
# 6. Machine Learning: Models
# ==========================================
# Linear Models
from sklearn.linear_model import LogisticRegression, Ridge
# Trees & Ensembles
from sklearn.ensemble import StackingClassifier, StackingRegressor
# Extras
from sklearn.base import clone
# Pipeline stuffs.
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# ==========================================
# 7. Global Configuration
# ==========================================
# Ignore convergence warnings.
warnings.filterwarnings("ignore", category=ConvergenceWarning)



class ModelTrainer:
    """
    Manages Training, Evaluation (Metrics), and Hyperparameter Optimization.
    """
    def __init__(self, dict_train: Dict[str, pd.DataFrame], dict_test: Dict[str, pd.DataFrame], problem_type: Literal["classification", "regression"], config_path: str = None) -> None:
        self.dict_train = dict_train
        self.dict_test = dict_test

        if problem_type != "classification" and problem_type != "regression":
            raise ValueError("❌ Error: 'problem_type' value should be 'classification' or 'regression'")

        self.problem_type = problem_type
        self.models = {}

        if config_path:
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                self.hyperparameter_grids = full_config[self.problem_type]
        else:
            self.hyperparameter_grids = None

        self.best_model_name = None
        self.best_dataset_name = None
        self.result_df = None

    def add_model(self, model: str, model_object) -> None:
        self.models[model] = model_object

    def train_all_configs(self, overfitting_filter: bool = False, gap_threshold: float = 0.30) -> pd.DataFrame:
        '''
        Method to train all the datasets with all the models found in self.models.

        Args:
            overfitting_filter (bool): If True, discards models with High Variance (Gap > threshold).
            gap_threshold (float): Max allowed difference between Train and Test Score.

        Returns:
            - Dataframe with all performance information.
        '''

        # Make a working copy of the data.
        train_data = self.dict_train.copy()
        test_data = self.dict_test.copy()

       # Extract the target data from the dictionaries and pop them out so there'll be only predictive data.
        y_test = test_data["y_test"]
        test_data.pop("y_test")
        y_train = train_data["y_train"]
        train_data.pop("y_train")

        print(f"\n{'-'*15} MODEL TRAINING & EVALUATION {'-'*15}")
        print(f"🏋️  Training {len(self.models)} models across {len(train_data)} datasets...")

        # Create an empty list to record all the performance data.
        results = []

        # Loop through all the datasets available and through all the models provided to train each dataset with each model.
        for dataset, data in train_data.items():
            for name, model in self.models.items():
                model = model
                model.fit(data, y_train)
                y_train_pred = model.predict(data)
                test_key = dataset.replace("train","test") # Get the test keys by using the train ones.
                if test_key not in test_data: continue
                y_test_pred = model.predict(test_data[test_key])

                # If the problem is set to 'classification' it'll calculate the accuracy, if not ('regression') RMSE, MAE and R2.
                if self.problem_type == "classification":
                    # Get both the train_score and test_score
                    train_score = accuracy_score(y_train, y_train_pred)
                    test_score = accuracy_score(y_test, y_test_pred)
                    results.append({
                                    'dataset': dataset,
                                    'model': name,
                                    'type': self.problem_type,
                                    'metric': "Accuracy",
                                    'train_score': train_score,
                                    'test_score': test_score})

                else:
                    train_score = r2_score(y_train, y_train_pred)
                    test_score = r2_score(y_test, y_test_pred)
                    results.append(
                        {
                            'dataset': dataset,
                            'model': name,
                            'type': self.problem_type,
                            'Coef': getattr(model, 'coef_', getattr(model, 'feature_importances_', None)),
                            'RMSE': round(root_mean_squared_error(y_test, y_test_pred),2),
                            'train_score': train_score,
                            'test_score': test_score,
                            'R2_score': test_score,
                        }
                    )

        # Create a DF with all the data and return it sorted.
        results_df = pd.DataFrame(results)

        if overfitting_filter:
            final_df = self.filter_results(results_df, gap_threshold)
            self.result_df = final_df
        else:
            sort_metric = "test_score" if self.problem_type == "classification" else "R2_score"
            final_df = results_df.sort_values(by=sort_metric, ascending=False)
            self.result_df = final_df

        if not final_df.empty:
            self.best_model_name = final_df.iloc[0]['model']
            self.best_dataset_name = final_df.iloc[0]['dataset']
            best_score = final_df.iloc[0]['test_score']

            print(f"🏆 Winner: '{self.best_model_name}' on '{self.best_dataset_name}' (Test Score: {best_score:.4f})")

        return final_df


    def filter_results(self, df: pd.DataFrame, gap_threshold: float = 0.30) -> pd.DataFrame:
        """
        Applies anti-overfitting filter, by comparing the gap between the test_score and train_score.

        Args:
            gap_threshold: acceptable gap between both scores. If the score is higher, the model is discarded.
        """

        # If dataset not given, generates it
        if df is None or df.empty:
            print("Warning: Empty DataFrame provided to filter.")
            return df

        filt_df = df.copy()

        # Create the new gap column
        filt_df['gap'] = (df['train_score'] - df['test_score']).abs()

        # Anti-Overfitting filter
        safe_df = filt_df[filt_df['gap'] < gap_threshold]

        if safe_df.empty:
            print("⚠️ Warning: All models have excessive overfitting. Turning back to original DF.")
            return df.sort_values(by="test_score", ascending=False)

        return safe_df.sort_values(by="test_score", ascending=False)

    def optimize_model(self, model_name: str  = None, dataset_name: str = None, trim_models: bool = False, param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Takes a Dataset, a model and applies the hyperparameter optimization.
        It'll print the best results and the hyperparameters used to get them.
        Also updates the self.models.

        Args:
            model_name: the best model used for the training.
            param_grid: a dictionary with the parameters to iterate through with GridSearch.
            dataset_name: the dataset with the best results.
            trim_models: after optimization, just keep the best model.

        Returns:
            Dict with the best hyperparameters.
        """

        # Try to initialize automatically.

        if model_name is None: 
            model_name = self.best_model_name
        if dataset_name is None: 
            dataset_name = self.best_dataset_name
            
        if model_name is None or dataset_name is None:
            raise ValueError("❌ Error: No base model/dataset selected. Run train_all_configs() first.")

        print(f"\n{'-'*15} HYPERPARAMETER OPTIMIZATION {'-'*15}")
        print(f"🔧 Optimizing '{model_name}' using dataset '{dataset_name}'...")


        # Checks if the information given by the user is alright.

        if model_name in self.models.keys():
            model = self.models[model_name]
        else:
            raise ValueError(f"❌ Error: No '{model_name}' found. Please check the model loading is correct and the name input.")

        if dataset_name in self.dict_train.keys():
            train_data = self.dict_train[dataset_name]
        else:
            raise ValueError(f"❌ Error: No '{dataset_name}' found. Please check the input data.")

        if param_grid == None:
            if model_name in self.hyperparameter_grids:
                param_grid = self.hyperparameter_grids[model_name]
                print("Automatic hyperparameter configuration loaded.")
            else:
                raise ValueError(f"No '{model_name}' configuration found. Please check the input data.")

        # Extracts y_train from the dict of datasets.
        y_train = self.dict_train["y_train"]

        # Run the GridSearch.
        metric = "accuracy" if self.problem_type == "classification" else "r2" # Let it choose depending the type of problem.
        grid = GridSearchCV(model, param_grid, scoring = metric, cv = 5)
        grid.fit(train_data, y_train)


        new_model_name = f"{model_name}_optimized"

        # Update the model with the best Hyperparameters.
        self.models[new_model_name] = grid.best_estimator_
        print(f"\n✅ Optimization Complete:")
        print(tabulate(grid.best_params_.items(), headers=["Hyperparam", "Best Value"], tablefmt="fancy_grid"))
        print(f"📈 Best CV Score: {grid.best_score_:.4f}")

        if trim_models:
            self.trim_model_list(model_name=new_model_name)


        return grid.best_params_

    def trim_model_list(self, model_name) -> None:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")

        # Overwrite the models with only the selected.
        self.models = {model_name: self.models[model_name]}
        print(f"🧹 Trim: Removed all other models. Keeping only '{model_name}'.")

    def save_best_model(self, model_name: str = None, dataset_name: str = None, folder_path: str = "./models", scaler=None, selector=None) -> None:
        """
        Select the best model (based either on the pre-selected on .train_all_configs()) or by user input.

        Args:
            model_name: the best model used for the training. If not given, it'll take self.best_model_name instead.
            scaler: MinMaxScaler/StandardScaler used in the dataset (if any used)
            folder_path: directory to save the pickle file.
            selector: Selector used in the dataset (if any used)
            dataset_name: the dataset with the best results. If not given, it'll take self.best_dataset_name instead.

        """
        print(f"\n{'='*15} 💾 MODEL EXPORT {'='*15}")

        # Initialize automatic data fetching

        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("❌ Error: No trained model found. You must run train_all_config() first.")
            model_name = self.best_model_name
            print(f"Best model found, saving: '{model_name}'...")

        if dataset_name is None:
            if self.best_dataset_name is None:
                raise ValueError("❌ Error: No dataset name found.")
            dataset_name = self.best_dataset_name
            print(f"Best dataset found, saving: '{dataset_name}'...")

        # Stacking logic

        is_stacking = (model_name == 'Stacking_Ensemble')

        if is_stacking:
            print("🧩 Stacking Ensemble detected. Adapting export logic...")
            # Stacking needs the raw base dataset, not the scaled one.
            if "without_outliers" in dataset_name:
                dataset_name = "X_train_without_outliers"
            else:
                dataset_name = "X_train_with_outliers"
                
            print(f"📂 Base dataset resolved to: '{dataset_name}'")
            print("🚫 External Scalers/Selectors will be ignored (already baked into Stacking Pipelines).")
            scaler = None
            selector = None
            
        else:
            print(f"📂 Dataset used: '{dataset_name}'...")
            # Decide which scaler save with the model if a dict was given.
            if isinstance(scaler, dict):
                if "minmax" in dataset_name:
                    scaler = scaler.get("minmax")
                elif "standard" in dataset_name:
                    scaler = scaler.get("standard")
                else:
                    scaler = None
                    print("⚠️ Dataset without scaler detected. No scaler will be saved.")

        # Validate input data

        if model_name not in self.models.keys():
            raise ValueError(f"❌ Error: No '{model_name}' found. Please check the model loading is correct.")
        if dataset_name not in self.dict_train.keys():
            raise ValueError(f"❌ Error: No '{dataset_name}' found. Please check the input data.")

        # Load the data to train
        X_train = self.dict_train[dataset_name]
        test_key = dataset_name.replace("train","test")
        X_test = self.dict_test[test_key]
        y_train = self.dict_train["y_train"]
        y_test = self.dict_test["y_test"]

        # Concat to train with the whole data (100% data)
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])

        # Save the Features
        features = X_full.columns.tolist()

        # Train the model on 100% of the data
        model = self.models[model_name]
        print(f"⏳ Retraining '{model_name}' on 100% of data (Train + Test)...")
        if is_stacking:
             print("   (This will take a moment due to internal Cross-Validation)")
             
        model.fit(X_full, y_full)


        # Initialize the "artifact", the container with all the information to save.
        artifact = {
            "model_name": model_name,
            "model": model,
            "features": features
        }
        if scaler:
            artifact["scaler"] = scaler
        if selector:
            artifact["selector"] = selector

        # Save the artifact.
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{model_name}_best.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
            print(f"✅ 📦 Artifact successfully saved at: {file_path}")

    def retrain_model(self, model_path: str, new_dataset_name: str) -> None:
        """
        Loads a model previously trained to train it with new data saved on self.dict_train.

        Args:
            model_path: Path to the model .pkl (ej: './models/best_model_RandomForest.pkl').
            new_dataset_name: Name of the new dataset (loaded on self.dict_train).
        """

        # Validate data
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Error: File not found: {model_path}")

        if new_dataset_name not in self.dict_train:
            raise ValueError(f"❌ Error: Dataset '{new_dataset_name}' not loaded on Trainer.")

        # Load the Artifact (Dict)
        print(f"🔄 Loading Artifact: {model_path}...")
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)

        # Check the integrity of the artifact.
        if isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
            old_features = artifact.get("features", [])
            model_name = artifact.get("model_name", "Retrained_Model")
        else:
            # Fallback in case the artifact only have the model.
            model = artifact
            old_features = []
            model_name = "Retrained_Model"
            print("⚠️ Warning: The loaded model is alone (no scaler or features)")

        # Fetch new data.
        X_new = self.dict_train[new_dataset_name]
        y_train = self.dict_train["y_train"]

        # Check if the features match
        if old_features:
            current_features = X_new.columns.tolist()
            if current_features != old_features:
                # X_new = X_new[old_features]
                raise ValueError(f"❌ Feature Mismatch: The number of columns are not the same.\nExpected: {old_features[:5]}...\nGiven: {current_features[:5]}...")
            else:
                print("Column validation succesful")

        # Retraining.
        print(f"🏋️ Re-training'{model_name}' with {len(X_new)} new samples...")
        model.fit(X_new, y_train)

        # Updating the model database.
        self.models[model_name] = model
        print("✅ Retraining Complete. Model updated in memory.")


    def evaluate_model(self, model_name: str  = None, dataset_name: str = None)-> None:
        """
        Retrains the selected model, calculates detailed metrics, 
        and plots the performance (Confusion Matrix or Regression Plot).
        """

        # Try to initialize automatically

        if model_name is None:
            if self.best_model_name is None:
                raise ValueError("❌ Error: No models trained yet. Run train_all_configs() first.")
            model_name = self.best_model_name

        if dataset_name is None:
            if self.best_dataset_name is None:
                raise ValueError("❌ Error: No best dataset found.")
            train_key = self.best_dataset_name
            test_key = train_key.replace("train", "test")
        else:

            if "train" in dataset_name:
                train_key = dataset_name
                test_key = dataset_name.replace("train", "test")
            elif "test" in dataset_name:
                test_key = dataset_name
                train_key = dataset_name.replace("test", "train")
            else:
                raise ValueError("❌ Error: Dataset name must contain 'train' or 'test' to identify the pair.")

        print(f"\n{'-'*15} 📊 DETAILED EVALUATION REPORT {'-'*15}")
        print(f"🔎 Model: '{model_name}'")
        print(f"📂 Dataset Pair: '{train_key}' / '{test_key}'")

        # Checks if the information given by the user is alright.

        if model_name not in self.models:
            raise ValueError(f"❌ Error: Model '{model_name}' not found.")
        
        if train_key not in self.dict_train:
             raise ValueError(f"❌ Error: Training data '{train_key}' not found.")
             
        if test_key not in self.dict_test:
             raise ValueError(f"❌ Error: Test data '{test_key}' not found.")
        
        model = self.models[model_name]
        X_train = self.dict_train[train_key]
        y_train = self.dict_train['y_train']
        X_test = self.dict_test[test_key]
        y_test = self.dict_test['y_test']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        print(f"\n{'-'*5} 📉 Numerical Metrics {'-'*5}")

        if self.problem_type == 'classification':

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            class_metrics = [
                ["Accuracy",  f"{acc:.4f}"],
                ["Precision (Weighted)", f"{prec:.4f}"],
                ["Recall (Weighted)",    f"{rec:.4f}"],
                ["F1 Score (Weighted)",  f"{f1:.4f}"]
            ]
            
            print("\n" + tabulate(class_metrics, headers=["Metric", "Value"], tablefmt="fancy_grid"))
            print("\n📋 Classification Report:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d',  cmap='Blues', cbar=False)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f"Confusion Matrix of {model_name}", fontsize=14)
            plt.tight_layout()
            plt.show()

        else:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            try:
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                mape_str = f"{mape:.2f}%"
            except:
                mape_str = "N/A (Zeros in target)"

            metrics_data = [
            ["R2 Score (Explained Variance)", f"{r2:.4f}"],
            ["MAE (Mean Absolute Error)",    f"{mae:.4f}"],
            ["RMSE (Root Mean Sq. Error)",   f"{rmse:.4f}"],
            ["MAPE (Mean Abs. % Error)",     mape_str]]

            print("\n" + tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

            plt.figure(figsize=(7, 6))
            sns.regplot(x=y_test, y=y_pred, 
                        line_kws={"color": "red", "label": "Perfect Fit"}, 
                        scatter_kws={"alpha": 0.5, "color": "blue"})
            
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.legend()
            plt.title(f"Regression Plot (Actual vs Pred): {model_name}", fontsize=12, weight='bold')
            plt.show()

    def _pipeline_from_config(self, dataset_name: str, model_instance)-> Pipeline:
        """
        Make reverse engineering using the standarized names of the models to extract the characteristics
        needed to make a pipeline to make an stacking ensemble.
        """
        
        steps = []

        if "minmax" in dataset_name:
            steps.append(MinMaxScaler())
        elif "standard" in dataset_name:
            steps.append(StandardScaler())

        if "sel" in dataset_name:
            score_func = f_regression if self.problem_type == 'regression' else f_classif
            steps.append(SelectKBest(score_func=score_func, k=5))

        steps.append(model_instance)

        return make_pipeline(*steps)


    def build_stacking_ensemble(self, top_n: int = 3, evaluate: bool =True, max_gap: float = 0.15)-> None:
        
        print(f"\n{'='*15} 🏗️ BUILDING STACKING ENSEMBLE {'='*15}")
        # Load and filter data.
        results_df = self.result_df.copy()

        # Apply yet another, stronger, overfiting filter.
        healthy_models = results_df[results_df['gap'] <= max_gap]

        # In case ALL models are disbalanced, it doesn't apply the filter.
        if len(healthy_models) < top_n:
            print(f"⚠️ Warning: Not enough models with gap <= {max_gap}. Relaxing overfitting filter...")
        else:
            results_df = healthy_models
            print(f"🛡️ Overfitting filter applied: Kept models with gap <= {max_gap}")

        # Create a temporal column to unify data (so the optimized versions of the model and the base ones will be treated the same).
        results_df['base_algorithm'] = results_df['model'].str.replace('_optimized', '')
        
        # Apply the filter in here.
        filtered_rdf = results_df.drop_duplicates(subset='base_algorithm').head(top_n)
        
        # Drop the support column to keep things clean.
        filtered_rdf = filtered_rdf.drop(columns=['base_algorithm'])

        # UX Print: to show user the models participating
        selected_models = filtered_rdf['model'].tolist()
        print(f"🥇 Selected Top {top_n} unique models: {', '.join(selected_models)}")

        # "Voting" system to determine if the master data will be with or without outliers.
        with_outliers = filtered_rdf['dataset'].str.contains("with_outliers").sum()
        without_outliers = len(filtered_rdf) - with_outliers

        print(f"🗳️ Outliers Voting: {with_outliers} (With) vs {without_outliers} (Without)")

        # Decide which master datasets will be used to train.
        if with_outliers > without_outliers:
            train_key = "X_train_with_outliers"

        else:
            train_key = "X_train_without_outliers"


        X_train_master = self.dict_train[train_key]
        # Get the target data.
        y_train = self.dict_train['y_train']

        # Create an estimator list with all the pipelines needed for the training.
        estimators = []

        for _, row in filtered_rdf[["dataset","model"]].iterrows():

            model_name = row["model"]
            dataset_name = row["dataset"]
            original_model = self.models[model_name]
            model_copy = clone(original_model) # Make a copy of the model to not re-train the original ones.
            pipeline =  self._pipeline_from_config(dataset_name=dataset_name, model_instance=model_copy)
            
            # Add the estimator in the form of a tuple. That way the Stacking models will recognize the data.
            estimators.append((model_name, pipeline))

        # Train the Stacking model with the extracted information.
        print("⏳ Training Meta-Learner (This involves internal Cross-Validation and may take a while)...")

        if self.problem_type == 'classification':
            stacking = StackingClassifier(estimators=estimators, passthrough=True, 
                                          final_estimator=LogisticRegression(), n_jobs=-1)
            stacking.fit(X_train_master, y_train)
            self.models['Stacking_Ensemble'] = stacking

        else:
            stacking = StackingRegressor(estimators=estimators, passthrough=True, 
                                         final_estimator=Ridge(), n_jobs=-1)
            stacking.fit(X_train_master, y_train)
            self.models['Stacking_Ensemble'] = stacking

        print("🎉 Stacking Ensemble successfully trained and saved into memory!")
        if evaluate:
            self.evaluate_model(model_name='Stacking_Ensemble', dataset_name=train_key)