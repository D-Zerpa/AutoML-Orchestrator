# ==========================================
# 1. Standard Python Libraries
# ==========================================
import os
import sys
import json
import math
import pickle
import warnings
from typing import Optional, List, Dict, Any, Literal

# ==========================================
# 2. Data Manipulation & Core
# ==========================================
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# ==========================================
# 3. Visualization & Utilities
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from IPython.display import display

# ==========================================
# 4. Machine Learning: Scikit-Learn Core
# ==========================================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# 5. Machine Learning: Metrics
# ==========================================
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score, 
    accuracy_score, 
    root_mean_squared_error
)

# ==========================================
# 6. Machine Learning: Models
# ==========================================
# Linear Models
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
# Trees & Ensembles
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Support Vector Machines
from sklearn.svm import SVR, SVC
# Gradient Boosting (External)
import xgboost as xgb

# ==========================================
# 7. Global Configuration
# ==========================================
# Ignore convergence warnings.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class DataExplorer:
    """
    Handles the initial inspection, type classification, and statistical summary of the dataset.
    """
    def __init__(self, df: pd.DataFrame, target: str):

        self.df = df.copy() # Loads the Dataframe and makes a copy to not mess with the original one.
        self.num_cols = []
        self.cat_cols = []
        self.target = target
        self.identify_types() # Fill the column list directly to not have the chicken-egg situation on the screening.


    def identify_types(self, cat_threshold: int = 10, force_cat: Optional[List[str]] = None) -> None:
        """
        Classifies columns into Numerical or Categorical based on data types and cardinality.

        Args:
            cat_threshold (int): If a numeric column has fewer unique values than this, it's treated as categorical.
            force_cat (List[str]): List of column names to strictly treat as categorical.
        """

        self.num_cols = []
        self.cat_cols = []
        force_cat = force_cat if force_cat else []

        for column in self.df.columns:

            if column in force_cat:
                self.cat_cols.append(column) # Forceful override.
                print(f"{column} forced as a Categorical.")
                continue

            # Automatic logic.
            if is_numeric_dtype(self.df[column]):
                unique_count = self.df[column].nunique() # Unique count to determine if surpasses the threshold.

                if unique_count <= cat_threshold:
                    print(f"Looks like {column} is a factorized categorical. Reclassifying...")
                    self.cat_cols.append(column) # Does not surpass, then might be categorical.
                else:
                    self.num_cols.append(column) # Surpass, then it's truly numerical.
            else:
                self.cat_cols.append(column)

        print(f"ℹ️  Column Classification: {len(self.num_cols)} Numerical | {len(self.cat_cols)} Categorical")


    def data_summary(self)-> None:
        """Prints a comprehensive summary of the dataset dimensions, types, and health (nulls/dupes)."""

        # Gathering
        rows, columns = self.df.shape # Dimensions
        null_vars = self.df.isnull().sum().loc[lambda x: x > 0] # Nulls
        duplicated_values = self.df.duplicated().sum() #Dupes

        # Showing
        print(f"\n{'-'*15} DATA SUMMARY {'-'*15}")
        print(f"📐 Dimensions:  {rows} Rows x {columns} Columns")
        print(f"📊 Data Types:  {len(self.cat_cols)} Categorical, {len(self.num_cols)} Numerical")

        if null_vars.empty:
            print("✅ Nulls:       None detected.")
        else:
            print(f"⚠️ Nulls:       Found in {len(null_vars)} columns.")
            
        if duplicated_values > 0:
            print(f"⚠️ Duplicates:  {duplicated_values} rows (Recommendation: Drop them).")
        else:
            print("✅ Duplicates:  None detected.")
        print("-" * 44)

    def univariate_analysis(self)-> None:
        """Prints statistical descriptions for numerical columns and frequency tables for categorical ones."""

        print(f"\n{'-'*15} UNIVARIATE ANALYSIS {'-'*15}")
        if self.num_cols:
            print(f"🔢 Numerical Statistics:")
            print(tabulate(self.df[self.num_cols].describe().T, headers="keys", tablefmt="simple", floatfmt=".2f"))
            print("-" * 44)

        if self.cat_cols:
            print(f"🔠 Categorical Frequencies (Top 5):")
            for col in self.cat_cols[:5]: # Show max 5 to avoid spamming console
                print(f"• {col}: {self.df[col].unique().tolist()[:10]}...") 
            print("-" * 44)


class Visualizer:
    """
    Handles all plotting logic using Matplotlib and Seaborn.
    Depends on a DataExplorer instance to know which columns are which.
    """
    def __init__(self, explorer: DataExplorer) -> None:
        self.explorer = explorer


    def plot_univariate_analysis(self) -> None:
        """Wrapper to generate both numerical and categorical plots."""
        print(f"\n🎨 Generating Univariate Plots...")
        self.plot_numerical_distribution()
        self.plot_categorical_distribution()



    def plot_numerical_distribution(self, n_cols: int = 2) -> None:
        """
        Plot histplot + boxplot for numerical columns.

        Args:
            n_cols: Number of wanted columns for the grid.
        """

        # Calculate the ammount of numerical columns to plot. Return nothing if they're 0.
        n_vars = len(self.explorer.num_cols)
        if n_vars == 0:
            return

        # Make a divide + ceil to calculate the ammount of rows the figure will have.
        n_rows = math.ceil(n_vars / n_cols)

        # Create the canvas, scaling wih the ammount of cols and rows.
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
        fig.suptitle("Numerical values Distribution", fontsize=16, weight='bold')
        subfigs = fig.subfigures(n_rows, n_cols).flatten()

        plot_df = self.explorer.df.copy() #We make a work copy, because we'll have to modify data at some point.
        total_stats = plot_df.describe() # Set a describe to get the limit's info.

        for index, column in enumerate(self.explorer.num_cols):
            subfig = subfigs[index]


            # We calculate the upper limit to have a top in case there's ouliers that makes hard to visualize data.
            if column in total_stats.columns:
                stats = total_stats[column]
                iqr = stats["75%"] - stats["25%"]
                upper_limit = stats["75%"] + (2.0 * iqr)
                plot_df[column] = plot_df[column].clip(upper=upper_limit)
                capped_info = f"(Capped @ {upper_limit:.2f})" # We show if the limits were clipped.
            else:
                capped_info = ""

            # We use the subfig as an independent figure.
            axs = subfig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1]})
            ax_hist, ax_box = axs[0], axs[1]
            subfig.suptitle(f"{column} {capped_info}", fontsize=10)

            # Plots
            sns.histplot(ax=ax_hist, data=plot_df, x=column, kde=True, color='skyblue')
            sns.boxplot(ax=ax_box, data=plot_df, x=column, orient='h', color='salmon')

            # Clean a bit.
            ax_box.set_xlabel("")
            ax_hist.set_ylabel("")

        for i in range(n_vars, len(subfigs)): # Clean the extra ones.
            subfigs[i].set_visible(False)

        plt.show()



    def plot_categorical_distribution(self, n_cols: int = 3) -> None:
        """
        Plot countplots for categorical columns.

        Args:
            n_cols: Number of wanted columns for the grid.
        """
        # Calculate the ammount of numerical columns to plot. Return nothing if they're 0.
        n_vars = len(self.explorer.cat_cols)
        if n_vars == 0:
            return
        # Make a divide + ceil to calculate the ammount of rows the figure will have.
        n_rows = math.ceil(n_vars / n_cols)

        # Simple grid.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, column in enumerate(self.explorer.cat_cols):
            ax = axes[i]
            order = self.explorer.df[column].value_counts().index
            sns.countplot(ax=ax, data=self.explorer.df, x=column, order=order, hue= column)
            ax.set_title(column)
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel("")

        for i in range(n_vars, len(axes)): # Clean the extra axis.
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, factorize: bool = True) -> None:
        """
        Factorize and make a correlation matrix for all the data.

        Args:
            factorize: If, for any reason we don't want to factorize, put False.
        """

        plot_df = self.explorer.df.copy()

        # If there are categorical cols and we want every column to appear in the correlation matrix, we have to factorize.
        if factorize:
            for col in plot_df.select_dtypes(include=['object', 'category']).columns:
                plot_df[col], _ = pd.factorize(plot_df[col])

        plt.figure(figsize=(10,7))
        sns.heatmap(plot_df.corr(), annot = True, fmt = ".2f", cmap= "RdBu", mask = np.triu(np.ones_like(plot_df.corr(), dtype=bool)))
        plt.title("Correlation Matrix (Peason)", fontsize=14)
        plt.tight_layout()
        plt.show()


    def plot_top_correlations(self, k: int = 3)-> None:
        """Plots scatter plots for the features most correlated with the target."""
        plot_df = self.explorer.df.copy()
        target = self.explorer.target

        # If there are categorical cols and we want every column to appear in the correlation matrix, we have to factorize.
        for col in plot_df.select_dtypes(include=['object', 'category']).columns:
            plot_df[col], _ = pd.factorize(plot_df[col])

        if target not in plot_df.columns:
            print(f"⚠️ Error: Target '{target}' not found or not numerical.")
            return

        # Get the correlations.
        matrix = plot_df.corr()

        # Drop the target row.
        target_corr = matrix[target].drop(target)

        # Order by absolute value.
        top_vars = target_corr.abs().sort_values(ascending= False).head(k).index.tolist()

        # Subplot definitions.
        fig, axes = plt.subplots(1, k, figsize=(5 * k, 5), constrained_layout=True, squeeze=False)
        axes = axes.flatten()

        for i, var_name in enumerate(top_vars):
            ax = axes[i]
            corr_value = target_corr[var_name]

            # Plot the scatterplot with regression line.
            sns.regplot(
                ax=ax,
                data=plot_df,
                x=var_name,
                y=target,
                line_kws={"color": "red"},
                scatter_kws={"alpha": 0.5})

            ax.set_title(f"{var_name} (Corr: {corr_value:.2f})", fontsize=11, weight='bold')

        plt.show()

class FeatureEngineer:
    """
    Handles Missing Values, Encoding (Factorization), and Outlier Treatment.
    Produces two versions of the dataset: With and Without Outliers.
    """

    def __init__(self, explorer: DataExplorer, save_path: str = ".") -> None:

        self.explorer = explorer
        self.save_path = save_path
        self.df_with_outliers = None
        self.df_without_outliers = None


    def auto_process_data(self, save_encoder_data: bool = True) -> tuple:
        """
        Applies both the prefactorization and outliers replacement logics.

        Args:
            save_encoder_data: True if you want to save the encoder from the factorization.

        Returns:
            A Tuple with two dataframes: one with outliers and one without outliers.
        """

        print(f"{'='*15} SETTING FEATURE ENGINEERING {'='*15}")
        self.handle_nulls()
        self.pre_factorize_data(save_data=save_encoder_data)
        self.replace_outliers_iqr()

        return self.df_with_outliers, self.df_without_outliers


    def pre_factorize_data(self, save_data: bool) -> None:
        """Converts categorical text columns into numbers (Factorize) and optionally saves the map."""
        factorized_df = self.explorer.df.copy() # Make a clean copy to work with.

        counter = 0

        # We make sure the folder is created.
        json_dir = os.path.join(self.save_path, "encoders")
        os.makedirs(json_dir, exist_ok=True)

        for col in self.explorer.cat_cols:
            codes, uniques = pd.factorize(factorized_df[col]) # Pandas Factorize returns a tuple of (code, unique values)
            if save_data:
                rules = dict(zip(uniques, range(len(uniques)))) # We create the encoding.
                rules = {str(k): int(v) for k, v in rules.items()} # Set the keys to str to make sure it's compatible with JSON.

                save_file = os.path.join(json_dir, f"{col}_encoder.json")
                with open(save_file, "w") as f:
                    json.dump(rules, f, indent=4)

            factorized_df[col] = codes
            counter += 1


        print(f"🔢 Encoding:    Factorized {counter} categorical columns.")
        if save_data:
            print(f"💾 Encoders:    Saved {counter} JSON files in '{json_dir}'")

        self.df_with_outliers = factorized_df


    def replace_outliers_iqr(self, multiplier: int = 1.5)-> None:
        """Caps numerical outliers using the IQR method (Winsorization)."""
        # As the factorized data is needed, we make a check.
        if self.df_with_outliers is None:
            raise ValueError("❌ Error: Factorized Dataset needed. You must run .pre_factorize_data() method first.")

        new_df = self.df_with_outliers.copy() # Make a clean copy to work with.
        counter = 0
        for column in new_df:
            # As the target is not touched, we add an exclusion.
            if column in self.explorer.num_cols and column != self.explorer.target:
                col_stats = new_df[column].describe() # Get the data with describe.
                col_iqr = col_stats["75%"] - col_stats["25%"] # Calculate IQR
                upper_limit = round(float(col_stats["75%"] + multiplier * col_iqr), 2)
                lower_limit = round(float(col_stats["25%"] - multiplier * col_iqr), 2)

                # Workaround the zeroes.
                if new_df[column].min() >= 0:
                    lower_limit = max(0, lower_limit)

                new_df[column] = new_df[column].clip(lower=lower_limit,upper=upper_limit)
                counter +=1

        print(f"✂️  Outliers:    Processed {counter} numerical columns (IQR Method).")
        self.df_without_outliers = new_df


    def handle_nulls(self, strategy_num: str = "median", strategy_cat: str = "mode") -> None:

            print(f"{"-"*28} Null handling {"-"*29}")

            if self.df_with_outliers is None:
                working_df = self.explorer.df.copy()
            else:
                working_df = self.df_with_outliers

            null_counts = working_df.isnull().sum()
            if null_counts == 0:
                print("✅ Nulls:       No missing values detected.")
                self.df_with_outliers = working_df
                return

            print(f"🛠️  Fixing Nulls: Using '{strategy_num}' for Num and '{strategy_cat}' for Cat.")

            # Numericals
            for col in self.explorer.num_cols:
                if working_df[col].isnull().sum() > 0:
                    if strategy_num == "median":
                        fill_value = working_df[col].median()
                    elif strategy_num == "mean":
                        fill_value = working_df[col].mean()
                    else:
                        fill_value = 0

                    working_df[col] = working_df[col].fillna(fill_value)
                    print(f" -> Numerical Col '{col}': Filling nuls with {strategy_num} ({fill_value:.2f})")

            # Categoricals
            for col in self.explorer.cat_cols:
                if working_df[col].isnull().sum() > 0:
                    if strategy_cat == "mode":
                        if not working_df[col].mode().empty:
                            fill_value = working_df[col].mode()[0]
                        else:
                            fill_value = "Unknown"
                    else:
                        fill_value = "Unknown"

                    working_df[col] = working_df[col].fillna(fill_value)
                    print(f" -> Categorical col '{col}': Filling nuls with {strategy_cat} ('{fill_value}')")

            # Save state.
            self.df_with_outliers = working_df

class DataPreparer:
    """
    Handles Train/Test Split, Feature Scaling (MinMax/Standard), and Feature Selection.
    """
    def __init__(self, target: str, test_size: float = 0.2, random_state: int = 42) -> None:
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        self.save_path = None


    def auto_prepare_data(self, dataset: pd.DataFrame, label: str, save_path: str = ".", save_data: bool = True, feature_sel: bool = True, k: int = 5) -> tuple:
        """
        Applies all the prepping to ML: Split, Scaling and (optional) feature selection.

        Args:
            label: to differenciate the outcome DF from others (eg. "with_outliers", "without_outliers")
            save_data: turns on/of the saving mode, if turned off, the method will only return.
            feature_sel: turns on/off the feature selection.
            k: K value for Feature Selection.

        Returns:
            Tuple with two dictionaries with Dataframes: One with the train data, one with the test data.
        """
        if save_data:
            self.save_path = save_path

        print(f"{"-"*14} Initiating automatic preparation for: {label} {"-"*14}")

        # Split the dataset and create a master_dict that will contain all datasets.
        X_train, X_test, y_train, y_test = self.split_data(dataset)
        master_dict = {
            "y_train": y_train,
            "y_test": y_test,
            f"X_train_{label}": X_train,
            f"X_test_{label}": X_test}


        # MinMaxScaler
        X_train_mm, X_test_mm = self.min_max_scaling(X_train, X_test, save_scaler=save_data)
        master_dict[f"X_train_{label}_minmax"] = X_train_mm
        master_dict[f"X_test_{label}_minmax"] = X_test_mm
        print(f" -> MinMaxScaler applied... ")

        # StandardScaler
        X_train_standard, X_test_standard = self.standard_scaling(X_train=X_train, X_test=X_test, save_scaler=save_data)
        master_dict[f"X_train_{label}_standard"] = X_train_standard
        master_dict[f"X_test_{label}_standard"] = X_test_standard
        print(f" -> StandardScaler applied... ")


        # Feature Selection (optional)
        if feature_sel:
            print(f" -> Feature selection ON, applying... ")
            selected_cols = self.kselection(X_train=X_train, y_train=y_train, k=k)

            print(f" -> Selected features ({k}): {selected_cols}")
            master_dict[f"X_train_{label}_minmax_sel"] = X_train_mm[selected_cols]
            master_dict[f"X_test_{label}_minmax_sel"] = X_test_mm[selected_cols]

            master_dict[f"X_train_{label}_standard_sel"] = X_train_standard[selected_cols]
            master_dict[f"X_test_{label}_standard_sel"] = X_test_standard[selected_cols]

        # Saving (optional)
        if save_data:
            dataset_dir = os.path.join(save_path, "datasets")
            os.makedirs(dataset_dir, exist_ok=True)
            for k, v in master_dict.items():
                file_path = os.path.join(dataset_dir, f"{k}.csv")
                v.to_csv(file_path, index=False)
            print(f"💾 Datasets:    Saved {len(master_dict)} files for '{label}' in '{dataset_dir}'")

        # Train/Test data split
        train_df = {k: v for k, v in master_dict.items() if "train" in k}
        test_df = {k: v for k, v in master_dict.items() if "test" in k}


        return train_df, test_df


    def split_data(self, df: pd.DataFrame) -> tuple:
        """Splits DataFrame into X and y, then Train and Test."""
        X = df.drop(columns=self.target)
        y = df[self.target]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def min_max_scaling(self, X_train: pd.DataFrame, X_test: pd.DataFrame, save_scaler: bool = True) -> tuple:
        """Applies MinMax Scaling (0-1) and saves the object."""
        scaler = MinMaxScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        self.scalers["minmax"] = scaler

        if save_scaler:
            scalers_dir = os.path.join(self.save_path, "scalers")
            os.makedirs(scalers_dir, exist_ok=True)
            file_path = os.path.join(scalers_dir, "scaler_minmax.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(scaler, f)
                print(f"Scaler (MinMax) saved in: {file_path}")


        return X_train_sc, X_test_sc

    def standard_scaling(self, X_train: pd.DataFrame, X_test: pd.DataFrame, save_scaler: bool = True) -> tuple:
        """Applies Standard Scaling (Z-Score) and saves the object."""
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        self.scalers["standard"] = scaler

        if save_scaler:
            scalers_dir = os.path.join(self.save_path, "scalers")
            os.makedirs(scalers_dir, exist_ok=True)
            file_path = os.path.join(scalers_dir, "scaler_standard.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(scaler, f)
                print(f"Scaler (Standard) saved in: {file_path}")

        return X_train_sc, X_test_sc

    def kselection(self, X_train: pd.DataFrame, y_train: pd.DataFrame, k: int) -> list:
        """Selects top K features using ANOVA F-value."""
        selection_model = SelectKBest(f_classif, k=k)
        selection_model.fit(X_train, y_train)
        cols_idxs = selection_model.get_support()

        return X_train.columns[cols_idxs].tolist()

    def transform_new_data(self, df: pd.DataFrame, method: str = "minmax", scaler=None) -> pd.DataFrame:
        """
        Applies a scaler ruler to a raw dataset previously scaled.

        Args:
            df: raw data DataFrame.
            method: 'minmax' or 'standard'.
            scaler: Optional scaler, if empty, fetch self.scalers.

        Returns:
            DF ready to train.
        """
        print(f"Transforming new data with the method: '{method}'...")

        current_scaler = None

        # Checks if an argument was passed.
        if scaler is not None:
            current_scaler = scaler

        # Checks if there's an scaler on self.
        elif method in self.scalers:
            current_scaler = self.scalers[method]

        # Fallback in case there's no scaler given.
        else:
            raise ValueError(f"❌ Error: No scaler found for '{method}'. "
                             f"Enter an argument or run auto_prepare_data().")

        # Checking if the scaler have the same features as the new DF.
        if hasattr(current_scaler, "feature_names_in_"):
            expected_cols = set(current_scaler.feature_names_in_)
            received_cols = set(df.columns)

            missing = expected_cols - received_cols
            if missing:
                raise ValueError(f"❌ Error: Missing {missing} columns")

            # Reorder the columns.
            df = df[current_scaler.feature_names_in_]

        data_transformed = current_scaler.transform(df)

        # Reconstruct DF.
        df_result = pd.DataFrame(data_transformed, columns=df.columns, index=df.index)

        return df_result

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
                            'Coef': model.coef_,
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

        else:
            sort_metric = "test_score" if self.problem_type == "classification" else "R2_score"
            final_df = results_df.sort_values(by=sort_metric, ascending=False)

        if not final_df.empty:
            self.best_model_name = final_df.iloc[0]['model']
            self.best_dataset_name = final_df.iloc[0]['dataset']
            best_score = final_df.iloc[0]['test_score']

            print(f"🏆 Winner: '{self.best_model_name}' on '{self.best_dataset_name}' (Test Score: {best_score['test_score']:.4f})")

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


        new_model_name = f"{model_name}"
        for k, v in grid.best_params_.items():
            new_model_name += f"_{str(k)}_{str(v)}"

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
            selector: Selector used in the dataset (if any used)
            dataset_name: the dataset with the best results. If not given, it'll take self.best_dataset_name instead.

        """

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


        # Validate input data

        if model_name in self.models.keys():
            model = self.models[model_name]
        else:
            raise ValueError(f"No '{model_name}' found. Please check the model loading is correct and the name input.")

        if dataset_name in self.dict_train.keys():
            X_train = self.dict_train[dataset_name]
        else:
            raise ValueError(f"No '{dataset_name}' found. Please check the input data.")

        # If a dict was given, decide which scaler save with the model.

        if isinstance(scaler, dict):
            if "minmax" in dataset_name:
                scaler = scaler.get("minmax")
            elif "standard" in dataset_name:
                scaler = scaler.get("standard")
            else:
                scaler = None
                print("Dataset without scaler detected. No scaler will be saved.")

        # Load the data to train.
        test_key = dataset_name.replace("train","test")
        X_test = self.dict_test[test_key]
        y_train = self.dict_train["y_train"]
        y_test = self.dict_test["y_test"]

        # Concat to train with the whole data.
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])

        # Save the Features.
        features = X_full.columns.tolist()

        # Train the model.
        model = self.models[model_name]
        model.fit(X_full, y_full)


        # Initialize the "artifact", the container with all the information to save.
        artifact = {}
        artifact["model_name"] = model_name
        artifact["model"] = model
        artifact["features"] = features
        if scaler:
            artifact["scaler"] = scaler
        if selector:
            artifact["selector"] = selector

        # Save the artifact.
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{model_name}_best.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
            print(f"📦 Artifact Saved: {file_path}")

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

class MLOrchestrator:
   
    """
    Central controller for the complete Machine Learning flow.
    
    This class manages communication between the different modules:
    - DataExplorer (EDA)
    - Visualizer (Plots)
    - FeatureEngineer (Cleaning and Transformation)
    - DataPreparer (Scaling and Splitting)
    - ModelTrainer (Training and Optimization)
    
    Allows for step-by-step execution (manual) or automated flow (run_pipeline).
    """



    def __init__(self) -> None:
        # Data status
        self.df_raw = None
        self.target = None
        self.save_path = None

        # Tools
        self.explorer = None
        self.visualizer = None

        # Training Pipeline
        self.feat_engineer = None
        self.preparer = None
        self.trainer = None

        # Generated artifacts
        self.train_dict = None
        self.test_dict = None
        self.scalers = None



    def load_data(self, df_path: str, target: str)-> None:
        """
        Loads the dataset and initializes exploration tools.

        Args:
            df_path (str): Relative or absolute path to the CSV file.
            target (str): Exact name of the target column.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file is empty or the target column is missing.
        """

        # Try to load the data from the path given by user.
        try:
            df_loaded = pd.read_csv(df_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"❌ Error: File not found at '{df_path}'")

        # Basic validations
        if df_loaded.empty:
            raise ValueError("❌ Error: The loaded file is empty.")
        
        if target not in df_loaded.columns:
            raise ValueError(f"❌ Error: Target column '{target}' not found in the dataset.")

        # Updates the needed data into the class configuration.
        self.df_raw = df_loaded
        self.target = target
        print(f"✅ Data successfully loaded!")

        # Initialize Explorer and Visualizer.
        self.explorer = DataExplorer(df=self.df_raw, target=self.target)
        self.visualizer = Visualizer(explorer=self.explorer)
        print("🛠️ Tools initialized: DataExplorer & Visualizer ready.")


    def execute_analysis(self, mode: Literal["visual", "data", "all"]) -> None:
        """
        Makes a whole analysis with user's instructions.

        Args:
            mode (Literal): 
                - 'visual': Generates plots (correlations, distributions).
                - 'data': Generates numerical summaries (describe, info, nulls).
                - 'all': Executes both.
        """

        # Small validation.
        if self.explorer is None:
            raise ValueError("⚠️ You must run .load_data() first.")

        print(f"\n--- 🔍 Starting Analysis (Mode: {mode}) ---")

        # Executes only the raw data preliminar analysis
        if mode in ["data", "all"]:
            self.explorer.data_summary()
            self.explorer.univariate_analysis()

        # Executes the main, relevant plotting methods.
        if mode in ["visual", "all"]:
            self.visualizer.plot_univariate_analysis()
            self.visualizer.plot_top_correlations()
            self.visualizer.plot_correlation_matrix()


    def prepare_features (self, save_path: str = ".", save_data: bool = True, k_features: int = 5) -> None:

        """
        Executes Feature Engineering and prepares dataset variants.
        
        Generates two versions of the data:
        1. 'with_outliers': Cleaned data but preserving outliers.
        2. 'without_outliers': Data with outlier treatment (Capping).
        
        Args:
            save_path (str): Root folder where encoders and processed datasets will be saved.
            save_data (bool): If True, saves intermediate CSVs to disk.
            k_features (int): Number of features to select using SelectKBest.
        """

        # Set the root path configuration
        self.save_path = save_path

        # Initialize the FeatureEngineer and DataPreparer tools.
        self.feat_engineer = FeatureEngineer(explorer=self.explorer, save_path=self.save_path)
        self.preparer = DataPreparer(target=self.target)

        # Run .auto_process_data() to generate the two dataset variants.
        df_w_out, df_wo_out = self.feat_engineer.auto_process_data(save_encoder_data=save_data)

        # Initialize a dictionary with both datasets, so we can iterate through them.
        datasets= {"with_outliers": df_w_out, "without_ouliers": df_wo_out}

        # Create empty dicts, so the data created by the iteration remains.
        merged_train = {}
        merged_test = {}
        merged_scalers = {}

        print(f"\n⚙️  Processing Variants...")
        # Iterate through the datasets and apply the .auto_prepare_data() that will fill our dicts.
        for key, dataset in datasets.items():
            train, test = self.preparer.auto_prepare_data(dataset=dataset,
                                                          label=key, save_data=save_data,
                                                          save_path= self.save_path, k=k_features)
            merged_train.update(train)
            merged_test.update(test)

            if hasattr(self.preparer, 'scalers'):
                 merged_scalers.update(self.preparer.scalers)

        # Save all the data into the mainframe
        self.train_dict = merged_train
        self.test_dict = merged_test
        self.scalers = merged_scalers

        print(f"✅ Preparation finished. Generated {len(self.train_dict)-1} training sets.")


    def initialize_trainer(self, problem_type: Literal["classification", "regression"], config_path: Optional[str] = None, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Configures the ModelTrainer and defines which algorithms will compete.

        Args:
            problem_type (Literal): ML problem type ('classification' or 'regression').
            config_path (str, optional): Path to a JSON with hyperparameters.
            model_config (Dict, optional): Manual model dictionary {Name: ModelObject}.
                                           If None, loads a robust default configuration.
        """

        # Make some validations to check if the data is available.
        if self.train_dict is None or self.test_dict is None:
            raise ValueError("⚠️ No training data found. Run .prepare_features() first.")

        # Initialize the ModelTrainer tool
        self.trainer = ModelTrainer(dict_train=self.train_dict, dict_test=self.test_dict, 
                                    problem_type=problem_type, config_path=config_path)

        print(f"\n🥊 --- Initializing ModelTrainer ({problem_type}) ---")
        
        # Check if a config is added by the user, else add defaults.
        if model_config:
            print(f"   -> Loading {len(model_config)} custom models...")
            for model, m_object in model_config.items():
                self.trainer.add_model(model=model, model_object=m_object)

        #By default, 5 models for each problem type.
        else:
            print(f"   -> Loading default configuration for {problem_type}...")
            if problem_type == "classification":
                self.trainer.add_model("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42))
                self.trainer.add_model("RandomForestClassifier", RandomForestClassifier(random_state=42))
                self.trainer.add_model("XGBClassifier", xgb.XGBClassifier(random_state= 42))
                self.trainer.add_model("LogisticRegression", LogisticRegression(random_state=42))
                self.trainer.add_model("SVC", SVC(random_state=42))
            else:
                self.trainer.add_model("RandomForestRegressor", RandomForestRegressor(random_state=42))
                self.trainer.add_model("DecisionTreeRegressor",DecisionTreeRegressor(random_state=42))
                self.trainer.add_model("Lasso",Lasso(random_state=42))
                self.trainer.add_model("Ridge",Ridge(random_state=42))
                self.trainer.add_model("SVR",SVR())

        print(f"Added models for: {self.trainer.problem_type}\n{list(self.trainer.models.keys())}")

    def run_training_cycle(self, optimize: bool = True, overfitting_filter: bool = True, 
                           gap_threshold: float = 0.3, display_results: bool = True, trim_models: bool = True) -> None:

        """
        Executes the training lifecycle: Selection -> Optimization.

        Args:
            optimize (bool): If True, runs hyperparameter search for the best model.
            overfitting_filter (bool): If True, discards models with gap > gap_threshold.
            gap_threshold (float): Max tolerated difference between Train and Test score.
            display_results (bool): Shows result tables in console/notebook.
            trim_models (bool): If True, deletes losing models from memory to save RAM.
        """

        # Make some validations to check if the data is available.
        if self.trainer is None:
            raise ValueError("⚠️ Trainer not initialized. Run .initialize_trainer() first.")

        # First training to obtain the best Dataset.
        first_training = self.trainer.train_all_configs(overfitting_filter=overfitting_filter, gap_threshold=gap_threshold)
        
        # Optional result display.
        if display_results:
            print("\n📊 Preliminary Results:")
            display(first_training)

        # Optimization.
        if optimize:
            self.trainer.optimize_model(trim_models=trim_models)
            second_training = self.trainer.train_all_configs(overfitting_filter=overfitting_filter, gap_threshold=gap_threshold)
            
            # Second training display
            if display_results:
                print("\n📊 Final Optimized Results:")
                display(second_training)

    def save_artifacts(self) -> None:
        """
        Persistence: Saves the best model and its associated scalers.
        
        Automatically creates the folder structure:
        /project_root/models/
        """
        # Validate the trainer.
        if self.trainer is None:
            raise ValueError("No trained model found. Please run run_training_cycle() first.")

        project_root = self.save_path if self.save_path else "."
        models_dir = os.path.join(project_root, "models")

        print(f"\n💾 --- Saving Production Artifacts ---")
        self.trainer.save_best_model(
            folder_path=models_dir,
            scaler=self.scalers)
        
    def run_pipeline(self, df_path: str, target: str, problem_type: Literal["classification", "regression"],
                     save_path: str = "./ml_project", analysis_mode: Literal["visual", "data", "all"] = "data", config_path: str = None,
                     optimize: bool = True, overfitting_filter: bool = True, gap_threshold: float = 0.3) -> None:
        """
        Runs the whole ML flow (End-to-End).
        
        Args:
            df_path: Path to the .csv file.
            target: Name of the target column.
            problem_type: "classification" or "regression".
            save_path: Project's root folder, all the data will be saved there into individual folders (datasets, encoders, models).
            analysis_mode: Exploratory analysis to show ('visual', 'data', 'all').
            optimize: If True, run the automatic hyperparam optimization.
            overfitting_filter: Activate the filter to drop the results with too much gap between the test_score and train_score (maybe Overfitting).
            gap_threshold: Max gap allowed between Train and test scores. (eg: 0.3).
        """
        
        print(f"\n{'='*60}")
        print(f"🚀 AUTOMATIC PIPELINE RUNNING: {problem_type.upper()}")
        print(f"🎯 Target: '{target}' | 📂 Output: '{save_path}'")
        print(f"{'='*60}")

        # 1. Load and initialize Explorer and Visualizer tool.
        # -----------------------------------------
        self.load_data(df_path=df_path, target=target)

        # 2. Data Analysis (Visual, just Data or both).
        # -----------------------------------------
        self.execute_analysis(mode=analysis_mode)

        # 3. Feature Engineering and Data Preparation.
        # -----------------------------------------
        self.prepare_features(save_path=save_path, save_data=True)

        # 4. Model Trainer configuration.
        # -----------------------------------------
        self.initialize_trainer(problem_type=problem_type, config_path=config_path)

        # 5. Training Cycle + Optimization.
        # -----------------------------------------

        self.run_training_cycle(
            optimize=optimize, 
            overfitting_filter=overfitting_filter, 
            gap_threshold=gap_threshold,
            display_results=True, 
            trim_models=True    
        )

        # 6. Final artifact savings.
        # -----------------------------------------
        self.save_artifacts()

        print(f"\n{'='*60}")
        print(f"✅ PIPELINE FINISHED SUCCESSFULLY")
        print(f"📁 Results saved on: '{save_path}'")
        print(f"{'='*60}\n")