# ==========================================
# 1. Standard Python Libraries
# ==========================================
import os
import pickle
from typing import Literal

# ==========================================
# 2. Data Manipulation & Core
# ==========================================
import pandas as pd

# ==========================================
# 4. Machine Learning: Scikit-Learn Core
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, f_regression, SelectKBest


class DataPreparer:
    """
    Handles Train/Test Split, Feature Scaling (MinMax/Standard), and Feature Selection.
    """
    def __init__(self, target: str, problem_type: Literal["classification", "regression"],test_size: float = 0.2, random_state: int = 42) -> None:
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        self.save_path = None
        
        if problem_type != "classification" and problem_type != "regression":
            raise ValueError("❌ Error: 'problem_type' value should be 'classification' or 'regression'")

        self.problem_type = problem_type


    def auto_prepare_data(self, dataset: pd.DataFrame, label: str, 
                          problem_type: Literal['classification', 'regression'],
                          save_path: str = ".", save_data: bool = True, 
                          feature_sel: bool = True, k: int = 5) -> tuple:
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
            selected_cols = self.kselection(X_train=X_train, y_train=y_train, k=k, problem_type=problem_type)

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

    def kselection(self, X_train: pd.DataFrame, y_train: pd.DataFrame, k: int, problem_type: Literal['classification', 'regression']) -> list:
        """Selects top K features using ANOVA F-value."""
        f_value = f_classif if problem_type == 'classification' else f_regression
        selection_model = SelectKBest(f_value, k=k)
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