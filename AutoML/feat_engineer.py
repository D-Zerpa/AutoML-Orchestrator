# ==========================================
# 1. Standard Python Libraries
# ==========================================
import os
import json

# ==========================================
# 2. Data Manipulation & Core
# ==========================================
import pandas as pd

# ==========================================
# Module Imports
# ==========================================

from .explorer import DataExplorer


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

            if self.df_with_outliers is None:
                working_df = self.explorer.df.copy()
            else:
                working_df = self.df_with_outliers

            null_counts = working_df.isnull().sum().sum()
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