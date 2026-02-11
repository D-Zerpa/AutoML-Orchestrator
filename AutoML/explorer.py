# ==========================================
# 1. Standard Python Libraries
# ==========================================
from typing import Optional, List

# ==========================================
# 2. Data Manipulation & Core
# ==========================================
import pandas as pd
from pandas.api.types import is_numeric_dtype

# ==========================================
# 3. Visualization & Utilities
# ==========================================
from tabulate import tabulate




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
            print(f"🔠 Categorical Frequencies:")
            print(self.df[self.cat_cols].value_counts(normalize=True)) 
            print("-" * 44)