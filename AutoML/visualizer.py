
# ==========================================
# 1. Standard Python Libraries
# ==========================================
import math

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

# ==========================================
# Module Imports
# ==========================================

from .explorer import DataExplorer


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