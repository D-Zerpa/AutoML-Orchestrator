# ==========================================
# 1. Standard Python Libraries
# ==========================================
import os
from typing import Optional, Dict, Any, Literal

# ==========================================
# 2. Data Manipulation & Core
# ==========================================
import pandas as pd

# ==========================================
# 3. Visualization & Utilities
# ==========================================
from IPython.display import display

# ==========================================
# 4. Machine Learning: Models
# ==========================================
# Linear Models
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
# Trees & Ensembles
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
# Gradient Boosting (External)
import xgboost as xgb


# ==========================================
# Module Imports
# ==========================================

from .explorer import DataExplorer
from .visualizer import Visualizer
from .feat_engineer import FeatureEngineer
from .preparer import DataPreparer
from .trainer import ModelTrainer


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
        self.problem_type = None

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



    def load_data(self, df: str | pd.DataFrame, target: str)-> None:
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
            if isinstance(df, str):
                df_loaded = pd.read_csv(df)

            elif isinstance(df, pd.DataFrame):
                df_loaded = df
        except FileNotFoundError:
             raise FileNotFoundError(f"❌ Error: File not found at '{df}'")

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


    def prepare_features (self, problem_type: Literal["classification", "regression"], save_path: str = ".", save_data: bool = True, feature_selection: bool = True, k_features: int = 5) -> None:

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

        # Save problem_type.
        self.problem_type = problem_type

        # Initialize the FeatureEngineer and DataPreparer tools.
        self.feat_engineer = FeatureEngineer(explorer=self.explorer, save_path=self.save_path)
        self.preparer = DataPreparer(target=self.target, problem_type=self.problem_type)

        # Run .auto_process_data() to generate the two dataset variants.
        df_w_out, df_wo_out = self.feat_engineer.auto_process_data(save_encoder_data=save_data)

        # Initialize a dictionary with both datasets, so we can iterate through them.
        datasets= {"with_outliers": df_w_out, "without_outliers": df_wo_out}

        # Create empty dicts, so the data created by the iteration remains.
        merged_train = {}
        merged_test = {}
        merged_scalers = {}

        print(f"\n⚙️  Processing Variants...")
        # Iterate through the datasets and apply the .auto_prepare_data() that will fill our dicts.
        for key, dataset in datasets.items():
            train, test = self.preparer.auto_prepare_data(dataset=dataset,
                                                          label=key, save_data=save_data,
                                                          save_path= self.save_path, feature_sel= feature_selection,
                                                          k=k_features, problem_type=self.problem_type)
            merged_train.update(train)
            merged_test.update(test)

            if hasattr(self.preparer, 'scalers'):
                 merged_scalers.update(self.preparer.scalers)

        # Save all the data into the mainframe
        self.train_dict = merged_train
        self.test_dict = merged_test
        self.scalers = merged_scalers

        print(f"✅ Preparation finished. Generated {len(self.train_dict)-1} training sets.")


    def initialize_trainer(self, config_path: Optional[str] = None, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Configures the ModelTrainer and defines which algorithms will compete.

        Args:
            config_path (str, optional): Path to a JSON with hyperparameters.
            model_config (Dict, optional): Manual model dictionary {Name: ModelObject}.
                                           If None, loads a robust default configuration.
        """

        # Make some validations to check if the data is available.
        if self.train_dict is None or self.test_dict is None:
            raise ValueError("⚠️ No training data found. Run .prepare_features() first.")

        # Initialize the ModelTrainer tool
        self.trainer = ModelTrainer(dict_train=self.train_dict, dict_test=self.test_dict, 
                                    problem_type=self.problem_type, config_path=config_path)

        print(f"\n🥊 --- Initializing ModelTrainer ({self.problem_type}) ---")
        
        # Check if a config is added by the user, else add defaults.
        if model_config:
            print(f"   -> Loading {len(model_config)} custom models...")
            for model, m_object in model_config.items():
                self.trainer.add_model(model=model, model_object=m_object)

        #By default, 5 models for each problem type.
        else:
            print(f"   -> Loading default configuration for {self.problem_type}...")
            if self.problem_type == "classification":
                self.trainer.add_model("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42))
                self.trainer.add_model("RandomForestClassifier", RandomForestClassifier(random_state=42))
                self.trainer.add_model("XGBClassifier", xgb.XGBClassifier(random_state= 42))
                self.trainer.add_model("LogisticRegression", LogisticRegression(random_state=42))
                self.trainer.add_model("KNeighbors", KNeighborsClassifier())
            else:
                self.trainer.add_model("RandomForestRegressor", RandomForestRegressor(random_state=42))
                self.trainer.add_model("DecisionTreeRegressor",DecisionTreeRegressor(random_state=42))
                self.trainer.add_model("Lasso",Lasso(random_state=42))
                self.trainer.add_model("Ridge",Ridge(random_state=42))
                self.trainer.add_model("ElasticNet",ElasticNet(random_state=42))

        print(f"Added models for: {self.trainer.problem_type}\n{list(self.trainer.models.keys())}")

    def run_training_cycle(self, optimize: bool = True, overfitting_filter: bool = True, 
                           gap_threshold: float = 0.3, display_results: bool = True, 
                           trim_models: bool = True, stacking_models: bool = False, stacking_top: int = 3) -> None:

        """
        Executes the training lifecycle: Selection -> Optimization.

        Args:
            optimize (bool): If True, runs hyperparameter search for the best model.
            overfitting_filter (bool): If True, discards models with gap > gap_threshold.
            gap_threshold (float): Max tolerated difference between Train and Test score.
            display_results (bool): Shows result tables in console/notebook.
            trim_models (bool): If True, deletes losing models from memory to save RAM.
            stacking_models (bool): If True, builds a meta-model with the best algorithms.
            stacking_top (int): Number of top models to include in the stacking ensemble.
        """

        # Make some validations to check if the data is available.
        if self.trainer is None:
            raise ValueError("⚠️ Trainer not initialized. Run .initialize_trainer() first.")

        # First training to obtain the best Dataset.
        first_training = self.trainer.train_all_configs(overfitting_filter=overfitting_filter, gap_threshold=gap_threshold)
        
        if stacking_models:
                trim_models = False # Turn off the trimming, because we need all the models.

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

        # Optional stacking.
        if stacking_models:
            self.trainer.build_stacking_ensemble(top_n=stacking_top)
        else:
            if display_results:
                self.trainer.evaluate_model()

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
                     optimize: bool = True, overfitting_filter: bool = True, gap_threshold: float = 0.3, 
                     stacking_models: bool = False, stacking_top: int = 3) -> None:
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
            stacking_models (bool): If True, builds a meta-model with the best algorithms.
            stacking_top (int): Number of top models to include in the stacking ensemble.
        """
        
        print(f"\n{'='*60}")
        print(f"🚀 AUTOMATIC PIPELINE RUNNING: {problem_type.upper()}")
        print(f"🎯 Target: '{target}' | 📂 Output: '{save_path}'")
        print(f"{'='*60}")

        # 1. Load and initialize Explorer and Visualizer tool.
        # -----------------------------------------
        self.load_data(df=df_path, target=target)

        # 2. Data Analysis (Visual, just Data or both).
        # -----------------------------------------
        self.execute_analysis(mode=analysis_mode)

        # 3. Feature Engineering and Data Preparation.
        # -----------------------------------------
        self.prepare_features(problem_type=problem_type, save_path=save_path, save_data=True)

        # 4. Model Trainer configuration.
        # -----------------------------------------
        self.initialize_trainer(config_path=config_path)

        # 5. Training Cycle + Optimization.
        # -----------------------------------------

        self.run_training_cycle(
            optimize=optimize, 
            overfitting_filter=overfitting_filter, 
            gap_threshold=gap_threshold,
            display_results=True, 
            trim_models=True,
            stacking_models = stacking_models, 
            stacking_top = stacking_top   
        )

        # 6. Final artifact savings.
        # -----------------------------------------
        self.save_artifacts()

        print(f"\n{'='*60}")
        print(f"✅ PIPELINE FINISHED SUCCESSFULLY")
        print(f"📁 Results saved on: '{save_path}'")
        print(f"{'='*60}\n")