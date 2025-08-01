# Assisted Living Facility (ALF) Incident Risk Prediction System

This repository contains the code for an Assisted Living Facility (ALF) Incident Risk Prediction System. The system aims to predict the likelihood of an incident occurring the next day for residents in an ALF, leveraging various patient vital signs, demographic information, and historical data.


## Project Overview

The primary goal of this project is to develop a predictive model that can identify residents at high risk of experiencing an incident on the following day. Early identification of such risks can enable proactive interventions, improving resident safety and quality of care in assisted living facilities. The system utilizes machine learning techniques, including CatBoost and LightGBM, along with robust data preprocessing and feature engineering.

## Features

*   **Data Loading and Initial Exploration**: Reads and displays basic information about the synthetic ALF dataset.
*   **Missing Value Visualization**: Heatmap to visualize the pattern of missing values in the dataset.
*   **Incident Rate Analysis**: Calculates and displays the overall incident rate.
*   **Numerical Feature Analysis**:
    *   Histograms to show distributions of numerical features (age, heart rate, blood pressure, temperature, medication adherence).
    *   Box plots for outlier detection in numerical features.
    *   Pearson correlation heatmap for numerical features.
*   **Categorical Feature Analysis**: Bar plot showing mean incident rate by diagnosis.
*   **Time-Series Analysis**: Line plot of weekly incident rates.
*   **Patient-Level Analysis**: Histograms showing the distribution of per-patient incident rates and the count of patients by the number of incidents.
*   **Statistical Tests**: Chi-squared test to assess the relationship between diagnosis and incident occurrence.
*   **Feature Engineering**:
    *   **Rolling Statistics**: Calculates rolling mean and standard deviation for vital signs over 1, 3, and 7 days.
    *   **Daily Deltas**: Computes daily changes in vital signs.
    *   **Risk Proxies**: Creates `age_group` and `med_adherence_bucket` features.
    *   **Interaction Flags**: Generates `high_risk_diag` and `elderly` binary flags.
*   **Data Preprocessing Pipeline**:
    *   **Imputation**: Uses `IterativeImputer` (MICE) for numerical features and `SimpleImputer` with a median strategy within the `ColumnTransformer`.
    *   **Scaling**: `StandardScaler` for numerical features.
    *   **Encoding**: `OneHotEncoder` for categorical features.
*   **Data Splitting**: Employs `GroupShuffleSplit` to ensure patient-wise separation across training, validation, and test sets, preventing data leakage.
*   **Imbalanced Data Handling**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) on the training data to address class imbalance.
*   **Model Training**: Trains CatBoost and LightGBM classifiers.
*   **Hyperparameter Tuning**: Implements `GridSearchCV` with a custom scoring function (Average Precision Score) to find optimal hyperparameters for CatBoost and LightGBM.
*   **Model Evaluation**: Reports various metrics including Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC on validation and test sets.
*   **Visualization of Model Performance**: ROC and Precision-Recall curves for the best performing model.

## Data Source

The project uses a synthetic dataset named `alf_synthetic.csv`. This dataset is downloaded from Kaggle via `kagglehub`.

## Installation

To set up the environment and run the code, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Assisted-Living-Risk-System
    ```

2.  **Install dependencies**:
    The project uses `uv pip` and `pip` for package management. Ensure you have `uv` installed, or replace `uv pip install` with `pip install`.

    ```bash
    !uv pip install -q scikit-learn==1.6.1 imbalanced-learn --system
    !pip install -q -U scikit-learn imbalanced-learn
    ```
    The full list of required libraries can be found in the imports section of the notebook:
    *   `pandas`
    *   `numpy`
    *   `seaborn`
    *   `matplotlib`
    *   `catboost`
    *   `lightgbm`
    *   `imblearn`
    *   `sklearn` (various modules for imputation, preprocessing, pipelines, model selection, metrics)
    *   `scipy`
    *   `torch` (checked for CUDA availability)
    *   `tensorflow` (checked for GPU availability)
    *   `joblib`

3.  **Download the dataset**:
    The notebook automatically downloads the `alf_synthetic.csv` file using `gdown` or reads it from the Kaggle input path.
    ```python
    !gdown '13t-mmHaCNOin0b2_6_o_TG2CS3nKxiPT' -O alf_synthetic.csv
    # or for Kaggle environment:
    # df = pd.read_csv("/kaggle/input/alf-synthetic/alf_synthetic.csv", parse_dates=["date"])
    ```

## Usage

The code is provided as a Jupyter Notebook (`Assisted-Living Risk system.py`). You can run it in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension, Google Colab, or Kaggle Notebooks).

1.  **Open the notebook**:
    ```bash
    jupyter notebook "main.ipynb"
    ```
2.  **Run all cells**: Execute the cells sequentially to perform data loading, exploration, feature engineering, model training, and evaluation.

## Model Training and Evaluation

The system trains and evaluates two gradient boosting models: CatBoost and LightGBM.

### Data Splitting Strategy

The data is split into training, validation, and test sets using `GroupShuffleSplit` to ensure that data from the same `patient_id` does not appear in different splits. This prevents data leakage and provides a more realistic evaluation of the model's generalization ability.

*   **Train**: 80% of patients
*   **Validation**: 10% of patients
*   **Test**: 10% of patients

### Preprocessing Pipeline

A `ColumnTransformer` is used to apply different preprocessing steps to numerical and categorical features:

*   **Numerical Features**:
    *   `SimpleImputer(strategy="median")`: Fills missing numerical values with the median.
    *   `StandardScaler()`: Scales numerical features to have zero mean and unit variance.
*   **Categorical Features**:
    *   `OneHotEncoder(handle_unknown="ignore")`: Converts categorical variables into a one-hot encoded numerical representation.

### Imbalance Handling

Due to the low incident rate, the dataset is highly imbalanced. SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to generate synthetic samples of the minority class, balancing the dataset for model training.

### Models

*   **CatBoostClassifier**:
    *   Initial parameters: `iterations=400`, `learning_rate=0.05`, `depth=6`, `loss_function='Logloss'`, `eval_metric='AUC'`, `verbose=0`, `random_seed=42`.
    *   Class weights are manually set to re-balance the classes: `{0:1, 1:int(len(y_train_bal)/sum(y_train_bal))}`.
*   **LGBMClassifier**:
    *   Initial parameters: `n_estimators=400`, `learning_rate=0.05`, `max_depth=-1`, `num_leaves=63`, `objective='binary'`, `metric='auc'`, `class_weight='balanced'`, `random_state=42`, `verbosity=-1`.

### Hyperparameter Tuning (Grid Search)

`GridSearchCV` is used to fine-tune the hyperparameters for both CatBoost and LightGBM. The scoring metric for grid search is `average_precision_score` (PR-AUC), which is more suitable for imbalanced datasets than accuracy or ROC-AUC.

**CatBoost Parameter Grid**:
\$$
\begin{cases}
\text{clf\_\_iterations}: [200, 400, 600] \\
\text{clf\_\_learning\_rate}: [0.03, 0.05, 0.07] \\
\text{clf\_\_depth}: [4, 6, 8] \\
\text{clf\_\_l2\_leaf\_reg}: [1, 3, 5] \\
\text{clf\_\_border\_count}: [32, 64]
\end{cases}
\$$

**LightGBM Parameter Grid**:
\$$
\begin{cases}
\text{clf\_\_n\_estimators}: [200, 400, 600] \\
\text{clf\_\_learning\_rate}: [0.03, 0.05, 0.07] \\
\text{clf\_\_max\_depth}: [-1, 4, 6] \\
\text{clf\_\_num\_leaves}: [31, 63, 127] \\
\text{clf\_\_min\_child\_samples}: [5, 15, 30] \\
\text{clf\_\_subsample}: [0.8, 1.0] \\
\text{clf\_\_colsample\_bytree}: [0.8, 1.0]
\end{cases}
\$$

## Results

The final evaluation is performed on the unseen test set using the best model identified through grid search. The key metrics reported are:

*   **Accuracy : 0.9383**
*   **Log-loss : 0.2302**
*   **Precision : 0.0000**
*   **Recall  : 0.0000**
*   **F1       : 0.0000**
*   **ROC-AUC  : 0.6019**
*   **PR-AUC   : 0.0951**

The summary indicates that while high accuracy might be achieved, the PR-AUC values are more indicative of performance on imbalanced datasets. The LightGBM model generally performed better in terms of PR-AUC after hyperparameter tuning.

## Future Work

The project identifies several areas for future improvement:

*   **Advanced Imbalance Handling**: Explore other techniques beyond SMOTE, such as ADASYN, NearMiss, or ensemble methods specifically designed for imbalanced learning (e.g., BalancedBaggingClassifier, EasyEnsembleClassifier).
*   **Data Distribution**: Investigate methods to handle non-normally distributed data, which was noted as affecting results. This could involve different scaling techniques or transformations.
*   **More Complex Feature Engineering**: Explore more sophisticated feature interactions or time-series features (e.g., Fourier transforms for cyclical patterns, more complex lag features).
*   **Deep Learning Models**: Consider recurrent neural networks (RNNs) or transformers for time-series data, which might capture temporal dependencies more effectively.
*   **Explainability**: Implement techniques like SHAP or LIME to understand model predictions and identify the most influential features for incident risk.
*   **Real-world Data**: Validate the models on real-world ALF data to assess their practical applicability and robustness.
