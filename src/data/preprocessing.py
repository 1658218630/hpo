import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")


def preprocess_dataset(
    file_path: str, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Comprehensive preprocessing pipeline addressing:
    - NaN handling
    - Duplicate removal
    - Feature correlation analysis
    - Categorical encoding
    - Feature scaling
    - Class imbalance detection
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE PREPROCESSING PIPELINE")
        print(f"{'='*60}")
        print(f"Processing: {file_path.split('/')[-1]}")

    # Load dataset
    df = pd.read_csv(file_path)
    original_shape = df.shape
    if verbose:
        print(f"Original shape: {original_shape}")

    # Identify target column (assume last column)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if verbose:
        print(f"Target column: '{target_col}'")
        print(f"Features: {X.shape[1]} columns")

    # ===== STEP 1: DUPLICATE REMOVAL =====
    if verbose:
        print(f"\n--- Step 1: Duplicate Removal ---")

    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)

    if duplicates_removed > 0:
        if verbose:
            print(
                f"Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_rows*100:.1f}%)"
            )
        # Update X and y after duplicate removal
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
    else:
        if verbose:
            print("No duplicates found")

    # ===== STEP 2: TARGET VARIABLE CLEANING =====
    if verbose:
        print(f"\n--- Step 2: Target Variable Processing ---")

    # Clean target variable (handle byte strings and inconsistencies)
    if y.dtype == "object":
        y = y.astype(str).str.replace(r"^b'|'$", "", regex=True).str.strip()
        y = y.str.lower()  # Normalize case

    if verbose:
        print(f"Target classes: {sorted(y.unique())}")
        print(f"Class distribution:")
        class_counts = y.value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")

        # Class imbalance detection
        min_class_ratio = class_counts.min() / class_counts.max()
        if min_class_ratio < 0.1:
            print(f"⚠️  SEVERE CLASS IMBALANCE detected! Ratio: {min_class_ratio:.3f}")
        elif min_class_ratio < 0.3:
            print(f"⚠️  Moderate class imbalance detected. Ratio: {min_class_ratio:.3f}")
        else:
            print(f"✓ Balanced classes. Ratio: {min_class_ratio:.3f}")

    # ===== STEP 3: NaN HANDLING =====
    if verbose:
        print(f"\n--- Step 3: Missing Value Analysis ---")

    # Analyze missing values
    missing_analysis = X.isnull().sum()
    missing_features = missing_analysis[missing_analysis > 0]

    if len(missing_features) > 0:
        if verbose:
            print(f"Features with missing values:")
            for feature, missing_count in missing_features.items():
                missing_pct = missing_count / len(X) * 100
                print(f"  {feature}: {missing_count} ({missing_pct:.1f}%)")

        # Drop features with >50% missing values
        high_missing_features = missing_features[
            missing_features > len(X) * 0.5
        ].index.tolist()
        if high_missing_features:
            if verbose:
                print(f"Dropping features with >50% missing: {high_missing_features}")
            X = X.drop(columns=high_missing_features)

        # Impute remaining missing values
        remaining_missing = X.isnull().sum().sum()
        if remaining_missing > 0:
            if verbose:
                print(f"Imputing {remaining_missing} remaining missing values...")

            # Separate numeric and categorical for different imputation strategies
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=["object"]).columns

            # Numeric imputation (median)
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy="median")
                X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

            # Categorical imputation (most frequent)
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy="most_frequent")
                X[categorical_cols] = categorical_imputer.fit_transform(
                    X[categorical_cols]
                )
    else:
        if verbose:
            print("No missing values found")

    # ===== STEP 4: CATEGORICAL ENCODING =====
    if verbose:
        print(f"\n--- Step 4: Categorical Variable Encoding ---")

    categorical_cols = X.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        if verbose:
            print(f"Encoding {len(categorical_cols)} categorical features:")

        for col in categorical_cols:
            # Clean categorical variables
            if X[col].dtype == "object":
                X[col] = X[col].astype(str)
                X[col] = X[col].str.replace(r"^b'|'$", "", regex=True).str.strip()
                X[col] = X[col].str.lower()  # Normalize case

            # Check cardinality
            unique_values = X[col].nunique()
            if verbose:
                print(f"  {col}: {unique_values} unique values")

            # Use LabelEncoder for now (could be enhanced with OneHotEncoder for low cardinality)
            if unique_values > 1:  # Only encode if there's variance
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            else:
                if verbose:
                    print(f"    ⚠️  Dropping {col} (only 1 unique value)")
                X = X.drop(columns=[col])
    else:
        if verbose:
            print("No categorical features found")

    # ===== STEP 5: FEATURE CORRELATION ANALYSIS =====
    if verbose:
        print(f"\n--- Step 5: Feature Correlation Analysis ---")

    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])

    # Remove zero-variance features
    initial_features = X.shape[1]
    variance_selector = VarianceThreshold(threshold=0)
    X = pd.DataFrame(
        variance_selector.fit_transform(X),
        columns=X.columns[variance_selector.get_support()],
        index=X.index,
    )

    zero_var_removed = initial_features - X.shape[1]
    if zero_var_removed > 0 and verbose:
        print(f"Removed {zero_var_removed} zero-variance features")

    # High correlation detection
    if X.shape[1] > 1:
        corr_matrix = X.corr().abs()

        # Find highly correlated pairs (>0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )

        if high_corr_pairs:
            if verbose:
                print(f"High correlation pairs found (>0.95):")
                for feat1, feat2, corr_val in high_corr_pairs[:5]:  # Show top 5
                    print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
                if len(high_corr_pairs) > 5:
                    print(f"  ... and {len(high_corr_pairs)-5} more")

            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2, _ in high_corr_pairs:
                if feat1 not in features_to_remove:
                    features_to_remove.add(feat2)

            if features_to_remove and verbose:
                print(f"Removing {len(features_to_remove)} highly correlated features")

            X = X.drop(columns=list(features_to_remove))
        else:
            if verbose:
                print("No highly correlated features found")

    # ===== STEP 6: FEATURE SCALING =====
    if verbose:
        print(f"\n--- Step 6: Feature Scaling ---")

    if X.shape[1] > 0:
        # Analyze data distribution for scaling decision
        data_range = X.max().max() - X.min().min()
        max_value = X.max().max()
        min_value = X.min().min()

        if verbose:
            print(
                f"Data range: [{min_value:.3f}, {max_value:.3f}] (range: {data_range:.3f})"
            )

        # Decide on scaling strategy
        if data_range > 10 or max_value > 100 or min_value < -10:
            # Check for outliers to decide between StandardScaler vs RobustScaler
            outlier_detection = []
            for col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))
                ).sum()
                outlier_detection.append(outliers)

            total_outliers = sum(outlier_detection)
            outlier_percentage = total_outliers / (len(X) * len(X.columns)) * 100

            if outlier_percentage > 5:
                if verbose:
                    print(
                        f"High outlier percentage ({outlier_percentage:.1f}%), using RobustScaler"
                    )
                scaler = RobustScaler()
            else:
                if verbose:
                    print(
                        f"Low outlier percentage ({outlier_percentage:.1f}%), using StandardScaler"
                    )
                scaler = StandardScaler()

            X_scaled = scaler.fit_transform(X)

            if verbose:
                print(
                    f"After scaling - range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]"
                )

            # Convert back to DataFrame to maintain column names
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            if verbose:
                print("Data range acceptable, no scaling applied")

    # ===== FINAL SUMMARY =====
    feature_names = X.columns.tolist()

    if verbose:
        print(f"\n{'='*60}")
        print(f"PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Original shape: {original_shape}")
        print(f"Final shape: {(len(X), len(feature_names))}")
        print(
            f"Rows removed: {original_shape[0] - len(X)} ({(original_shape[0] - len(X))/original_shape[0]*100:.1f}%)"
        )
        print(
            f"Features removed: {original_shape[1] - 1 - len(feature_names)} (excluding target)"
        )
        print(f"Final feature count: {len(feature_names)}")
        print(f"Data type: {X.dtypes.value_counts().to_dict()}")
        print(f"Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"Ready for machine learning!")
        print(f"{'='*60}\n")

    return X.values, y.values, feature_names


def get_preprocessing_report(file_path: str) -> dict:
    """
    Generate a detailed preprocessing report without actually preprocessing
    """
    df = pd.read_csv(file_path)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    report = {
        "original_shape": df.shape,
        "target_classes": (
            sorted(y.unique()) if y.dtype == "object" else list(y.unique())
        ),
        "class_distribution": y.value_counts().to_dict(),
        "duplicates": df.duplicated().sum(),
        "missing_values": X.isnull().sum().to_dict(),
        "categorical_features": X.select_dtypes(include=["object"]).columns.tolist(),
        "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        "data_range": {
            "min": (
                float(X.select_dtypes(include=[np.number]).min().min())
                if len(X.select_dtypes(include=[np.number]).columns) > 0
                else None
            ),
            "max": (
                float(X.select_dtypes(include=[np.number]).max().max())
                if len(X.select_dtypes(include=[np.number]).columns) > 0
                else None
            ),
        },
    }

    return report
