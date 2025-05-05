import os
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

# # latents_dir_path = r"D:\Projects\Annotators\data\llm_latents\llm_cls"
# latents_dir_path = r"D:\Projects\Annotators\data\moments\llm_image_embeds"
# # latents_dir_path = r"D:\Projects\Annotators\data\moments\gazelle_dinov2"
# # gt_path = r"D:\Projects\Annotators\data\llava_3s_video_results_w_gt.csv"
# gt_path = r"D:\Projects\Annotators\data\llava_3s_moments_results_primitives2_w_gt.csv"
# test_files = r"E:\moments\fmri\osfstorage-archive\annotations\test.csv"
# train_files = r"E:\moments\fmri\osfstorage-archive\annotations\train.csv"

# model_name = os.path.basename(latents_dir_path) if 'gazelle' in latents_dir_path else 'llava_next'
# results_path = f"D:\Projects\Annotators\data\moments\pls_results.csv"

def load_split_data(test_files, train_files):
    """Load and combine test and train split data"""
    test_df = pd.read_csv(test_files)
    train_df = pd.read_csv(train_files)
    test_df['split'] = 'test'
    train_df['split'] = 'train'
    return pd.concat([test_df, train_df], ignore_index=True)

def latent_loop(gt_df, latents_dir, train_latents, test_latents, splits_df, gt_columns, combined_features, gazzele_results_key, latent_shape):
    for idx, row in gt_df.iterrows():
        video_name = row['video_id']
        pt_name = video_name.replace('.mp4', '')
        # Use glob to find matching PT files
        matching_files = list(latents_dir.glob(f'*{pt_name}*.pt'))

        if matching_files:
            # Load and process all matching files
            latents = []
            for latent_path in matching_files:
                latent = torch.load(latent_path, map_location=torch.device('cpu'))
                if isinstance(latent, dict):
                    latent = latent[gazzele_results_key]
                if len(latent_shape) > 1:
                    latent = latent.mean(dim=0)
                latents.append(latent)

            # Average the latents from all matching files
            latent = torch.stack(latents).mean(dim=0)

            # Get ground truth values
            gt_vals = row[gt_columns].values

            # Add combined feature values if specified
            if combined_features:
                for feature_group in combined_features:
                    combined_val = row[feature_group].values.mean()
                    gt_vals = np.append(gt_vals, combined_val)

            gt_vals = gt_vals.astype(np.float32)
            if np.isnan(gt_vals).any():
                continue

            # Sort into train/test based on split
            split_type = splits_df.loc[splits_df['video_name'] == video_name, 'split'].values[0]
            if split_type == 'test':
                test_latents[video_name] = {'latent': latent.numpy(), 'gt': gt_vals}
            else:
                train_latents[video_name] = {'latent': latent.numpy(), 'gt': gt_vals}
    return train_latents, test_latents

def csv_loop(gt_df, latents_df, train_latents, test_latents, splits_df, gt_columns, combined_features, gazzele_results_key, latent_shape):
    for idx, row in gt_df.iterrows():
        video_name = row['video_id']
        filename_name = video_name.replace('.mp4', '')
        # find matches in latents csv
        # matching_files = list(latents_df[latents_df['video_id'].str.contains(filename_name)]['video_id'])
        matching_rows = latents_df[latents_df['video_id'].str.contains(filename_name)]
        for _, latent_row in matching_rows.iterrows():
            latent = latent_row[gazzele_results_key]
            if len(latent_shape) > 1:
                latent = latent.mean()
            gt_vals = row[gt_columns].values

            # Add combined feature values if specified
            if combined_features:
                for feature_group in combined_features:
                    combined_val = row[feature_group].values.mean()
                    gt_vals = np.append(gt_vals, combined_val)

            gt_vals = gt_vals.astype(np.float32)
            if np.isnan(gt_vals).any():
                continue

            # Sort into train/test based on split
            split_type = splits_df.loc[splits_df['video_name'] == video_name, 'split'].values[0]
            if split_type == 'test':
                test_latents[video_name] = {'latent': latent, 'gt': gt_vals}
            else:
                train_latents[video_name] = {'latent': latent, 'gt': gt_vals}
    return train_latents, test_latents

def load_latents_and_gt(latents_dir_path, gt_path, splits_df, gt_columns, combined_features=None):
    """
    Load latents and ground truth data

    Parameters:
        combined_features: list of lists, where each inner list contains features to be averaged
    """
    if latents_dir_path.endswith('.csv'):
        latents_df = pd.read_csv(latents_dir_path)
        csv_mode = True
    else:
        csv_mode = False
    gt_df = pd.read_csv(gt_path)
    train_latents = {}
    test_latents = {}
    gazzele_results_key = 'dino_cls'
    latents_dir = Path(latents_dir_path)

    # Find latent shape from first file
    if csv_mode:
        latent_shape = [1]
        gazzele_results_key = 'facingness_score'
        train_latents, test_latents = csv_loop(gt_df, latents_df, train_latents, test_latents, splits_df, gt_columns, combined_features, gazzele_results_key, latent_shape)
    else:
        first_latent = torch.load(next(latents_dir.glob('*.pt')), map_location='cpu')
        if isinstance(first_latent, dict):
            latent_shape = first_latent[gazzele_results_key].shape
        else:
            latent_shape = first_latent.shape
        train_latents, test_latents = latent_loop(gt_df, latents_dir, train_latents, test_latents, splits_df, gt_columns, combined_features, gazzele_results_key, latent_shape)



    return train_latents, test_latents

def train_pls_model(X, y, n_components_range=range(1, 40)):
    """Train PLS model with cross-validation"""
    pipeline = Pipeline([('pls', PLSRegression())])
    param_grid = {'pls__n_components': list(n_components_range)}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2')
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_score_, grid.best_params_['pls__n_components']

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = np.abs(y_true - y_pred).mean()
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    return r2, corr, mae

def save_latent_vectors(model_name, feature, train_latents, test_latents, reduced_train_dim_latents,
                        reduced_test_dim_latents, n_components, save_dir):
    """
    Save the dimensionality-reduced latent vectors to disk.

    Args:
        model_name: Name of the model used
        feature: Name of the feature being processed
        train_latents: Dictionary of training latent data
        test_latents: Dictionary of testing latent data
        reduced_train_dim_latents: Reduced dimensionality vectors for training data
        reduced_test_dim_latents: Reduced dimensionality vectors for testing data
        n_components: Number of components used in the reduction
    """
    # Create directory structure
    # output_dir = Path(f"D:/Projects/Annotators/data/moments/reduced_latents/{model_name}/{feature}_{n_components}comp")
    output_dir = Path(save_dir) / model_name / f"{feature}_{n_components}comp"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training data
    train_dict = {}
    for idx, video_id in enumerate(train_latents.keys()):
        train_dict[video_id] = {
            'reduced_latent': reduced_train_dim_latents[idx],
            'original_gt': train_latents[video_id]['gt']
        }

    # Save test data
    test_dict = {}
    for idx, video_id in enumerate(test_latents.keys()):
        test_dict[video_id] = {
            'reduced_latent': reduced_test_dim_latents[idx],
            'original_gt': test_latents[video_id]['gt']
        }

    # Save as numpy files
    np.save(output_dir / 'train_reduced_latents.npy', train_dict)
    np.save(output_dir / 'test_reduced_latents.npy', test_dict)

    print(f"Saved reduced latent vectors for feature '{feature}' to {output_dir}")

    return output_dir

def process_feature(train_latents, test_latents, feature_idx, model_name, feature_name, save_dir,
                    scaler=None, save_latents=False):
    """Process a single feature"""
    # Prepare training data
    X_train = np.stack([d['latent'] for d in train_latents.values()])
    y_train = np.array([d['gt'][feature_idx] for d in train_latents.values()])

    X_train = X_train.reshape(-1, 1) if len(X_train.shape) == 1 else X_train

    # Scale features
    # if X_train.shape[1] > 1:
    # if scaler is None:
    #     if X_train.shape[1] > 1:
    #         scaler = StandardScaler()
    #     else:
    #         scaler = MinMaxScaler()
    #     X_train = scaler.fit_transform(X_train)

    if X_train.shape[1] > 1:
        # X_train = scaler.transform(X_train)
        y_scaler = StandardScaler()
        # y_scaler = MinMaxScaler()
    else:
        y_scaler = MinMaxScaler()


    # Scale targets


    # y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Train model
    if X_train.shape[1] > 1:
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        model, cv_score, n_components = train_pls_model(X_train, y_train)

        # Prepare test data
        X_test = np.stack([d['latent'] for d in test_latents.values()])
        # X_test = scaler.transform(X_test)
        y_test = np.array([d['gt'][feature_idx] for d in test_latents.values()])

        # Make predictions
        Z_test = model.predict(X_test)
        Z_test = y_scaler.inverse_transform(Z_test.reshape(-1, 1)).ravel()
    else:
        # No PLS model for single feature
        # Prepare test data
        X_test = np.stack([d['latent'] for d in test_latents.values()])
        X_test = X_test.reshape(-1, 1)
        X_test = scaler.transform(X_test)
        X_test = X_test.reshape(-1)
        y_test = np.array([d['gt'][feature_idx] for d in test_latents.values()])
        # y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        y_test = y_test.reshape(-1)
        Z_test = X_test
        cv_score = 0
        n_components = 1

    # Calculate metrics
    r2, corr, mae = evaluate_model(y_test, Z_test)
    train_r2, train_corr, train_mae = evaluate_model(y_train, model.predict(X_train))
    compare = np.c_[list(test_latents.keys()), y_test, Z_test, np.abs(y_test-Z_test)]

    if save_latents:
# Transform data using the PLS model
        reduced_train_dim_latents = model.transform(X_train)
        reduced_test_dim_latents = model.transform(X_test)

        if model_name is not None and feature_name is not None:
            # Save latent vectors
            output_dir = save_latent_vectors(
                model_name,
                feature_name,
                train_latents,
                test_latents,
                reduced_train_dim_latents,
                reduced_test_dim_latents,
                n_components,
                save_dir
            )

            # Save the fitted model
            import joblib
            model_filename = (output_dir /
                              f"{feature_name}_{n_components}comp_pls_model.joblib")
            model_filename.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_filename)

            # Save the scalers if they exist
            if hasattr(model, 'steps') and 'y_scaler' in locals() and y_scaler is not None:
                scaler_filename = output_dir / f"{feature_name}_{n_components}comp_y_scaler.joblib"
                joblib.dump(y_scaler, scaler_filename)

            print(f"Saved PLS model to {model_filename}")



    return r2, corr, mae, cv_score, n_components, scaler

def main():
    # Paths
    # latents_dir_path = r"D:\Projects\Annotators\data\moments\gazzele_results"
    latents_dir_path = r"D:\Projects\Annotators\data\moments\llm_image_embeds"
    latents_dir_path = r"D:\Projects\Annotators\data\moments\llava_general_people_desc"
    # latents_dir_path = r"D:\Projects\Annotators\data\moments\llava_facing_llm_cls"
    # latents_dir_path = r"D:\Projects\gazelle\facingness_results.csv"        # gazelle post processed facing results
    # latents_dir_path = r"D:\Projects\Annotators\data\moments\comm_llm_cls"
    # latents_dir_path = r"D:\Projects\Annotators\data\moments\llava_joint_action_cls"
    gt_path = r"D:\Projects\Annotators\data\moments\llava_3s_moments_results_primitives2_w_gt.csv"
    test_files = r"D:\Projects\data\Moments_in_Time_layla\fmri\osfstorage-archive\annotations\test.csv"
    train_files = r"D:\Projects\data\Moments_in_Time_layla\fmri\osfstorage-archive\annotations\train.csv"
    results_path = r"D:\Projects\Annotators\data\moments\pls_results.csv"
    save_dir = r"D:\Projects\Annotators\data\moments\reduced_latents"

    # Model name
    model_name = os.path.basename(latents_dir_path) if 'gazelle' in latents_dir_path else 'llava_next'
    if 'llava' in model_name:
        # take the dirname of the path
        model_name = os.path.basename(latents_dir_path)

    # Load data
    splits_df = load_split_data(test_files, train_files)

    # Load or create results DataFrame
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        print(f"Results loaded from {results_path}")
    else:
        results_df = pd.DataFrame(columns=['model', 'split', 'feature', 'R^2', 'corr', 'mae', 'cv_score', 'n_components'])

    # Features to analyze
    gt_columns = ['indoor', 'expanse', 'transitivity', 'agent distance',
       'facingness', 'joint action', 'communication', 'cooperation',
       'dominance', 'intimacy', 'valence', 'arousal']

    # Define combined features
    combined_features = [['facingness', 'agent distance']]

    # Create feature names including combined features
    all_feature_names = gt_columns + ['combined_facingness_distance']

    # Load latents and ground truth with combined features
    train_latents, test_latents = load_latents_and_gt(
        latents_dir_path,
        gt_path,
        splits_df,
        gt_columns,
        combined_features
    )

    # Process each feature
    scaler = None  # Will be initialized in first iteration
    for idx, feature in enumerate(all_feature_names):
        print(f"\nProcessing feature: {feature}")

        r2, corr, mae, cv_score, n_components, scaler = process_feature(
            train_latents, test_latents, idx, model_name, feature, save_dir, scaler, save_latents=True
        )

        # Update results
        results = {
            'model': model_name,
            'split': 'test',
            'feature': feature,
            'R^2': r2,
            'corr': corr,
            'mae': mae,
            'cv_score': cv_score,
            'n_components': n_components
        }

        # Update or append results
        existing_row = ((results_df['model'] == model_name) &
                       (results_df['split'] == 'test') &
                       (results_df['feature'] == feature))

        if existing_row.any():
            for key, value in results.items():
                results_df.loc[existing_row, key] = value
        else:
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

        print(f"Feature: {feature}")
        print(f"Test R^2: {r2:.3f}")
        print(f"Test Correlation: {corr:.3f}")
        print(f"Test Mean absolute error: {mae:.3f}")
        print(f"Best CV score: {cv_score:.3f}")
        print(f"Number of components: {n_components}")

    # Save results
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
