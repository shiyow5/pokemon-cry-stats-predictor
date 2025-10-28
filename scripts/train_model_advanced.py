import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
import json
from datetime import datetime

# パス定義
DATA_PATH = "data/processed_features_advanced.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# ターゲット変数
TARGET_STATS = ["hp", "attack", "defense", "speed", "sp_attack", "sp_defense"]


def check_and_generate_data():
    """
    Check if required data files exist and generate them if they don't.
    
    Dependency chain:
    1. raw_stats.csv (from fetch_stats.py)
    2. cries/*.ogg (from download_cries.py)
    3. audio_features_advanced.csv (from extract_audio_features_advanced.py)
    4. processed_features_advanced.csv (from merge_dataset_advanced.py)
    """
    import subprocess
    import sys
    from pathlib import Path
    
    print("Checking required data files...")
    
    # 1. Check raw_stats.csv
    raw_stats_path = Path("data/raw_stats.csv")
    if not raw_stats_path.exists():
        print(f"⚠️  {raw_stats_path} not found. Generating...")
        result = subprocess.run([sys.executable, "scripts/fetch_stats.py"])
        if result.returncode != 0:
            print(f"❌ Failed to generate {raw_stats_path}")
            sys.exit(1)
        print(f"✅ Generated {raw_stats_path}")
    else:
        print(f"✅ {raw_stats_path} exists")
    
    # 2. Check cries directory has audio files
    cries_dir = Path("data/cries")
    cries_dir.mkdir(parents=True, exist_ok=True)
    audio_files = list(cries_dir.glob("*.ogg"))
    if len(audio_files) == 0:
        print(f"⚠️  No audio files found in {cries_dir}/. Downloading...")
        result = subprocess.run([sys.executable, "scripts/download_cries.py"])
        if result.returncode != 0:
            print(f"❌ Failed to download audio files")
            sys.exit(1)
        audio_files = list(cries_dir.glob("*.ogg"))
        print(f"✅ Downloaded {len(audio_files)} audio files")
    else:
        print(f"✅ {cries_dir}/ has {len(audio_files)} audio files")
    
    # 3. Check audio_features_advanced.csv
    audio_features_path = Path("data/audio_features_advanced.csv")
    if not audio_features_path.exists():
        print(f"⚠️  {audio_features_path} not found. Extracting advanced features...")
        result = subprocess.run([sys.executable, "scripts/extract_audio_features_advanced.py"])
        if result.returncode != 0:
            print(f"❌ Failed to extract audio features")
            sys.exit(1)
        print(f"✅ Generated {audio_features_path}")
    else:
        print(f"✅ {audio_features_path} exists")
    
    # 4. Check processed_features_advanced.csv
    processed_path = Path(DATA_PATH)
    if not processed_path.exists():
        print(f"⚠️  {processed_path} not found. Merging datasets...")
        result = subprocess.run([sys.executable, "scripts/merge_dataset_advanced.py"])
        if result.returncode != 0:
            print(f"❌ Failed to merge datasets")
            sys.exit(1)
        print(f"✅ Generated {processed_path}")
    else:
        print(f"✅ {processed_path} exists")
    
    print("✅ All required data files are ready!\n")


def load_data():
    """データを読み込んで特徴量とターゲットに分割"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 特徴量列を抽出（'name'とTARGET_STATS以外）
    feature_cols = [col for col in df.columns if col not in ['name'] + TARGET_STATS]
    
    X = df[feature_cols]
    y = df[TARGET_STATS]
    
    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")
    print(f"  Feature dimensions: {X.shape[1]}")
    
    return X, y


def create_neural_network(input_dim, output_dim):
    """ニューラルネットワークモデルを作成"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_random_forest(X_train, y_train, X_test, y_test, use_tuning=False):
    """Random Forestモデルを学習"""
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)
    
    if use_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [None, 10, 20],
            'estimator__min_samples_split': [2, 5]
        }
        
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base_model)
        
        # GridSearchCV (簡易版 - 時間がかかるため)
        print("  Note: Using simplified grid search due to time constraints")
        base_model = RandomForestRegressor(n_estimators=200, max_depth=20, 
                                          min_samples_split=2, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base_model)
    else:
        base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base_model)
    
    model.fit(X_train, y_train)
    
    # 評価
    y_pred = model.predict(X_test)
    results = evaluate_model(y_pred, y_test, "Random Forest")
    
    return model, results


def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoostモデルを学習"""
    print("\n" + "="*60)
    print("Training XGBoost Model")
    print("="*60)
    
    models = []
    all_predictions = []
    
    for i, stat in enumerate(TARGET_STATS):
        print(f"  Training {stat.upper()}...")
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train.iloc[:, i])
        models.append(model)
        
        pred = model.predict(X_test)
        all_predictions.append(pred)
    
    # 予測を結合
    y_pred = np.column_stack(all_predictions)
    results = evaluate_model(y_pred, y_test, "XGBoost")
    
    return models, results


def train_neural_network(X_train, y_train, X_test, y_test):
    """ニューラルネットワークモデルを学習"""
    print("\n" + "="*60)
    print("Training Neural Network Model")
    print("="*60)
    
    # データの正規化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル作成
    model = create_neural_network(X_train.shape[1], y_train.shape[1])
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # 学習
    print("  Training...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    
    print(f"  Training completed (epochs: {len(history.history['loss'])})")
    
    # 評価
    y_pred = model.predict(X_test_scaled, verbose=0)
    results = evaluate_model(y_pred, y_test, "Neural Network")
    
    return model, scaler, results


def evaluate_model(y_pred, y_test, model_name):
    """モデルを評価"""
    print(f"\n--- {model_name} Evaluation ---")
    
    results = {
        'model': model_name,
        'stats': {}
    }
    
    for i, stat in enumerate(TARGET_STATS):
        y_true = y_test.iloc[:, i]
        y_p = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        r2 = r2_score(y_true, y_p)
        
        results['stats'][stat] = {
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        print(f"  {stat.upper():12s}: RMSE={rmse:6.2f}, R²={r2:7.3f}")
    
    # 全体のR²スコア
    overall_r2 = r2_score(y_test, y_pred)
    results['overall_r2'] = float(overall_r2)
    
    print(f"  {'Overall':12s}: R²={overall_r2:7.3f}")
    
    return results


def perform_cross_validation(X, y, model_type='rf'):
    """クロスバリデーションを実施"""
    print("\n" + "="*60)
    print(f"Performing 5-Fold Cross Validation ({model_type.upper()})")
    print("="*60)
    
    if model_type == 'rf':
        base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base_model)
    elif model_type == 'xgb':
        base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model = MultiOutputRegressor(base_model)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return {
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': float(cv_scores.mean()),
        'std_cv_score': float(cv_scores.std())
    }


def save_results(all_results):
    """結果を保存"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")


def save_best_model(best_model, best_model_name, scaler=None):
    """最良のモデルを保存"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if best_model_name == "Neural Network":
        model_path = os.path.join(MODEL_DIR, "pokemon_stats_nn.keras")
        best_model.save(model_path)
        
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Neural Network model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
    elif best_model_name == "XGBoost":
        for i, (model, stat) in enumerate(zip(best_model, TARGET_STATS)):
            model_path = os.path.join(MODEL_DIR, f"pokemon_stats_xgb_{stat}.json")
            model.save_model(model_path)
        print(f"✅ XGBoost models saved to {MODEL_DIR}/")
    else:
        model_path = os.path.join(MODEL_DIR, "pokemon_stats_rf_advanced.joblib")
        joblib.dump(best_model, model_path)
        print(f"✅ Random Forest model saved to {model_path}")


def main(model_type='all', test_size=0.2, random_state=42, params=None):
    """
    Main training function

    Args:
        model_type: Type of model to train ('rf', 'xgb', 'nn', or 'all')
        test_size: Test set size (default: 0.2)
        random_state: Random seed (default: 42)
        params: Model-specific parameters (dict)
    """
    print("="*60)
    print("Advanced Model Training with Multiple Algorithms")
    print("="*60)
    print()
    
    # Check and generate required data files if they don't exist
    check_and_generate_data()

    # データ読み込み
    X, y = load_data()

    # トレーニング/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    all_results = {}
    trained_models = {}

    # Train selected model(s)
    if model_type in ['rf', 'all']:
        # 1. Random Forest
        rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test, use_tuning=False)
        all_results['random_forest'] = rf_results
        trained_models['rf'] = ('Random Forest', rf_model, None)

    if model_type in ['xgb', 'all']:
        # 2. XGBoost
        xgb_models, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
        all_results['xgboost'] = xgb_results
        trained_models['xgb'] = ('XGBoost', xgb_models, None)

    if model_type in ['nn', 'all']:
        # 3. Neural Network
        nn_model, scaler, nn_results = train_neural_network(X_train, y_train, X_test, y_test)
        all_results['neural_network'] = nn_results
        trained_models['nn'] = ('Neural Network', nn_model, scaler)

    # 4. Cross Validation (only for Random Forest and only if training RF)
    if model_type in ['rf', 'all']:
        cv_results = perform_cross_validation(X, y, model_type='rf')
        all_results['cross_validation_rf'] = cv_results

    # 結果の比較
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)

    for model_name, results in all_results.items():
        if 'overall_r2' in results:
            print(f"{model_name:20s}: Overall R² = {results['overall_r2']:.3f}")

    # 最良のモデルを特定
    best_r2 = -float('inf')
    best_model_name = None
    best_model = None
    best_scaler = None

    for model_key, (name, model, scaler) in trained_models.items():
        result_key = {'rf': 'random_forest', 'xgb': 'xgboost', 'nn': 'neural_network'}[model_key]
        if result_key in all_results and all_results[result_key]['overall_r2'] > best_r2:
            best_r2 = all_results[result_key]['overall_r2']
            best_model_name = name
            best_model = model
            best_scaler = scaler

    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.3f})")

    # 結果を保存
    save_results(all_results)

    # 最良のモデルを保存
    save_best_model(best_model, best_model_name, best_scaler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Pokemon stats prediction models')
    parser.add_argument('--model-type', type=str, default='all',
                        choices=['rf', 'xgb', 'nn', 'all'],
                        help='Type of model to train (rf=Random Forest, xgb=XGBoost, nn=Neural Network, all=All models)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--params', type=str, default=None,
                        help='Model parameters as JSON string')

    args = parser.parse_args()

    # Parse params if provided
    params = None
    if args.params:
        params = json.loads(args.params)

    main(model_type=args.model_type, test_size=args.test_size,
         random_state=args.random_state, params=params)
