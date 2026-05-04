import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional
import logging
import sys
import os
from dotenv import load_dotenv

# Импорт функций из sns_ml_add_features
from sns_ml_add_features import load_and_add_features

# Импорт функций для работы с тестовыми данными и сохранения предсказаний
from sns_ml_fetch_data import fetch_test_data, save_predictions_to_sql

# Импорты для CatBoost и машинного обучения
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Импорты для ансамблирования (XGBoost, LightGBM)
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    STACKING_AVAILABLE = True
except ImportError:
    STACKING_AVAILABLE = False
    logger.warning("XGBoost или LightGBM не установлены. Ансамблирование будет недоступно.")

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data_for_training(df: pd.DataFrame, 
                               target_col: str = 'SumRoubles',
                               exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Подготавливает данные для обучения модели.
    
    Args:
        df: Исходный датафрейм с признаками
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        exclude_cols: Список колонок для исключения из признаков (помимо target_col)
        
    Returns:
        Кортеж (X, y, categorical_features, feature_names):
            - X: Датафрейм с признаками
            - y: Серия с целевой переменной
            - categorical_features: Список индексов категориальных признаков
            - feature_names: Список имен признаков
    """
    logger.info("Подготовка данных для обучения модели...")
    
    # Создаем копию датафрейма
    df_clean = df.copy()
    
    # Удаляем строки с NaN в целевой переменной
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[target_col])
    if len(df_clean) < initial_rows:
        logger.info(f"Удалено {initial_rows - len(df_clean)} строк с NaN в целевой переменной")
    
    # Определяем колонки для исключения
    if exclude_cols is None:
        exclude_cols = []
    
    # Исключаем целевую переменную и указанные колонки
    cols_to_exclude = [target_col] + exclude_cols
    
    # Определяем признаки
    feature_cols = [col for col in df_clean.columns if col not in cols_to_exclude]
    
    # Определяем категориальные признаки
    # Типичные категориальные колонки в данных SNS
    categorical_feature_names = [
        'PointID', 'CategoryID', 'DayOfWeek', 'Quarter', 'Month', 'WeekOfYear',
        'IsFriday', 'IsMonday', 'IsPreHoliday', 'IsPostHoliday', 'isEndOfMonth',
        'BranchID', 'PointClass', 'PointType', 'MicroRegionID', 'LastSalesCategory'
    ]
    
    # Находим индексы категориальных признаков
    categorical_features = []
    for idx, col in enumerate(feature_cols):
        if col in categorical_feature_names:
            categorical_features.append(idx)
            logger.debug(f"Категориальный признак: {col} (индекс {idx})")
    
    # Создаем X и y
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    
    # Преобразуем целевую переменную в числовой тип (требуется для XGBoost/LightGBM)
    if y.dtype == 'object':
        logger.info(f"Преобразование целевой переменной '{target_col}' из object в numeric...")
        # Сначала пробуем заменить запятые на точки (для европейского формата чисел)
        y = y.astype(str).str.replace(',', '.').str.strip()
        y = pd.to_numeric(y, errors='coerce')
        # Удаляем строки с NaN после преобразования
        valid_mask = ~y.isna()
        if not valid_mask.all():
            logger.warning(f"Удалено {(~valid_mask).sum()} строк с нечисловыми значениями в целевой переменной")
            y = y[valid_mask]
            X = X[valid_mask]
    
    # Дополнительная проверка: убеждаемся, что y имеет правильный тип для sklearn/lightgbm/xgboost
    if str(y.dtype) not in ['int64', 'float64', 'int32', 'float32', 'bool']:
        logger.warning(f"Целевая переменная имеет тип {y.dtype}, пробуем преобразовать в float64...")
        y = y.astype('float64')
    
    # Преобразуем категориальные признаки в строковый тип для корректной обработки CatBoost
    # Заполняем NaN значением 'Unknown' перед конвертацией в строку
    for idx in categorical_features:
        col_name = feature_cols[idx]
        X[col_name] = X[col_name].fillna('Unknown').astype(str)
    
    logger.info(f"Количество признаков: {len(feature_cols)}")
    logger.info(f"Количество категориальных признаков: {len(categorical_features)}")
    logger.info(f"Количество наблюдений: {len(X)}")
    
    return X, y, categorical_features, feature_cols


def train_catboost_model(X: pd.DataFrame, 
                         y: pd.Series, 
                         categorical_features: List[int],
                         n_splits: int = 5,
                         iterations: int = 1000,
                         depth: int = 6,
                         learning_rate: float = 0.1,
                         random_seed: int = 42,
                         loss_function: str = 'MAE',
                         model_name: str = "") -> Tuple[CatBoostRegressor, dict]:
    """
    Обучает модель CatBoost с использованием кросс-валидации на временных рядах.
    
    Args:
        X: Датафрейм с признаками
        y: Серия с целевой переменной
        categorical_features: Список индексов категориальных признаков
        n_splits: Количество фолдов для кросс-валидации
        iterations: Количество итераций обучения
        depth: Максимальная глубина деревьев
        learning_rate: Скорость обучения
        random_seed: Случайное зерно
        loss_function: Функция потерь ('MAE' или 'RMSE')
        model_name: Имя модели для логгирования
        
    Returns:
        Кортеж (model, metrics):
            - model: Обученная модель CatBoost
            - metrics: Словарь с метриками качества модели
    """
    if model_name:
        logger.info(f"Обучение модели {model_name}...")
    else:
        logger.info("Обучение модели CatBoost...")
    
    # Параметры модели
    base_params = {
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'loss_function': loss_function,
        'eval_metric': loss_function,
        'verbose': 200,
        'cat_features': categorical_features if categorical_features else None,
        'random_seed': random_seed,
        'early_stopping_rounds': 50
    }
    
    # ==================== КРОСС-ВАЛИДАЦИЯ С ВРЕМЕННЫМИ РЯДАМИ ====================
    logger.info(f"Проведение кросс-валидации с временными рядами ({n_splits} folds)...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores_mae = []
    cv_scores_rmse = []
    cv_scores_r2 = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]
        
        # Создаем Pool для CatBoost
        train_pool = Pool(X_fold_train, y_fold_train, cat_features=categorical_features if categorical_features else None)
        val_pool = Pool(X_fold_val, y_fold_val, cat_features=categorical_features if categorical_features else None)
        
        model_cv = CatBoostRegressor(**base_params)
        model_cv.fit(train_pool, eval_set=val_pool, verbose=False)
        
        # Предсказания на валидации
        y_pred_fold = model_cv.predict(val_pool)
        
        # Метрики
        mae_fold = mean_absolute_error(y_fold_val, y_pred_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_fold_val, y_pred_fold))
        r2_fold = r2_score(y_fold_val, y_pred_fold)
        
        cv_scores_mae.append(mae_fold)
        cv_scores_rmse.append(rmse_fold)
        cv_scores_r2.append(r2_fold)
        
        logger.info(f"Fold {fold_idx+1}: MAE={mae_fold:.4f}, RMSE={rmse_fold:.4f}, R²={r2_fold:.4f}")
    
    # Средние метрики по кросс-валидации
    metrics = {
        'mae_mean': np.mean(cv_scores_mae),
        'mae_std': np.std(cv_scores_mae),
        'rmse_mean': np.mean(cv_scores_rmse),
        'rmse_std': np.std(cv_scores_rmse),
        'r2_mean': np.mean(cv_scores_r2),
        'r2_std': np.std(cv_scores_r2)
    }
    
    logger.info(f"Кросс-валидация завершена. Средние метрики:")
    logger.info(f"  MAE: {metrics['mae_mean']:.4f} (+/- {metrics['mae_std']:.4f})")
    logger.info(f"  RMSE: {metrics['rmse_mean']:.4f} (+/- {metrics['rmse_std']:.4f})")
    logger.info(f"  R²: {metrics['r2_mean']:.4f} (+/- {metrics['r2_std']:.4f})")
    
    # ==================== ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ВСЕЙ ВЫБОРКЕ ====================
    logger.info("Финальное обучение модели на всей выборке...")
    
    train_pool = Pool(X, y, cat_features=categorical_features if categorical_features else None)
    
    final_model = CatBoostRegressor(**base_params)
    final_model.fit(train_pool, verbose=200)
    
    logger.info("Модель успешно обучена!")
    
    return final_model, metrics


def train_stacking_ensemble(X: pd.DataFrame, 
                            y: pd.Series, 
                            categorical_features: List[int],
                            n_splits: int = 5,
                            random_seed: int = 42,
                            model_name: str = "Advanced") -> Tuple[dict, pd.DataFrame, dict]:
    """
    Обучает ансамбль моделей (CatBoost + XGBoost + LightGBM) с использованием стекинга.
    
    Стекинг работает следующим образом:
    1. Обучаются 3 базовые модели (CatBoost, XGBoost, LightGBM) с кросс-валидацией
    2. Для каждой модели получаем out-of-fold предсказания
    3. Эти предсказания используются как признаки для мета-модели (CatBoost)
    4. Мета-модель обучается на предсказаниях базовых моделей
    
    Args:
        X: Датафрейм с признаками
        y: Серия с целевой переменной
        categorical_features: Список индексов категориальных признаков
        n_splits: Количество фолдов для кросс-валидации
        random_seed: Случайное зерно
        model_name: Имя модели для логгирования
        
    Returns:
        Кортеж (base_models, meta_model, metrics):
            - base_models: Словарь с обученными базовыми моделями
            - meta_predictions: DataFrame с предсказаниями базовых моделей для мета-обучения
            - metrics: Словарь с метриками качества ансамбля
    """
    logger.info(f"Обучение модели {model_name} (Stacking Ensemble)...")
    
    if not STACKING_AVAILABLE:
        logger.error("XGBoost или LightGBM не установлены. Обучение ансамбля невозможно.")
        raise ImportError("XGBoost или LightGBM не установлены")
    
    # Преобразуем целевую переменную в числовой тип (требуется для XGBoost/LightGBM)
    # Эта проверка дублирует логику из prepare_data_for_training, но нужна для надежности
    if y.dtype == 'object':
        logger.info(f"Преобразование целевой переменной из object в numeric в train_stacking_ensemble...")
        y = y.astype(str).str.replace(',', '.').str.strip()
        y = pd.to_numeric(y, errors='coerce')
        if y.isna().any():
            logger.warning(f"В целевой переменной обнаружены нечисловые значения, они будут удалены")
            valid_mask = ~y.isna()
            y = y[valid_mask]
            X = X[valid_mask]
    
    # Дополнительная проверка типа
    if str(y.dtype) not in ['int64', 'float64', 'int32', 'float32', 'bool']:
        logger.warning(f"Целевая переменная имеет тип {y.dtype}, преобразуем в float64...")
        y = y.astype('float64')
    
    # Создаем копию данных с кодированными категориальными признаками для XGBoost/LightGBM
    # CatBoost работает со строковыми категориальными признаками, а XGBoost/LightGBM требуют числовые
    X_encoded = X.copy()
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    for idx in categorical_features:
        col_name = X.columns[idx]
        le = LabelEncoder()
        # Заполняем пропуски перед кодированием
        X_encoded[col_name] = X_encoded[col_name].fillna('Unknown').astype(str)
        X_encoded[col_name] = le.fit_transform(X_encoded[col_name])
        label_encoders[col_name] = le
        logger.debug(f"Кодирован признак {col_name}: {len(le.classes_)} уникальных значений")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Параметры базовых моделей
    catboost_params = {
        'iterations': 1500,
        'depth': 8,
        'learning_rate': 0.05,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': False,
        'cat_features': categorical_features if categorical_features else None,
        'random_seed': random_seed,
        'early_stopping_rounds': 50
    }
    
    xgboost_params = {
        'n_estimators': 1500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': random_seed,
        'early_stopping_rounds': 50,
        'verbosity': 0
    }
    
    lightgbm_params = {
        'n_estimators': 1500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': random_seed,
        'early_stopping_rounds': 50,
        'verbose': -1,
        'categorical_feature': categorical_features if categorical_features else 'auto'
    }
    
    # Инициализация моделей
    catboost_model = CatBoostRegressor(**catboost_params)
    xgboost_model = XGBRegressor(**xgboost_params)
    lightgbm_model = LGBMRegressor(**lightgbm_params)
    
    base_models = {
        'catboost': catboost_model,
        'xgboost': xgboost_model,
        'lightgbm': lightgbm_model
    }
    
    # Out-of-fold предсказания для каждой модели
    oof_predictions = {name: np.zeros(len(X)) for name in base_models.keys()}
    
    # Метрики для каждой модели
    model_metrics = {name: {'mae': [], 'rmse': [], 'r2': []} for name in base_models.keys()}
    
    logger.info(f"Проведение кросс-валидации для базовых моделей ({n_splits} folds)...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        X_fold_train_encoded = X_encoded.iloc[train_idx]
        X_fold_val_encoded = X_encoded.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]
        
        logger.info(f"\nFold {fold_idx+1}:")
        
        # CatBoost (работает с оригинальными строковыми данными и cat_features)
        train_pool_cb = Pool(X_fold_train, y_fold_train, cat_features=categorical_features if categorical_features else None)
        val_pool_cb = Pool(X_fold_val, y_fold_val, cat_features=categorical_features if categorical_features else None)
        catboost_model.fit(train_pool_cb, eval_set=val_pool_cb, verbose=False)
        oof_predictions['catboost'][val_idx] = catboost_model.predict(val_pool_cb)
        
        # XGBoost (работает с кодированными числовыми данными)
        xgboost_model.fit(X_fold_train_encoded, y_fold_train, eval_set=[(X_fold_val_encoded, y_fold_val)], verbose=False)
        oof_predictions['xgboost'][val_idx] = xgboost_model.predict(X_fold_val_encoded)
        
        # LightGBM (работает с кодированными числовыми данными)
        lightgbm_model.fit(X_fold_train_encoded, y_fold_train, eval_set=[(X_fold_val_encoded, y_fold_val)])
        oof_predictions['lightgbm'][val_idx] = lightgbm_model.predict(X_fold_val_encoded)
        
        # Метрики для каждого fold
        for name, preds in oof_predictions.items():
            fold_preds = preds[val_idx]
            mae_fold = mean_absolute_error(y_fold_val, fold_preds)
            rmse_fold = np.sqrt(mean_squared_error(y_fold_val, fold_preds))
            r2_fold = r2_score(y_fold_val, fold_preds)
            model_metrics[name]['mae'].append(mae_fold)
            model_metrics[name]['rmse'].append(rmse_fold)
            model_metrics[name]['r2'].append(r2_fold)
            logger.info(f"  {name}: MAE={mae_fold:.4f}, RMSE={rmse_fold:.4f}, R²={r2_fold:.4f}")
    
    # Создаем DataFrame с out-of-fold предсказаниями для мета-обучения
    meta_df = pd.DataFrame(oof_predictions)
    meta_df.columns = [f'pred_{name}' for name in meta_df.columns]
    
    # Обучаем финальные модели на всех данных
    logger.info("\nФинальное обучение базовых моделей на всей выборке...")
    
    train_pool_full = Pool(X, y, cat_features=categorical_features if categorical_features else None)
    
    # CatBoost на полных данных (работает с оригинальными строковыми данными)
    catboost_final = CatBoostRegressor(**catboost_params)
    catboost_final.fit(train_pool_full, verbose=200)
    base_models['catboost'] = catboost_final
    
    # XGBoost на полных данных (работает с кодированными числовыми данными)
    xgboost_final = XGBRegressor(**xgboost_params)
    xgboost_final.fit(X_encoded, y, verbose=False)
    base_models['xgboost'] = xgboost_final
    
    # LightGBM на полных данных (работает с кодированными числовыми данными)
    lightgbm_final = LGBMRegressor(**lightgbm_params)
    lightgbm_final.fit(X_encoded, y)
    base_models['lightgbm'] = lightgbm_final
    
    # Мета-модель (CatBoost) обучается на предсказаниях базовых моделей
    logger.info("Обучение мета-модели...")
    meta_params = {
        'iterations': 500,
        'depth': 4,
        'learning_rate': 0.1,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': 200,
        'random_seed': random_seed,
        'early_stopping_rounds': 30
    }
    
    meta_model = CatBoostRegressor(**meta_params)
    meta_pool = Pool(meta_df, y, cat_features=None)
    meta_model.fit(meta_pool, verbose=200)
    
    # Финальные предсказания ансамбля через мета-модель
    ensemble_predictions = meta_model.predict(meta_df)
    
    # Расчет итоговых метрик ансамбля
    ensemble_mae = mean_absolute_error(y, ensemble_predictions)
    ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_predictions))
    ensemble_r2 = r2_score(y, ensemble_predictions)
    
    metrics = {
        'ensemble_mae': ensemble_mae,
        'ensemble_rmse': ensemble_rmse,
        'ensemble_r2': ensemble_r2,
        'base_models_metrics': {}
    }
    
    for name in base_models.keys():
        metrics['base_models_metrics'][name] = {
            'mae_mean': np.mean(model_metrics[name]['mae']),
            'rmse_mean': np.mean(model_metrics[name]['rmse']),
            'r2_mean': np.mean(model_metrics[name]['r2'])
        }
    
    logger.info(f"\nМетрики ансамбля {model_name}:")
    logger.info(f"  MAE: {ensemble_mae:.4f}")
    logger.info(f"  RMSE: {ensemble_rmse:.4f}")
    logger.info(f"  R²: {ensemble_r2:.4f}")
    
    logger.info("\nМетрики базовых моделей (CV среднее):")
    for name, m in metrics['base_models_metrics'].items():
        logger.info(f"  {name}: MAE={m['mae_mean']:.4f}, RMSE={m['rmse_mean']:.4f}, R²={m['r2_mean']:.4f}")
    
    return base_models, meta_model, meta_df, metrics


def save_model(model: CatBoostRegressor, model_path: str = 'catboost_model.cbm') -> None:
    """
    Сохраняет обученную модель CatBoost в файл.
    
    Args:
        model: Обученная модель CatBoost
        model_path: Путь для сохранения модели
    """
    model.save_model(model_path)
    logger.info(f"Модель сохранена в файл: {model_path}")


def load_model(model_path: str = 'catboost_model.cbm') -> CatBoostRegressor:
    """
    Загружает модель CatBoost из файла.
    
    Args:
        model_path: Путь к файлу модели
        
    Returns:
        Загруженная модель CatBoost
    """
    model = CatBoostRegressor()
    model.load_model(model_path)
    logger.info(f"Модель загружена из файла: {model_path}")
    return model


def main():
    """
    Основная функция для обучения модели CatBoost и прогнозирования на тестовых данных.
    """
    logger.info("=" * 60)
    logger.info("Запуск обучения модели CatBoost для прогнозирования SumRoubles")
    logger.info("=" * 60)
    
    # Установка дат: end_date = текущая дата, start_date = end_date - 1 год
    today = date.today()
    end_date = today
    start_date = end_date - timedelta(days=365)
    
    # Параметры из командной строки
    # Использование: python sns_ml_model.py [start_date] [end_date] [model_path]
    if len(sys.argv) > 1 and sys.argv[1] not in ['--predict', '-p']:
        try:
            start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Неверный формат даты start_date: {sys.argv[1]}. Используйте YYYY-MM-DD")
            sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] not in ['--predict', '-p']:
        try:
            end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Неверный формат даты end_date: {sys.argv[2]}. Используйте YYYY-MM-DD")
            sys.exit(1)
    
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'catboost_model.cbm'
    
    logger.info(f"Период обучения: {start_date} - {end_date}")
    logger.info(f"Путь сохранения модели: {model_path}")
    
    try:
        # Шаг 1: Загрузка данных с признаками
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 1: Загрузка данных с признаками")
        logger.info("=" * 60)
        
        df = load_and_add_features(start_date, end_date)
        
        logger.info(f"Загружено {len(df)} записей")
        logger.info(f"Колонки в датасете: {list(df.columns)}")
        
        # Шаг 2: Подготовка данных для обучения
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 2: Подготовка данных для обучения")
        logger.info("=" * 60)
        
        # Исключаем служебные колонки, если они есть
        exclude_cols = ['VisitDate']  # Исключаем дату, т.к. она уже преобразована в признаки
        
        # Проверяем наличие категориальных колонок с NaN и заполняем их перед подготовкой
        categorical_cols_to_check = ['PointID', 'CategoryID', 'BranchID', 'PointClass', 'PointType', 'MicroRegionID']
        for col in categorical_cols_to_check:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.info(f"Заполнено {nan_count} NaN в колонке {col} значением 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
        
        X, y, categorical_features, feature_names = prepare_data_for_training(
            df, 
            target_col='SumRoubles',
            exclude_cols=exclude_cols
        )
        
        # Шаг 3: Обучение базовой модели (MAE)
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Обучение базовой модели CatBoost (MAE)")
        logger.info("=" * 60)

        model_base, metrics_base = train_catboost_model(
            X, y,
            categorical_features=categorical_features,
            n_splits=5,
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            loss_function='MAE',
            model_name="Базовая модель (MAE)"
        )

        # Шаг 4: Обучение улучшенной модели (RMSE с другими параметрами)
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 4: Обучение улучшенной модели CatBoost (RMSE)")
        logger.info("=" * 60)

        model_new, metrics_new = train_catboost_model(
            X, y,
            categorical_features=categorical_features,
            n_splits=5,
            iterations=1500,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            loss_function='RMSE',
            model_name="Улучшенная модель (RMSE)"
        )

        # Шаг 5: Сохранение моделей
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 5: Сохранение моделей")
        logger.info("=" * 60)

        base_model_path = 'catboost_model_base.cbm'
        new_model_path = 'catboost_model_new.cbm'

        save_model(model_base, base_model_path)
        save_model(model_new, new_model_path)

        # Шаг 6: Обучение модели Advanced (Stacking Ensemble)
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 6: Обучение модели Advanced (Stacking Ensemble)")
        logger.info("=" * 60)

        if STACKING_AVAILABLE:
            try:
                base_models, meta_model, meta_df, metrics_advanced = train_stacking_ensemble(
                    X, y,
                    categorical_features=categorical_features,
                    n_splits=5,
                    random_seed=42,
                    model_name="Advanced"
                )
                
                logger.info("\n" + "=" * 60)
                logger.info("Метрики модели Advanced (Stacking Ensemble)")
                logger.info("=" * 60)
                logger.info(f"MAE: {metrics_advanced['ensemble_mae']:.4f}")
                logger.info(f"RMSE: {metrics_advanced['ensemble_rmse']:.4f}")
                logger.info(f"R²: {metrics_advanced['ensemble_r2']:.4f}")
                
                # Сохраняем мета-модель
                advanced_model_path = 'catboost_model_advanced.cbm'
                save_model(meta_model, advanced_model_path)
                logger.info(f"Мета-модель Advanced сохранена в файл: {advanced_model_path}")
                
            except Exception as e:
                logger.error(f"Ошибка при обучении ансамбля: {e}")
                logger.warning("Продолжаем работу без модели Advanced")
                base_models = None
                meta_model = None
                meta_df = None
        else:
            logger.warning("XGBoost или LightGBM не установлены. Пропускаем обучение модели Advanced.")
            base_models = None
            meta_model = None
            meta_df = None

        # Шаг 7: Вывод итоговой информации по всем моделям
        logger.info("\n" + "=" * 60)
        logger.info("Обучение всех моделей завершено успешно!")
        logger.info("=" * 60)

        # Шаг 7: Прогнозирование на тестовых данных (выполняется всегда)
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 7: Прогнозирование на тестовых данных всеми моделями")
        logger.info("=" * 60)

        # Загружаем тестовые данные на текущую дату
        test_date = today
        logger.info(f"Загрузка тестовых данных на дату: {test_date}")

        test_df = fetch_test_data(test_date)
        logger.info(f"Загружено {len(test_df)} строк тестовых данных")
        logger.info(f"Колонки тестового набора: {list(test_df.columns)}")

        # Преобразование данных для совместимости с моделью
        # Убеждаемся, что все необходимые признаки присутствуют
        logger.info("Преобразование тестовых данных для прогнозирования...")

        # Проверяем наличие всех признаков из обучения
        missing_cols = set(feature_names) - set(test_df.columns)
        if missing_cols:
            logger.warning(f"Отсутствуют колонки в тестовых данных: {missing_cols}")
            # Добавляем отсутствующие колонки со значением 0 или другим дефолтным
            for col in missing_cols:
                test_df[col] = 0

        # Отбираем только нужные колонки в правильном порядке
        X_test = test_df[feature_names].copy()

        # Преобразуем категориальные признаки в строковый тип (как при обучении)
        for idx in categorical_features:
            col_name = feature_names[idx]
            X_test[col_name] = X_test[col_name].fillna('Unknown').astype(str)

        # Словарь для хранения всех моделей и их предсказаний
        # Формат: {model_name: (model_object, predictions_array)}
        models_dict = {
            'base': (model_base, model_base.predict(X_test)),
            'new': (model_new, model_new.predict(X_test))
        }

        # Добавляем предсказания модели Advanced (Stacking Ensemble)
        if STACKING_AVAILABLE and base_models is not None and meta_model is not None:
            logger.info("\nГенерация предсказаний модели Advanced...")
            
            # Для CatBoost используем оригинальные строковые данные
            pred_catboost = base_models['catboost'].predict(X_test)
            
            # Для XGBoost и LightGBM нужно закодировать категориальные признаки
            X_test_encoded = X_test.copy()
            for idx in categorical_features:
                col_name = feature_names[idx]
                # Используем тот же LabelEncoder, что и при обучении
                # Если категория не была в обучении, используем 0 (или можно использовать 'Unknown')
                try:
                    le = label_encoders.get(col_name)
                    if le:
                        # Преобразуем значения, которые были в обучении
                        X_test_encoded[col_name] = X_test_encoded[col_name].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                except Exception as e:
                    logger.warning(f"Ошибка кодирования признака {col_name}: {e}")
                    X_test_encoded[col_name] = 0
            
            pred_xgboost = base_models['xgboost'].predict(X_test_encoded)
            pred_lightgbm = base_models['lightgbm'].predict(X_test_encoded)
            
            # Создаем DataFrame для мета-модели
            meta_test_df = pd.DataFrame({
                'pred_catboost': pred_catboost,
                'pred_xgboost': pred_xgboost,
                'pred_lightgbm': pred_lightgbm
            })
            
            # Получаем финальное предсказание ансамбля через мета-модель
            predictions_advanced = meta_model.predict(meta_test_df)
            
            models_dict['advanced'] = (meta_model, predictions_advanced)
            logger.info(f"Модель 'advanced': выполнено {len(predictions_advanced)} предсказаний")
            logger.info(f"  Среднее: {np.mean(predictions_advanced):.4f}, Std: {np.std(predictions_advanced):.4f}")

        # Логирование предсказаний
        for model_name, (model_obj, predictions) in models_dict.items():
            if model_name not in ['advanced']:  # advanced уже залогирован
                logger.info(f"Модель '{model_name}': выполнено {len(predictions)} предсказаний")
                logger.info(f"  Среднее: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")

        # Расчет вероятности/уверенности прогноза (на основе улучшенной модели)
        predictions_new = models_dict['new'][1]
        mean_pred_new = np.mean(predictions_new)
        std_pred_new = np.std(predictions_new)
        confidence_new = np.exp(-np.abs(predictions_new - mean_pred_new) / (std_pred_new + 1e-6))

        # Сохранение результатов от всех моделей в одну таблицу базы данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 8: Сохранение результатов от всех моделей в SNS_ML_Predictions")
        logger.info("=" * 60)

        # Создаем результирующий датафрейм для сохранения
        # Сначала копируем все фичи из тестовых данных
        result_df = test_df.copy()

        # Добавляем предсказания от каждой модели в порядке добавления в словарь
        for model_name, (model_obj, predictions) in models_dict.items():
            col_name = f'predict_{model_name}'
            result_df[col_name] = predictions
            logger.info(f"Добавлены предсказания модели '{model_name}' в колонку {col_name}")

        # Добавляем служебные колонки
        result_df['CreatedAt'] = datetime.now()
        
        # Сохраняем в таблицу
        save_predictions_to_sql(result_df, table_name='SNS_ML_Predictions')

        logger.info("\n" + "=" * 60)
        logger.info("Прогнозирование и сохранение результатов завершены успешно!")
        logger.info(f"Результаты сохранены в таблицу: SNS_ML_Predictions")
        logger.info(f"Количество колонок с предсказаниями: {len(models_dict)}")
        logger.info(f"Имена колонок предсказаний: {[f'predict_{name}' for name in models_dict.keys()]}")
        logger.info("=" * 60)

        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
