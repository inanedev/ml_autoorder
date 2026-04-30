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

# Импорты для CatBoost и машинного обучения
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        'BranchID', 'PointClass', 'PointType', 'MicroRegionID'
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
                         random_seed: int = 42) -> Tuple[CatBoostRegressor, dict]:
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
        
    Returns:
        Кортеж (model, metrics):
            - model: Обученная модель CatBoost
            - metrics: Словарь с метриками качества модели
    """
    logger.info("Обучение модели CatBoost...")
    
    # Параметры модели
    base_params = {
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'loss_function': 'MAE',  # MAE более устойчив к выбросам
        'eval_metric': 'MAE',
        'verbose': 100,
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
    final_model.fit(train_pool, verbose=True)
    
    logger.info("Модель успешно обучена!")
    
    return final_model, metrics


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
    Основная функция для обучения модели CatBoost.
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
    if len(sys.argv) > 1:
        try:
            start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Неверный формат даты start_date: {sys.argv[1]}. Используйте YYYY-MM-DD")
            sys.exit(1)
    
    if len(sys.argv) > 2:
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
        
        # Шаг 3: Обучение модели
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Обучение модели CatBoost")
        logger.info("=" * 60)
        
        model, metrics = train_catboost_model(
            X, y, 
            categorical_features=categorical_features,
            n_splits=5,
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            random_seed=42
        )
        
        # Шаг 4: Сохранение модели
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 4: Сохранение модели")
        logger.info("=" * 60)
        
        save_model(model, model_path)
        
        # Шаг 5: Вывод итоговой информации
        logger.info("\n" + "=" * 60)
        logger.info("Итоговые метрики модели")
        logger.info("=" * 60)
        logger.info(f"MAE: {metrics['mae_mean']:.4f} (+/- {metrics['mae_std']:.4f})")
        logger.info(f"RMSE: {metrics['rmse_mean']:.4f} (+/- {metrics['rmse_std']:.4f})")
        logger.info(f"R²: {metrics['r2_mean']:.4f} (+/- {metrics['r2_std']:.4f})")
        
        # Важность признаков
        logger.info("\n" + "=" * 60)
        logger.info("Важность признаков (топ-15)")
        logger.info("=" * 60)
        
        feature_importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(importance_df.head(15).to_string(index=False))
        
        logger.info("\n" + "=" * 60)
        logger.info("Обучение модели завершено успешно!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
