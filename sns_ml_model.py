import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict
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
from sklearn.model_selection import TimeSeriesSplit, KFold, RandomizedSearchCV
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
    
    # Преобразуем целевую переменную в числовой тип
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
    
    # Дополнительная проверка: убеждаемся, что y имеет правильный тип для sklearn
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
                         iterations: int = 1500,
                         depth: int = 8,
                         learning_rate: float = 0.05,
                         random_seed: int = 42,
                         loss_function: str = 'Huber:delta=1.345',
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


def tune_huber_alpha(X: pd.DataFrame, 
                     y: pd.Series, 
                     categorical_features: List[int],
                     n_iter: int = 20,
                     cv_splits: int = 3,
                     random_seed: int = 42,
                     base_iterations: int = 1500,
                     base_depth: int = 10,
                     base_learning_rate: float = 0.05) -> Tuple[float, CatBoostRegressor, dict]:
    """
    Подбирает оптимальный коэффициент alpha (delta) для функции потерь Huber 
    с помощью RandomizedSearchCV.
    
    Args:
        X: Датафрейм с признаками
        y: Серия с целевой переменной
        categorical_features: Список индексов категориальных признаков
        n_iter: Количество итераций поиска
        cv_splits: Количество фолдов для кросс-валидации
        random_seed: Случайное зерно
        base_iterations: Базовое количество итераций обучения (по умолчанию 1500)
        base_depth: Базовая глубина деревьев (по умолчанию 10)
        base_learning_rate: Базовая скорость обучения (по умолчанию 0.05)
        
    Returns:
        Кортеж (best_alpha, best_model, search_results):
            - best_alpha: Лучшее значение delta для Huber
            - best_model: Модель с лучшими параметрами
            - search_results: Словарь с результатами поиска
    """
    logger.info("=" * 60)
    logger.info("Подбор оптимального коэффициента alpha (delta) для Huber")
    logger.info("Использование RandomizedSearchCV")
    logger.info("=" * 60)
    
    # Параметр delta в CatBoost указывается как Huber:delta=value
    # Будем подбирать delta в диапазоне от 0.5 до 3.0
    param_distributions = {
        'loss_function': ['Huber:delta=0.5', 'Huber:delta=0.75', 'Huber:delta=1.0', 
                          'Huber:delta=1.25', 'Huber:delta=1.345', 'Huber:delta=1.5',
                          'Huber:delta=1.75', 'Huber:delta=2.0', 'Huber:delta=2.5', 'Huber:delta=3.0']
    }
    
    # Базовые параметры модели
    base_params = {
        'iterations': base_iterations,
        'depth': base_depth,
        'learning_rate': base_learning_rate,
        'eval_metric': 'MAE',
        'verbose': False,
        'cat_features': categorical_features if categorical_features else None,
        'random_seed': random_seed,
        'early_stopping_rounds': 50,
        'allow_writing_files': False
    }
    
    # Используем TimeSeriesSplit для временных рядов
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Выполняем поиск по сетке вручную, т.к. CatBoost не полностью совместим со sklearn API
    logger.info(f"Выполнение {n_iter} итераций RandomizedSearchCV...")
    
    best_score = float('inf')
    best_alpha = None
    best_model = None
    all_results = []
    
    # Для случайного выбора параметров
    np.random.seed(random_seed)
    
    for iter_idx in range(n_iter):
        # Случайный выбор delta из диапазона [0.5, 3.0]
        delta = np.random.uniform(0.5, 3.0)
        loss_function = f'Huber:delta={delta:.3f}'
        
        logger.info(f"\nИтерация {iter_idx + 1}/{n_iter}: delta = {delta:.3f}")
        
        # Кросс-валидация для текущих параметров
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_fold_train = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]
            
            train_pool = Pool(X_fold_train, y_fold_train, cat_features=categorical_features if categorical_features else None)
            val_pool = Pool(X_fold_val, y_fold_val, cat_features=categorical_features if categorical_features else None)
            
            params = {**base_params, 'loss_function': loss_function}
            
            model_cv = CatBoostRegressor(**params)
            model_cv.fit(train_pool, eval_set=val_pool, verbose=False)
            
            y_pred_fold = model_cv.predict(val_pool)
            mae_fold = mean_absolute_error(y_fold_val, y_pred_fold)
            cv_scores.append(mae_fold)
            
            logger.debug(f"  Fold {fold_idx + 1}: MAE = {mae_fold:.4f}")
        
        avg_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        logger.info(f"  Средний MAE: {avg_score:.4f} (+/- {std_score:.4f})")
        
        all_results.append({
            'delta': delta,
            'loss_function': loss_function,
            'mae_mean': avg_score,
            'mae_std': std_score
        })
        
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = delta
            best_loss_function = loss_function
    
    # ==================== ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ВСЕЙ ВЫБОРКЕ ====================
    # Обучаем модель один раз с лучшим найденным параметром
    logger.info(f"\\nФинальное обучение модели с лучшим delta={best_alpha:.3f} на всей выборке...")
    train_pool_full = Pool(X, y, cat_features=categorical_features if categorical_features else None)
    best_model = CatBoostRegressor(**{**base_params, 'loss_function': best_loss_function})
    best_model.fit(train_pool_full, verbose=200)
    
    # Сортируем результаты по MAE
    all_results.sort(key=lambda x: x['mae_mean'])
    
    logger.info("\n" + "=" * 60)
    logger.info("Результаты подбора гиперпараметра Huber:")
    logger.info("Топ-5 значений delta:")
    for i, result in enumerate(all_results[:5]):
        logger.info(f"  {i+1}. delta={result['delta']:.3f}: MAE={result['mae_mean']:.4f} (+/- {result['mae_std']:.4f})")
    
    logger.info("=" * 60)
    logger.info(f"Лучшее значение delta: {best_alpha:.3f}")
    logger.info(f"Лучший средний MAE: {best_score:.4f}")
    logger.info("=" * 60)
    
    search_results = {
        'best_delta': best_alpha,
        'best_mae': best_score,
        'all_results': all_results,
        'n_iterations': n_iter
    }
    
    return best_alpha, best_model, search_results


def train_catboost_per_category(df: pd.DataFrame,
                                 target_col: str = 'SumRoubles',
                                 category_col: str = 'CategoryID',
                                 exclude_cols: Optional[List[str]] = None,
                                 param_grid: Optional[Dict] = None) -> Tuple[Dict[int, CatBoostRegressor], Dict[int, dict], pd.DataFrame]:
    """
    Обучает отдельные модели CatBoost для каждой категории с подбором гиперпараметров.
    
    Args:
        df: Исходный датафрейм с признаками и целевой переменной
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        category_col: Название колонки категории (по умолчанию 'CategoryID')
        exclude_cols: Список колонок для исключения из признаков
        param_grid: Сетка гиперпараметров для подбора
        
    Returns:
        Кортеж (models_dict, metrics_dict, results_df):
            - models_dict: Словарь {category_id: model}
            - metrics_dict: Словарь {category_id: {'mae': ..., 'rmse': ..., 'r2': ..., 'best_params': ...}}
            - results_df: Датафрейм с лучшими гиперпараметрами по категориям
    """
    logger.info("=" * 60)
    logger.info("Подбор гиперпараметров для каждой категории")
    logger.info("=" * 60)
    
    # Параметры по умолчанию для подбора
    if param_grid is None:
        param_grid = {
            'iterations': [700, 1000, 1500],
            'depth': [8, 10],
            'learning_rate': [0.03, 0.05 ,0.1],
            'loss_function': ['RMSE', 'MAE']
        }
    
    # Получаем уникальные категории
    categories = df[category_col].unique()
    logger.info(f"Найдено {len(categories)} уникальных категорий: {sorted(categories)}")
    
    models_dict = {}
    metrics_dict = {}
    results_list = []
    
    for cat_id in categories:
        logger.info(f"\n{'='*60}")
        logger.info(f"Обработка категории {cat_id}")
        logger.info(f"{'='*60}")
        
        # Фильтруем данные по категории
        cat_df = df[df[category_col] == cat_id].copy()
        logger.info(f"Количество записей для категории {cat_id}: {len(cat_df)}")
        
        if len(cat_df) < 100:
            logger.warning(f"Категория {cat_id} имеет слишком мало данных ({len(cat_df)}). Пропускаем.")
            continue
        
        # Подготавливаем данные для обучения
        X_cat, y_cat, cat_features_idx, feature_names = prepare_data_for_training(
            cat_df, 
            target_col=target_col,
            exclude_cols=exclude_cols or ['VisitDate']
        )
        
        if len(X_cat) < 100:
            logger.warning(f"Категория {cat_id} имеет слишком мало данных после подготовки ({len(X_cat)}). Пропускаем.")
            continue
        
        # Подбор гиперпараметров
        best_score = float('inf')
        best_params = None
        best_model = None
        best_metrics = None
        
        # Перебираем комбинации гиперпараметров
        from itertools import product
        param_combinations = list(product(
            param_grid['iterations'],
            param_grid['depth'],
            param_grid['learning_rate'],
            param_grid['loss_function']
        ))
        
        logger.info(f"Всего комбинаций гиперпараметров: {len(param_combinations)}")
        
        for iterations, depth, learning_rate, loss_function in param_combinations:
            # Упрощенная кросс-валидация (3 фолда для скорости)
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_cat)):
                X_fold_train = X_cat.iloc[train_idx]
                X_fold_val = X_cat.iloc[val_idx]
                y_fold_train = y_cat.iloc[train_idx]
                y_fold_val = y_cat.iloc[val_idx]
                
                train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_features_idx if cat_features_idx else None)
                val_pool = Pool(X_fold_val, y_fold_val, cat_features=cat_features_idx if cat_features_idx else None)
                
                params = {
                    'iterations': iterations,
                    'depth': depth,
                    'learning_rate': learning_rate,
                    'loss_function': loss_function,
                    'eval_metric': 'MAE',
                    'verbose': False,
                    'cat_features': cat_features_idx if cat_features_idx else None,
                    'random_seed': 42,
                    'early_stopping_rounds': 30,
                    'allow_writing_files' : False
                }
                print(f'{fold_idx} - номер фолда, {params} - параметры')
                model_cv = CatBoostRegressor(**params)
                model_cv.fit(train_pool, eval_set=val_pool, verbose=False)
                
                y_pred_fold = model_cv.predict(val_pool)
                mae_fold = mean_absolute_error(y_fold_val, y_pred_fold)
                cv_scores.append(mae_fold)
            
            avg_score = np.mean(cv_scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = {
                    'iterations': iterations,
                    'depth': depth,
                    'learning_rate': learning_rate,
                    'loss_function': loss_function
                }
        
        logger.info(f"Лучшие гиперпараметры для категории {cat_id}: {best_params}")
        logger.info(f"Лучший MAE: {best_score:.4f}")
        
        # Финальное обучение на всей выборке с лучшими параметрами
        train_pool = Pool(X_cat, y_cat, cat_features=cat_features_idx if cat_features_idx else None)
        final_model = CatBoostRegressor(
            **best_params,
            verbose=200,
            random_seed=42,
            early_stopping_rounds=50,
            allow_writing_files=False
        )
        final_model.fit(train_pool)
        
        # Расчет метрик на обучающей выборке
        y_pred_train = final_model.predict(train_pool)
        mae_train = mean_absolute_error(y_cat, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_cat, y_pred_train))
        r2_train = r2_score(y_cat, y_pred_train)
        
        logger.info(f"Метрики на обучающей выборке: MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R²={r2_train:.4f}")
        
        # Сохраняем модель и метрики
        models_dict[cat_id] = final_model
        metrics_dict[cat_id] = {
            'mae': mae_train,
            'rmse': rmse_train,
            'r2': r2_train,
            'best_params': best_params
        }
        
        # Добавляем в результаты
        results_list.append({
            'CategoryID': cat_id,
            'Iterations': best_params['iterations'],
            'Depth': best_params['depth'],
            'LearningRate': best_params['learning_rate'],
            'LossFunction': best_params['loss_function'],
            'MAE': mae_train,
            'RMSE': rmse_train,
            'R2': r2_train,
            'NumSamples': len(X_cat)
        })
    
    # Создаем датафрейм с результатами
    results_df = pd.DataFrame(results_list)
    
    logger.info("\n" + "=" * 60)
    logger.info("Подбор гиперпараметров завершен")
    logger.info(f"Обучено моделей: {len(models_dict)}")
    logger.info("=" * 60)
    
    return models_dict, metrics_dict, results_df


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
        
        # Шаг 3: Подбор оптимального коэффициента delta для Huber через RandomizedSearchCV
        # Фиксированные гиперпараметры: Iterations=1500, Depth=10, LearningRate=0.05
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Подбор оптимального коэффициента delta для Huber (RandomizedSearchCV)")
        logger.info("Фиксированные параметры: Iterations=1500, Depth=10, LearningRate=0.05")
        logger.info("=" * 60)

        best_alpha, final_model, huber_search_results = tune_huber_alpha(
            X, y,
            categorical_features=categorical_features,
            n_iter=20,
            cv_splits=3,
            random_seed=42,
            base_iterations=1500,
            base_depth=10,
            base_learning_rate=0.05
        )

        # Обновляем метрики модели с лучшим delta
        optimal_loss_function = f'Huber:delta={best_alpha:.3f}'
        logger.info(f"Использование оптимальной функции потерь: {optimal_loss_function}")

        # Сохранение лучшей модели
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 4: Сохранение модели")
        logger.info("=" * 60)

        save_model(final_model, model_path)

        # Шаг 5: Вывод итоговой информации по модели
        logger.info("\n" + "=" * 60)
        logger.info("Обучение модели завершено успешно!")
        logger.info("=" * 60)

        # Шаг 6: Прогнозирование на тестовых данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 6: Прогнозирование на тестовых данных")
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

        # Делаем предсказание единственной моделью
        predictions = final_model.predict(X_test)

        logger.info(f"Модель: выполнено {len(predictions)} предсказаний")
        logger.info(f"  Среднее: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")

        # Расчет вероятности/уверенности прогноза
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        confidence = np.exp(-np.abs(predictions - mean_pred) / (std_pred + 1e-6))

        # Сохранение результатов в таблицу базы данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 7: Сохранение результатов в SNS_ML_Predictions")
        logger.info("=" * 60)

        # Создаем результирующий датафрейм для сохранения
        # Сначала копируем все фичи из тестовых данных
        result_df = test_df.copy()

        # Добавляем предсказания модели
        result_df['predict'] = predictions
        logger.info("Добавлены предсказания модели в колонку predict")

        # Добавляем служебные колонки
        result_df['CreatedAt'] = datetime.now()
        
        # Сохраняем в таблицу
        save_predictions_to_sql(result_df, table_name='SNS_ML_Predictions')

        logger.info("\n" + "=" * 60)
        logger.info("Прогнозирование и сохранение результатов завершены успешно!")
        logger.info(f"Результаты сохранены в таблицу: SNS_ML_Predictions")
        logger.info("=" * 60)

        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
