import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import sys
import os
from dotenv import load_dotenv
import json

# Импорт функций из sns_ml_add_features
from sns_ml_add_features import load_and_add_features

# Импорт функций для работы с тестовыми данными и сохранения предсказаний
from sns_ml_fetch_data import fetch_test_data, save_predictions_to_sql

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

# Глобальный словарь для хранения лучших гиперпараметров по категориям
CATEGORY_BEST_PARAMS = {}

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
            'iterations': [500, 1000, 1500],
            'depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'loss_function': ['Huber:delta=1.345', 'RMSE', 'MAE']
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
                    'early_stopping_rounds': 30
                }
                
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
            early_stopping_rounds=50
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
        
        # Сохраняем лучшие параметры в глобальный словарь
        CATEGORY_BEST_PARAMS[cat_id] = best_params
    
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
        
        # Шаг 3: Обучение базовой модели (MAE)
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Обучение базовой модели CatBoost (HUBER)")
        logger.info("=" * 60)

        model_base, metrics_base = train_catboost_model(
            X, y,
            categorical_features=categorical_features,
            n_splits=5,
            iterations=1500,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            loss_function='Huber:delta=1.345',
            model_name="Базовая модель (Huber)"
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

        # Шаг 5: Подбор гиперпараметров для каждой категории
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 5: Подбор гиперпараметров для каждой категории")
        logger.info("=" * 60)

        category_models, category_metrics, category_results = train_catboost_per_category(
            df,
            target_col='SumRoubles',
            category_col='CategoryID',
            exclude_cols=['VisitDate']
        )

        # Сохранение лучших гиперпараметров по категориям в JSON файл
        params_file = 'category_best_params.json'
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(CATEGORY_BEST_PARAMS, f, indent=2, ensure_ascii=False)
        logger.info(f"Лучшие гиперпараметры сохранены в файл: {params_file}")

        # Вывод результатов подбора
        logger.info("\nРезультаты подбора гиперпараметров по категориям:")
        print(category_results.to_string(index=False))

        # Шаг 6: Сохранение глобальных моделей
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 6: Сохранение моделей")
        logger.info("=" * 60)

        base_model_path = 'catboost_model_base.cbm'
        new_model_path = 'catboost_model_new.cbm'

        save_model(model_base, base_model_path)
        save_model(model_new, new_model_path)

        # Сохранение моделей по категориям
        category_models_dir = 'category_models'
        os.makedirs(category_models_dir, exist_ok=True)
        for cat_id, model in category_models.items():
            cat_model_path = os.path.join(category_models_dir, f'catboost_model_cat_{cat_id}.cbm')
            save_model(model, cat_model_path)

        # Шаг 7: Вывод итоговой информации по моделям
        logger.info("\n" + "=" * 60)
        logger.info("Обучение моделей завершено успешно!")
        logger.info("=" * 60)

        # Шаг 8: Прогнозирование на тестовых данных (выполняется всегда)
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 8: Прогнозирование на тестовых данных")
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

        # Логирование предсказаний
        for model_name, (model_obj, predictions) in models_dict.items():
            logger.info(f"Модель '{model_name}': выполнено {len(predictions)} предсказаний")
            logger.info(f"  Среднее: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")

        # Шаг 9: Предсказание advanced с использованием моделей по категориям
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 9: Предсказание advanced (модели по категориям)")
        logger.info("=" * 60)

        predictions_advanced = np.zeros(len(test_df))
        logger.info(f"Выполнение предсказаний для {len(test_df)} записей...")

        # Для каждой категории используем свою модель
        for cat_id, cat_model in category_models.items():
            # Фильтруем записи для данной категории
            cat_mask = test_df['CategoryID'] == cat_id
            cat_count = cat_mask.sum()
            
            if cat_count > 0:
                logger.info(f"Категория {cat_id}: {cat_count} записей")
                
                # Получаем признаки для этой категории
                X_test_cat = X_test[cat_mask]
                
                # Делаем предсказание
                pred_cat = cat_model.predict(X_test_cat)
                predictions_advanced[cat_mask] = pred_cat
                
                logger.info(f"  Среднее предсказание: {np.mean(pred_cat):.4f}, Std: {np.std(pred_cat):.4f}")
        
        # Проверяем, остались ли записи без предсказания (категории, для которых не было модели)
        uncovered_mask = ~test_df['CategoryID'].isin(category_models.keys())
        uncovered_count = uncovered_mask.sum()
        if uncovered_count > 0:
            logger.warning(f"{uncovered_count} записей имеют категории, для которых нет обученной модели.")
            logger.warning("Используем базовую модель для этих записей...")
            # Используем базовую модель для категорий без своей модели
            predictions_advanced[uncovered_mask] = model_base.predict(X_test[uncovered_mask])

        # Добавляем advanced предсказания в словарь
        models_dict['advanced'] = (None, predictions_advanced)
        logger.info(f"Advanced модель: выполнено {len(predictions_advanced)} предсказаний")
        logger.info(f"  Среднее: {np.mean(predictions_advanced):.4f}, Std: {np.std(predictions_advanced):.4f}")

        # Расчет вероятности/уверенности прогноза (на основе улучшенной модели)
        predictions_new = models_dict['new'][1]
        mean_pred_new = np.mean(predictions_new)
        std_pred_new = np.std(predictions_new)
        confidence_new = np.exp(-np.abs(predictions_new - mean_pred_new) / (std_pred_new + 1e-6))

        # Сохранение результатов от всех моделей в одну таблицу базы данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 10: Сохранение результатов от всех моделей в SNS_ML_Predictions")
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
