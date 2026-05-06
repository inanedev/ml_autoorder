import pandas as pd
import numpy as np
import math
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
from sklearn.model_selection import TimeSeriesSplit, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RMSLE(object):
    """
    Пользовательская функция потерь RMSLE для CatBoost.
    Обучает модель напрямую минимизировать RMSLE.
    """
    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            val = max(approxes[index], 0)
            der1 = math.log1p(targets[index]) - math.log1p(max(0, approxes[index]))
            der2 = -1 / (max(0, approxes[index]) + 1)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


class RMSLE_val(object):
    """
    Пользовательская метрика RMSLE для оценки качества модели в CatBoost.
    """
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((math.log1p(max(0, approx[i])) - math.log1p(max(0, target[i])))**2)

        return error_sum, weight_sum


def rmsle(y_true, y_pred):
    """
    Вычисляет Root Mean Squared Logarithmic Error (RMSLE).
    
    Args:
        y_true: Фактические значения
        y_pred: Предсказанные значения
        
    Returns:
        RMSLE метрика
    """
    # Убеждаемся, что все значения неотрицательные
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    
    # Конвертируем в float для поддержки np.log1p (на случай decimal.Decimal)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


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



def train_single_model(df: pd.DataFrame, 
                       target_col: str = 'SumRoubles',
                       exclude_cols: Optional[List[str]] = None,
                       n_splits: int = 5,
                       iterations: int = 1500,
                       depth: int = 8,
                       learning_rate: float = 0.05,
                       random_seed: int = 42) -> Tuple[CatBoostRegressor, dict, List[int], List[str]]:
    """
    Обучает единую модель CatBoost для всех категорий с использованием RMSLE как метрики.
    
    Args:
        df: Исходный датафрейм с признаками
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        exclude_cols: Список колонок для исключения из признаков
        n_splits: Количество фолдов для кросс-валидации
        iterations: Количество итераций обучения
        depth: Максимальная глубина деревьев
        learning_rate: Скорость обучения
        random_seed: Случайное зерно
        
    Returns:
        Кортеж (model, metrics, categorical_features, feature_names):
            - model: Обученная модель CatBoost
            - metrics: Словарь с метриками качества модели
            - categorical_features: Список индексов категориальных признаков
            - feature_names: Список имен признаков
    """
    logger.info("=" * 60)
    logger.info("Обучение единой модели для всех категорий (метрика: RMSLE)")
    logger.info("=" * 60)
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Подготавливаем данные для обучения
    X, y, categorical_features, feature_names = prepare_data_for_training(
        df,
        target_col=target_col,
        exclude_cols=exclude_cols
    )
    
    # Создаем пул данных для CatBoost
    pool = Pool(X, y, cat_features=categorical_features)
    
    # Настраиваем модель CatBoost с пользовательской функцией потерь RMSLE
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function=RMSLE(),
        eval_metric=RMSLE_val(),
        random_seed=random_seed,
        verbose=True
    )
    
    # Кросс-валидация с временным разделением
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    rmsle_scores = []
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    logger.info(f"Проведение кросс-валидации ({n_splits} фолдов)...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Создаем пулы для фолда
        train_pool = Pool(X_train_fold, y_train_fold, cat_features=categorical_features)
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=categorical_features)
        
        # Обучаем модель на фолде с выводом метрики RMSLE
        fold_model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function=RMSLE(),
            eval_metric=RMSLE_val(),
            random_seed=random_seed,
            verbose=False
        )
        fold_model.fit(train_pool, eval_set=val_pool)
        
        # Делаем предсказания
        y_pred = fold_model.predict(val_pool)
        
        # Вычисляем метрики
        fold_rmsle = rmsle(y_val_fold.values, y_pred)
        fold_mae = mean_absolute_error(y_val_fold.values, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold.values, y_pred))
        fold_r2 = r2_score(y_val_fold.values, y_pred)
        
        rmsle_scores.append(fold_rmsle)
        mae_scores.append(fold_mae)
        rmse_scores.append(fold_rmse)
        r2_scores.append(fold_r2)
        
        logger.info(f"Фолд {fold}: RMSLE={fold_rmsle:.4f}, MAE={fold_mae:.4f}, RMSE={fold_rmse:.4f}, R2={fold_r2:.4f}")
    
    # Вычисляем средние значения метрик
    metrics = {
        'rmsle_mean': np.mean(rmsle_scores),
        'rmsle_std': np.std(rmsle_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores)
    }
    
    logger.info("\nСредние метрики по всем фолдам:")
    logger.info(f"  RMSLE: {metrics['rmsle_mean']:.4f} (+/- {metrics['rmsle_std']:.4f})")
    logger.info(f"  MAE: {metrics['mae_mean']:.4f} (+/- {metrics['mae_std']:.4f})")
    logger.info(f"  RMSE: {metrics['rmse_mean']:.4f} (+/- {metrics['rmse_std']:.4f})")
    logger.info(f"  R2: {metrics['r2_mean']:.4f} (+/- {metrics['r2_std']:.4f})")
    
    # Обучаем финальную модель на всех данных
    logger.info("\nОбучение финальной модели на всех данных...")
    model.fit(pool)
    
    logger.info(f"Единая модель успешно обучена на {len(X)} записях")
    logger.info(f"Основная метрика (RMSLE): {metrics['rmsle_mean']:.4f} (+/- {metrics['rmsle_std']:.4f})")
    
    return model, metrics, categorical_features, feature_names






def train_models_per_category(df: pd.DataFrame, 
                               target_col: str = 'SumRoubles',
                               exclude_cols: Optional[List[str]] = None,
                               params: Optional[Dict] = None,
                               n_splits: int = 3,
                               random_seed: int = 42) -> Tuple[Dict, Dict, List[str]]:
    """
    Обучает отдельные модели CatBoost для каждой категории с подбором гиперпараметров.
    Использует RMSLE как метрику для выбора лучших параметров.
    
    Args:
        df: Исходный датафрейм с признаками
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        exclude_cols: Список колонок для исключения из признаков
        params: Словарь с параметрами для Grid Search
        n_splits: Количество фолдов для кросс-валидации
        random_seed: Случайное зерно
        
    Returns:
        Кортеж (models_dict, best_params_dict, feature_names):
            - models_dict: Словарь {category: {'model': model, 'metrics': metrics}}
            - best_params_dict: Словарь {category: best_params}
            - feature_names: Список имен признаков
    """
    logger.info("=" * 60)
    logger.info("Обучение отдельных моделей для каждой категории с подбором гиперпараметров")
    logger.info("Используется метрика: RMSLE")
    logger.info("=" * 60)
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Параметры для подбора по умолчанию
    if params is None:
        params = {
            'l2_leaf_reg': [1, 4, 8],
            'learning_rate': [0.03, 0.1, 0.5],
            'depth': [6, 8, 10]
        }
    
    # Создаем копию датафрейма
    df_clean = df.copy()
    
    # Удаляем строки с NaN в целевой переменной
    df_clean = df_clean.dropna(subset=[target_col])
    
    # Определяем признаки
    cols_to_exclude = [target_col] + exclude_cols
    feature_cols = [col for col in df_clean.columns if col not in cols_to_exclude]
    
    # Категориальные признаки
    categorical_feature_names = [
        'PointID', 'DayOfWeek', 'Quarter', 'Month', 'WeekOfYear',
        'IsFriday', 'IsMonday', 'IsPreHoliday', 'IsPostHoliday', 'isEndOfMonth',
        'BranchID', 'PointClass', 'PointType', 'MicroRegionID', 'LastSalesCategory'
    ]
    
    # Находим индексы категориальных признаков
    categorical_features = []
    for idx, col in enumerate(feature_cols):
        if col in categorical_feature_names:
            categorical_features.append(idx)
    
    # Заполняем NaN в категориальных признаках
    for col in categorical_feature_names:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown').astype(str)
    
    # Получаем список категорий
    categories = df_clean['CategoryID'].unique()
    logger.info(f"Найдено {len(categories)} уникальных категорий")
    
    models_dict = {}
    best_params_dict = {}
    
    # Обучаем модель для каждой категории
    for i, category in enumerate(categories, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Категория {i}/{len(categories)}: {category}")
        logger.info(f"{'='*60}")
        
        # Фильтруем данные по категории
        df_cat = df_clean[df_clean['CategoryID'] == category].copy()
        
        if len(df_cat) < 100:
            logger.warning(f"Недостаточно данных для категории {category} ({len(df_cat)} записей). Пропускаем.")
            continue
        
        X_cat = df_cat[feature_cols].copy()
        y_cat = df_cat[target_col].copy()
        
        # Преобразуем категориальные признаки в строковый тип
        for idx in categorical_features:
            col_name = feature_cols[idx]
            X_cat[col_name] = X_cat[col_name].fillna('Unknown').astype(str)
        
        # Создаем пул данных
        pool_cat = Pool(X_cat, y_cat, cat_features=categorical_features)
        
        # Базовая модель CatBoost с RMSLE
        base_model = CatBoostRegressor(
            iterations=1500,
            loss_function=RMSLE(),
            eval_metric=RMSLE_val(),
            random_seed=random_seed,
            verbose=False
        )
        
        # Grid Search с кросс-валидацией
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info("Проведение Grid Search для подбора гиперпараметров...")
        
        best_score = float('inf')
        best_params = {}
        best_fold_model = None
        
        # Перебор всех комбинаций параметров
        from itertools import product
        param_keys = list(params.keys())
        param_values = list(params.values())
        
        total_combinations = len(param_values[0])
        for key in param_keys[1:]:
            total_combinations *= len(param_values[param_keys.index(key)])
        
        logger.info(f"Всего комбинаций параметров: {total_combinations}")
        
        combination_count = 0
        for param_combo in product(*param_values):
            combination_count += 1
            current_params = dict(zip(param_keys, param_combo))
            
            # Кросс-валидация для данной комбинации параметров
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cat), 1):
                X_train_fold = X_cat.iloc[train_idx]
                y_train_fold = y_cat.iloc[train_idx]
                X_val_fold = X_cat.iloc[val_idx]
                y_val_fold = y_cat.iloc[val_idx]
                
                # Создаем модель с текущими параметрами
                fold_model = CatBoostRegressor(
                    iterations=1500,
                    l2_leaf_reg=current_params['l2_leaf_reg'],
                    learning_rate=current_params['learning_rate'],
                    depth=current_params['depth'],
                    loss_function=RMSLE(),
                    eval_metric=RMSLE_val(),
                    random_seed=random_seed,
                    verbose=False
                )
                
                train_pool = Pool(X_train_fold, y_train_fold, cat_features=categorical_features)
                val_pool = Pool(X_val_fold, y_val_fold, cat_features=categorical_features)
                
                fold_model.fit(train_pool, eval_set=val_pool)
                
                # Предсказания и расчет RMSLE
                y_pred = fold_model.predict(val_pool)
                fold_rmsle = rmsle(y_val_fold.values, y_pred)
                fold_scores.append(fold_rmsle)
            
            mean_score = np.mean(fold_scores)
            
            if mean_score < best_score:
                best_score = mean_score
                best_params = current_params.copy()
                
                # Обучаем лучшую модель на всех данных категории
                best_fold_model = CatBoostRegressor(
                    iterations=1500,
                    l2_leaf_reg=best_params['l2_leaf_reg'],
                    learning_rate=best_params['learning_rate'],
                    depth=best_params['depth'],
                    loss_function=RMSLE(),
                    eval_metric=RMSLE_val(),
                    random_seed=random_seed,
                    verbose=False
                )
            
            if combination_count % 5 == 0 or combination_count == total_combinations:
                logger.info(f"  Комбинация {combination_count}/{total_combinations}: "
                          f"params={current_params}, CV RMSLE={mean_score:.4f}"
                          f"{' *лучший*' if mean_score == best_score else ''}")
        
        logger.info(f"\nЛучшие параметры для категории {category}: {best_params}")
        logger.info(f"Лучший RMSLE (CV): {best_score:.4f}")
        
        # Обучаем финальную модель на всех данных категории с лучшими параметрами
        logger.info("Обучение финальной модели на всех данных категории...")
        final_model = CatBoostRegressor(
            iterations=1500,
            l2_leaf_reg=best_params['l2_leaf_reg'],
            learning_rate=best_params['learning_rate'],
            depth=best_params['depth'],
            loss_function=RMSLE(),
            eval_metric=RMSLE_val(),
            random_seed=random_seed,
            verbose=False
        )
        final_model.fit(pool_cat)
        
        # Сохраняем модель и информацию о ней
        models_dict[category] = {
            'model': final_model,
            'metrics': {'rmsle_cv': best_score},
            'best_params': best_params
        }
        best_params_dict[category] = best_params
        
        logger.info(f"Модель для категории {category} успешно обучена")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Обучено моделей: {len(models_dict)}")
    logger.info("=" * 60)
    
    return models_dict, best_params_dict, feature_cols


def save_single_model(model: CatBoostRegressor, model_path: str = 'catboost_model.cbm') -> None:
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


def save_models_per_category(models_dict: Dict, base_path: str = 'catboost_model_category_{}.cbm') -> Dict[str, str]:
    """
    Сохраняет модели для каждой категории в отдельные файлы.
    
    Args:
        models_dict: Словарь с моделями и информацией о них
        base_path: Шаблон пути для сохранения моделей (с {} для подстановки category)
        
    Returns:
        Словарь {category: путь_к_файлу}
    """
    saved_paths = {}
    for category, model_info in models_dict.items():
        model = model_info['model']
        # Преобразуем category в строку и заменяем недопустимые символы
        category_str = str(category).replace('/', '_').replace('\\', '_')
        path = base_path.format(category_str)
        model.save_model(path)
        saved_paths[category] = path
        logger.info(f"Модель для категории {category} сохранена в файл: {path}")
    
    return saved_paths


def load_models_per_category(categories: List, base_path: str = 'catboost_model_category_{}.cbm') -> Dict:
    """
    Загружает модели для каждой категории из файлов.
    
    Args:
        categories: Список категорий
        base_path: Шаблон пути к файлам моделей
        
    Returns:
        Словарь с загруженными моделями и информацией
    """
    models_dict = {}
    for category in categories:
        category_str = str(category).replace('/', '_').replace('\\', '_')
        path = base_path.format(category_str)
        model = CatBoostRegressor()
        model.load_model(path)
        models_dict[category] = {'model': model}
        logger.info(f"Модель для категории {category} загружена из файла: {path}")
    
    return models_dict


def main():
    """
    Основная функция для обучения моделей CatBoost:
    1. Единая модель для всех категорий (метрика: RMSLE)
    2. Отдельные модели для каждой категории с подбором гиперпараметров (метрика: RMSLE)
    Прогнозирование на тестовых данных с использованием моделей по категориям.
    """
    logger.info("=" * 60)
    logger.info("Запуск обучения моделей CatBoost для прогнозирования SumRoubles")
    logger.info("1. Единая модель для всех категорий")
    logger.info("2. Отдельные модели для каждой категории с подбором гиперпараметров")
    logger.info("=" * 60)
    
    # Установка дат: end_date = текущая дата, start_date = end_date - 1 год
    today = date.today()
    end_date = today
    start_date = end_date - timedelta(days=365)
    
    # Параметры из командной строки
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
    
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'catboost_model_single.cbm'
    
    logger.info(f"Период обучения: {start_date} - {end_date}")
    logger.info(f"Путь сохранения единой модели: {model_path}")
    
    try:
        # Шаг 1: Загрузка данных с признаками
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 1: Загрузка данных с признаками")
        logger.info("=" * 60)
        
        df = load_and_add_features(start_date, end_date)
        
        logger.info(f"Загружено {len(df)} записей")
        logger.info(f"Колонки в датасете: {list(df.columns)}")
        
        # Проверяем наличие категориальных колонок с NaN и заполняем их
        categorical_cols_to_check = ['PointID', 'CategoryID', 'BranchID', 'PointClass', 'PointType', 'MicroRegionID']
        for col in categorical_cols_to_check:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.info(f"Заполнено {nan_count} NaN в колонке {col} значением 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
        
        exclude_cols = ['VisitDate']
        
        # Шаг 2: Обучение единой модели для всех категорий
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 2: Обучение единой модели для всех категорий (RMSLE)")
        logger.info("=" * 60)
        
        # Обучаем единую модель для всех категорий
        model, metrics, categorical_features, feature_names = train_single_model(
            df,
            target_col='SumRoubles',
            exclude_cols=exclude_cols
        )
        
        # Шаг 3: Сохранение единой модели
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Сохранение единой модели")
        logger.info("=" * 60)
        
        model.save_model(model_path)
        logger.info(f"Модель сохранена в файл: {model_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Обучение единой модели завершено успешно!")
        logger.info(f"Основная метрика RMSLE: {metrics['rmsle_mean']:.4f} (+/- {metrics['rmsle_std']:.4f})")
        logger.info("=" * 60)
        
        # Шаг 4: Обучение отдельных моделей для каждой категории
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 4: Обучение отдельных моделей для каждой категории")
        logger.info("=" * 60)
        
        params = {
            'l2_leaf_reg': [1, 4, 8],
            'learning_rate': [0.03, 0.1, 0.5],
            'depth': [6, 8, 10]
        }
        
        models_per_category, best_params_dict, feature_names_cat = train_models_per_category(
            df,
            target_col='SumRoubles',
            exclude_cols=exclude_cols,
            params=params,
            n_splits=3
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Обучение моделей по категориям завершено успешно!")
        logger.info(f"Обучено моделей: {len(models_per_category)}")
        logger.info("=" * 60)
        
        # Вывод лучших параметров для каждой категории
        logger.info("\nЛучшие параметры для категорий:")
        for category, params_best in best_params_dict.items():
            logger.info(f"  Категория {category}: {params_best}")
        
        # Шаг 5: Прогнозирование на тестовых данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 5: Прогнозирование на тестовых данных")
        logger.info("=" * 60)
        
        test_date = today
        logger.info(f"Загрузка тестовых данных на дату: {test_date}")
        
        test_df = fetch_test_data(test_date)
        logger.info(f"Загружено {len(test_df)} строк тестовых данных")
        logger.info(f"Колонки тестового набора: {list(test_df.columns)}")
        
        # Проверяем наличие всех признаков
        missing_cols = set(feature_names) - set(test_df.columns)
        if missing_cols:
            logger.warning(f"Отсутствуют следующие признаки в тестовых данных: {missing_cols}")
            for col in missing_cols:
                test_df[col] = 0
        
        X_test = test_df[feature_names].copy()
        
        # Преобразуем категориальные признаки в строковый тип
        for idx in categorical_features:
            col_name = feature_names[idx]
            X_test[col_name] = X_test[col_name].fillna('Unknown').astype(str)
        
        # Делаем предсказания используя модели по категориям
        predictions = np.zeros(len(X_test))
        
        logger.info("Выполнение прогнозирования с использованием моделей по категориям...")
        
        # Для каждой категории используем свою модель
        for category in models_per_category.keys():
            mask = test_df['CategoryID'] == category
            if mask.sum() > 0:
                X_cat = X_test[mask]
                model_cat = models_per_category[category]['model']
                predictions[mask] = model_cat.predict(X_cat)
                logger.info(f"  Категория {category}: {mask.sum()} предсказаний")
        
        # Для категорий, для которых нет отдельной модели, используем единую модель
        all_categories_with_models = set(models_per_category.keys())
        mask_no_model = ~test_df['CategoryID'].isin(all_categories_with_models)
        if mask_no_model.sum() > 0:
            X_no_model = X_test[mask_no_model]
            predictions[mask_no_model] = model.predict(X_no_model)
            logger.info(f"  Категории без отдельной модели: {mask_no_model.sum()} предсказаний (использована единая модель)")
        
        # Расчет уверенности прогноза
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        confidence = np.exp(-np.abs(predictions - mean_pred) / (std_pred + 1e-6))
        
        # Создаем датафрейм с предсказаниями
        result_df = test_df.copy()
        result_df['predict_new'] = predictions
        result_df['confidence'] = confidence
        
        # Логгируем статистику
        logger.info(f"Выполнено {len(predictions)} предсказаний")
        logger.info(f"  Среднее: {result_df['predict_new'].mean():.4f}")
        logger.info(f"  Std: {result_df['predict_new'].std():.4f}")
        
        # Шаг 6: Сохранение результатов в базу данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 6: Сохранение результатов в SNS_ML_Predictions")
        logger.info("=" * 60)
        
        # Добавляем служебные колонки
        result_df['CreatedAt'] = datetime.now()
        
        # Сохраняем в таблицу
        save_predictions_to_sql(result_df, table_name='SNS_ML_Predictions')
        
        logger.info("\n" + "=" * 60)
        logger.info("Прогнозирование и сохранение результатов завершены успешно!")
        logger.info(f"Результаты сохранены в таблицу: SNS_ML_Predictions")
        logger.info(f"Поле predict_new содержит предсказания от моделей по категориям")
        logger.info("=" * 60)

        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
