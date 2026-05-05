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


def tune_huber_alpha_fast(X: pd.DataFrame, 
                     y: pd.Series, 
                     categorical_features: List[int],
                     n_iter: int = 10,
                     cv_splits: int = 2,
                     random_seed: int = 42,
                     base_iterations: int = 1500,
                     base_depth: int = 10,
                     base_learning_rate: float = 0.05) -> Tuple[float, dict]:
    """
    Быстрый подбор оптимального коэффициента alpha (delta) для функции потерь Huber 
    с помощью RandomizedSearchCV. Оптимизировано для скорости.
    
    Args:
        X: Датафрейм с признаками
        y: Серия с целевой переменной
        categorical_features: Список индексов категориальных признаков
        n_iter: Количество итераций поиска (уменьшено для скорости)
        cv_splits: Количество фолдов для кросс-валидации (уменьшено для скорости)
        random_seed: Случайное зерно
        base_iterations: Базовое количество итераций обучения
        base_depth: Базовая глубина деревьев
        base_learning_rate: Базовая скорость обучения
        
    Returns:
        Кортеж (best_alpha, search_results):
            - best_alpha: Лучшее значение delta для Huber
            - search_results: Словарь с результатами поиска
    """
    logger.info(f"Быстрый подбор delta для Huber (итераций={n_iter}, фолдов={cv_splits})...")
    
    # Базовые параметры модели (без early_stopping для скорости на малых данных)
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
    
    best_score = float('inf')
    best_alpha = None
    all_results = []
    
    np.random.seed(random_seed)
    
    for iter_idx in range(n_iter):
        # Случайный выбор delta из диапазона [0.5, 3.0]
        delta = np.random.uniform(0.5, 3.0)
        loss_function = f'Huber:delta={delta:.3f}'
        
        # Кросс-валидация для текущих параметров
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < 100 or len(val_idx) < 50:
                continue  # Пропускаем слишком маленькие фолды
                
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
        
        if not cv_scores:
            continue
            
        avg_score = np.mean(cv_scores)
        
        all_results.append({
            'delta': delta,
            'loss_function': loss_function,
            'mae_mean': avg_score
        })
        
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = delta
    
    # Если не удалось подобрать, используем дефолтное значение
    if best_alpha is None:
        best_alpha = 1.345
        logger.warning(f"Не удалось подобрать delta, используем дефолтное значение: {best_alpha}")
    
    return best_alpha, {'best_delta': best_alpha, 'best_mae': best_score, 'all_results': all_results}


def train_models_per_category(df: pd.DataFrame, 
                               target_col: str = 'SumRoubles',
                               exclude_cols: Optional[List[str]] = None,
                               category_col: str = 'CategoryID') -> Dict:
    """
    Обучает отдельную модель для каждой товарной категории с индивидуальным подбором delta для Huber.
    
    Args:
        df: Исходный датафрейм с признаками
        target_col: Название целевой колонки
        exclude_cols: Список колонок для исключения из признаков
        category_col: Название колонки с категорией товара
        
    Returns:
        Словарь с моделями, параметрами и метриками для каждой категории
    """
    logger.info("=" * 60)
    logger.info("Обучение моделей для каждой товарной категории")
    logger.info("=" * 60)
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Получаем уникальные категории
    categories = df[category_col].unique()
    logger.info(f"Найдено {len(categories)} уникальных категорий")
    
    models_dict = {}
    results_summary = {}
    
    for idx, category in enumerate(categories):
        logger.info(f"\n{'='*60}")
        logger.info(f"Категория {idx+1}/{len(categories)}: {category}")
        logger.info(f"{'='*60}")
        
        # Фильтруем данные по категории
        df_category = df[df[category_col] == category].copy()
        
        # Подготавливаем данные для этой категории
        X_cat, y_cat, categorical_features, feature_names = prepare_data_for_training(
            df_category,
            target_col=target_col,
            exclude_cols=exclude_cols + [category_col] if category_col not in exclude_cols else exclude_cols
        )
        
        # Определяем тип модели в зависимости от количества данных
        is_small_category = len(X_cat) < 200
        
        if is_small_category:
            logger.info(f"Категория {category}: мало данных ({len(X_cat)} записей), используем упрощенную модель с MAE")
            # Для малых категорий используем упрощенную модель без подбора параметров
            base_params = {
                'iterations': 700,
                'depth': 8,
                'learning_rate': 0.03,
                'loss_function': 'MAE',
                'eval_metric': 'MAE',
                'verbose': False,
                'cat_features': categorical_features if categorical_features else None,
                'random_seed': 42,
                'early_stopping_rounds': 50
            }
            best_alpha = None
            search_results = {'best_mae': None}
        else:
            # Быстрый подбор delta для категорий с достаточным количеством данных
            best_alpha, search_results = tune_huber_alpha_fast(
                X_cat, y_cat,
                categorical_features=categorical_features,
                n_iter=10,  # Минимум итераций для скорости
                cv_splits=2,  # Минимум фолдов для скорости
                random_seed=42,
                base_iterations=1500,
                base_depth=10,
                base_learning_rate=0.05
            )
            
            logger.info(f"Лучшее delta для категории {category}: {best_alpha:.3f} (MAE: {search_results['best_mae']:.4f})")
            
            # Финальное обучение модели с лучшим delta на всех данных категории
            optimal_loss_function = f'Huber:delta={best_alpha:.3f}'
            
            base_params = {
                'iterations': 1500,
                'depth': 10,
                'learning_rate': 0.05,
                'loss_function': optimal_loss_function,
                'eval_metric': 'MAE',
                'verbose': False,
                'cat_features': categorical_features if categorical_features else None,
                'random_seed': 42,
                'early_stopping_rounds': 50
            }
        
        train_pool = Pool(X_cat, y_cat, cat_features=categorical_features if categorical_features else None)
        model_category = CatBoostRegressor(**base_params)
        model_category.fit(train_pool, verbose=False)
        
        # Сохраняем модель и информацию о ней
        models_dict[category] = {
            'model': model_category,
            'best_delta': best_alpha,
            'best_mae': search_results['best_mae'],
            'categorical_features': categorical_features,
            'feature_names': feature_names
        }
        
        results_summary[category] = {
            'best_delta': best_alpha,
            'best_mae': search_results['best_mae'],
            'n_samples': len(X_cat)
        }
        
        logger.info(f"Модель для категории {category} обучена успешно")
    
    # Вывод сводной информации
    logger.info("\n" + "=" * 60)
    logger.info("Сводная информация по всем категориям:")
    logger.info("=" * 60)
    for category, info in sorted(results_summary.items(), key=lambda x: x[1]['best_mae']):
        logger.info(f"  Категория {category}: delta={info['best_delta']:.3f}, MAE={info['best_mae']:.4f}, записей={info['n_samples']}")
    
    return models_dict




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
    Основная функция для обучения моделей CatBoost (по одной на категорию) 
    и прогнозирования на тестовых данных.
    """
    logger.info("=" * 60)
    logger.info("Запуск обучения моделей CatBoost для прогнозирования SumRoubles")
    logger.info("Обучение отдельной модели для каждой товарной категории")
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
    
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'catboost_model_category_{}.cbm'
    
    logger.info(f"Период обучения: {start_date} - {end_date}")
    logger.info(f"Шаблон пути сохранения моделей: {model_path}")
    
    try:
        # Шаг 1: Загрузка данных с признаками
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 1: Загрузка данных с признаками")
        logger.info("=" * 60)
        
        df = load_and_add_features(start_date, end_date)
        
        logger.info(f"Загружено {len(df)} записей")
        logger.info(f"Колонки в датасете: {list(df.columns)}")
        
        # Шаг 2: Обучение моделей для каждой категории
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 2: Обучение моделей для каждой товарной категории")
        logger.info("Для каждой категории подбирается оптимальный delta для Huber")
        logger.info("Фиксированные параметры: Iterations=1500, Depth=10, LearningRate=0.05")
        logger.info("=" * 60)
        
        exclude_cols = ['VisitDate']
        
        # Проверяем наличие категориальных колонок с NaN и заполняем их
        categorical_cols_to_check = ['PointID', 'CategoryID', 'BranchID', 'PointClass', 'PointType', 'MicroRegionID']
        for col in categorical_cols_to_check:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.info(f"Заполнено {nan_count} NaN в колонке {col} значением 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
        
        # Обучаем модели для каждой категории
        models_dict = train_models_per_category(
            df,
            target_col='SumRoubles',
            exclude_cols=exclude_cols,
            category_col='CategoryID'
        )
        
        if not models_dict:
            logger.error("Не удалось обучить ни одну модель!")
            sys.exit(1)
        
        # Шаг 3: Сохранение моделей
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Сохранение моделей для каждой категории")
        logger.info("=" * 60)
        
        saved_paths = save_models_per_category(models_dict, model_path)
        
        # Шаг 4: Вывод итоговой информации
        logger.info("\n" + "=" * 60)
        logger.info("Обучение моделей завершено успешно!")
        logger.info(f"Обучено {len(models_dict)} моделей для разных категорий")
        logger.info("=" * 60)
        
        # Шаг 5: Прогнозирование на тестовых данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 4: Прогнозирование на тестовых данных")
        logger.info("=" * 60)
        
        test_date = today
        logger.info(f"Загрузка тестовых данных на дату: {test_date}")
        
        test_df = fetch_test_data(test_date)
        logger.info(f"Загружено {len(test_df)} строк тестовых данных")
        logger.info(f"Колонки тестового набора: {list(test_df.columns)}")
        
        # Проверяем наличие CategoryID в тестовых данных
        if 'CategoryID' not in test_df.columns:
            logger.error("Отсутствует колонка CategoryID в тестовых данных!")
            sys.exit(1)
        
        # Получаем уникальные категории из тестовых данных
        test_categories = test_df['CategoryID'].unique()
        logger.info(f"Уникальные категории в тестовых данных: {len(test_categories)}")
        
        # Для каждой категории делаем предсказание своей моделью
        all_predictions = []
        all_confidences = []
        
        for category in test_categories:
            if category not in models_dict:
                logger.warning(f"Нет модели для категории {category}, пропускаем...")
                continue
            
            model_info = models_dict[category]
            model = model_info['model']
            feature_names = model_info['feature_names']
            categorical_features = model_info['categorical_features']
            
            # Фильтруем тестовые данные по категории
            test_cat_mask = test_df['CategoryID'] == category
            test_cat_df = test_df[test_cat_mask].copy()
            
            if len(test_cat_df) == 0:
                continue
            
            # Проверяем наличие всех признаков
            missing_cols = set(feature_names) - set(test_cat_df.columns)
            if missing_cols:
                for col in missing_cols:
                    test_cat_df[col] = 0
            
            X_test_cat = test_cat_df[feature_names].copy()
            
            # Преобразуем категориальные признаки в строковый тип
            for idx in categorical_features:
                col_name = feature_names[idx]
                X_test_cat[col_name] = X_test_cat[col_name].fillna('Unknown').astype(str)
            
            # Делаем предсказание
            predictions_cat = model.predict(X_test_cat)
            
            # Расчет уверенности прогноза для этой категории
            mean_pred_cat = np.mean(predictions_cat)
            std_pred_cat = np.std(predictions_cat)
            confidence_cat = np.exp(-np.abs(predictions_cat - mean_pred_cat) / (std_pred_cat + 1e-6))
            
            # Сохраняем предсказания и индексы
            all_predictions.extend(zip(test_cat_df.index, predictions_cat, confidence_cat, [category] * len(predictions_cat)))
        
        # Создаем датафрейм с предсказаниями
        result_df = test_df.copy()
        result_df['predict'] = np.nan
        result_df['confidence'] = np.nan
        
        for idx, pred, conf, cat in all_predictions:
            result_df.loc[idx, 'predict'] = pred
            result_df.loc[idx, 'confidence'] = conf
        
        # Логгируем статистику
        valid_predictions = result_df['predict'].notna()
        if valid_predictions.sum() > 0:
            logger.info(f"Выполнено {valid_predictions.sum()} предсказаний")
            logger.info(f"  Среднее: {result_df.loc[valid_predictions, 'predict'].mean():.4f}")
            logger.info(f"  Std: {result_df.loc[valid_predictions, 'predict'].std():.4f}")
        else:
            logger.warning("Не выполнено ни одного предсказания!")
        
        # Шаг 6: Сохранение результатов в базу данных
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 5: Сохранение результатов в SNS_ML_Predictions")
        logger.info("=" * 60)
        
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
