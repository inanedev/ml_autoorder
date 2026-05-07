import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import os
from dotenv import load_dotenv

# Импорт функций из sns_ml_add_features
from sns_ml_add_features import load_and_add_features

# Импорт CatBoost
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost не установлен. Обучение моделей будет недоступно.")


# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RMSLE:
    """
    Метрика RMSLE (Root Mean Squared Logarithmic Error) для CatBoost.
    Вычисляет корень из среднего квадратичного логарифмического ошибки.
    """
    
    @staticmethod
    def calc_ders(approx, target, weight):
        """
        Вычисляет градиент и гессиан для RMSLE.
        
        Args:
            approx: Предсказания модели (до применения экспоненты)
            target: Целевые значения
            weight: Веса наблюдений
            
        Returns:
            Кортеж (grad, hess) - градиенты и гессианы
        """
        # Преобразуем предсказания обратно из логарифмического пространства
        pred = np.exp(approx) - 1
        
        # Вычисляем градиент и гессиан
        grad = (pred - target) / (pred + 1)
        hess = 1.0 / (pred + 1)
        
        return grad, hess
    
    @staticmethod
    def get_final_metric(approx, target, weight):
        """
        Вычисляет итоговое значение метрики RMSLE.
        
        Args:
            approx: Предсказания модели (до применения экспоненты)
            target: Целевые значения
            weight: Веса наблюдений
            
        Returns:
            Значение метрики RMSLE
        """
        pred = np.exp(approx) - 1
        log_pred = np.log(pred + 1)
        log_target = np.log(target + 1)
        
        rmsle = np.sqrt(np.mean((log_pred - log_target) ** 2))
        return ('RMSLE', rmsle)


class RMSLE_val:
    """
    Метрика RMSLE для валидации.
    Используется для оценки качества модели на валидационной выборке.
    """
    
    @staticmethod
    def calc_ders(approx, target, weight):
        """
        Вычисляет градиент и гессиан для RMSLE.
        
        Args:
            approx: Предсказания модели (до применения экспоненты)
            target: Целевые значения
            weight: Веса наблюдений
            
        Returns:
            Кортеж (grad, hess) - градиенты и гессианы
        """
        pred = np.exp(approx) - 1
        
        grad = (pred - target) / (pred + 1)
        hess = 1.0 / (pred + 1)
        
        return grad, hess
    
    @staticmethod
    def get_final_metric(approx, target, weight):
        """
        Вычисляет итоговое значение метрики RMSLE.
        
        Args:
            approx: Предсказания модели (до применения экспоненты)
            target: Целевые значения
            weight: Веса наблюдений
            
        Returns:
            Кортеж (имя_метрики, значение)
        """
        pred = np.exp(approx) - 1
        log_pred = np.log(pred + 1)
        log_target = np.log(target + 1)
        
        rmsle = np.sqrt(np.mean((log_pred - log_target) ** 2))
        return ('RMSLE_val', rmsle)


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


def train_models_per_category(df: pd.DataFrame,
                               category_col: str = 'CategoryID',
                               target_col: str = 'SumRoubles',
                               exclude_cols: Optional[List[str]] = None,
                               verbose: bool = True) -> Dict:
    """
    Обучает отдельную модель CatBoost для каждой категории.
    
    Args:
        df: Исходный датафрейм с признаками и целевой переменной
        category_col: Название колонки с категориями (по умолчанию 'CategoryID')
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        exclude_cols: Список колонок для исключения из признаков
        verbose: Флаг для вывода подробной информации
        
    Returns:
        Словарь с обученными моделями и метаданными:
            - models: dict {category_id: trained_model}
            - categories: list of category_ids
            - feature_names: list of feature names used for training
            - categorical_features: list of categorical feature indices
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен. Установите его: pip install catboost")
    
    logger.info("Начало обучения моделей для каждой категории...")
    
    # Проверяем наличие колонки категории
    if category_col not in df.columns:
        raise ValueError(f"Колонка '{category_col}' не найдена в датафрейме")
    
    # Получаем уникальные категории
    categories = df[category_col].unique()
    logger.info(f"Найдено {len(categories)} уникальных категорий")
    
    models = {}
    feature_names = None
    categorical_features = None
    
    for idx, category in enumerate(categories):
        logger.info(f"Обучение модели для категории {category} ({idx + 1}/{len(categories)})...")
        
        # Фильтруем данные по категории
        df_category = df[df[category_col] == category].copy()
        
        if len(df_category) < 10:
            logger.warning(f"Пропуск категории {category}: недостаточно данных ({len(df_category)} записей)")
            continue
        
        # Подготавливаем данные для обучения
        X, y, cat_features, feat_names = prepare_data_for_training(
            df_category, 
            target_col=target_col,
            exclude_cols=exclude_cols
        )
        
        # Сохраняем имена признаков и индексы категориальных признаков (они одинаковы для всех категорий)
        if feature_names is None:
            feature_names = feat_names
            categorical_features = cat_features
        
        # Создаём пулы CatBoost
        train_pool = Pool(X, y, cat_features=cat_features)
        
        # Параметры модели CatBoost
        model_params = {
            'iterations': 1000,
            'loss_function': RMSLE(),
            'eval_metric': RMSLE_val(),
            'l2_leaf_reg': 1,
            'learning_rate': 0.03,
            'depth': 6,
            'verbose': verbose,
            'random_seed': 42
        }
        
        # Создаём и обучаем модель
        model = CatBoostRegressor(**model_params)
        model.fit(train_pool)
        
        models[category] = model
        logger.info(f"Модель для категории {category} обучена успешно")
    
    logger.info(f"Обучено {len(models)} моделей")
    
    return {
        'models': models,
        'categories': list(models.keys()),
        'feature_names': feature_names,
        'categorical_features': categorical_features
    }


def predict_with_category_models(models_dict: Dict, 
                                  df: pd.DataFrame,
                                  category_col: str = 'CategoryID',
                                  exclude_cols: Optional[List[str]] = None) -> pd.Series:
    """
    Делает предсказания используя обученные модели для каждой категории.
    
    Args:
        models_dict: Словарь с результатами train_models_per_category()
        df: Датафрейм с данными для предсказания
        category_col: Название колонки с категориями
        exclude_cols: Список колонок для исключения из признаков
        
    Returns:
        Серия с предсказаниями
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен")
    
    models = models_dict['models']
    feature_names = models_dict['feature_names']
    categorical_features = models_dict['categorical_features']
    
    logger.info("Выполнение предсказаний для всех категорий...")
    
    # Создаём серию для предсказаний
    predictions = pd.Series(index=df.index, dtype=float)
    
    # Для каждой категории делаем предсказания своей моделью
    for category, model in models.items():
        # Фильтруем данные по категории
        mask = df[category_col] == category
        df_category = df[mask].copy()
        
        if len(df_category) == 0:
            continue
        
        # Подготавливаем признаки
        X_category = df_category[feature_names].copy()
        
        # Преобразуем категориальные признаки в строковый тип
        for idx in categorical_features:
            col_name = feature_names[idx]
            X_category[col_name] = X_category[col_name].fillna('Unknown').astype(str)
        
        # Делаем предсказания
        pred_values = model.predict(X_category)
        predictions.loc[mask] = pred_values
    
    logger.info(f"Предсказания сделаны для {len(predictions)} записей")
    
    return predictions


def save_models_per_category(models_dict: Dict, save_dir: str) -> None:
    """
    Сохраняет обученные модели для каждой категории в отдельные файлы.
    
    Args:
        models_dict: Словарь с результатами train_models_per_category()
        save_dir: Директория для сохранения моделей
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен")
    
    os.makedirs(save_dir, exist_ok=True)
    
    models = models_dict['models']
    
    # Сохраняем каждую модель
    for category, model in models.items():
        model_path = os.path.join(save_dir, f"model_category_{category}.cbm")
        model.save_model(model_path)
        logger.info(f"Модель для категории {category} сохранена в {model_path}")
    
    # Сохраняем метаданные
    metadata = {
        'categories': models_dict['categories'],
        'feature_names': models_dict['feature_names'],
        'categorical_features': models_dict['categorical_features']
    }
    metadata_path = os.path.join(save_dir, "model_metadata.json")
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Метаданные моделей сохранены в {metadata_path}")


def load_models_per_category(load_dir: str) -> Dict:
    """
    Загружает обученные модели для каждой категории из файлов.
    
    Args:
        load_dir: Директория с сохранёнными моделями
        
    Returns:
        Словарь с загруженными моделями и метаданными
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен")
    
    import json
    
    # Загружаем метаданные
    metadata_path = os.path.join(load_dir, "model_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    models = {}
    for category in metadata['categories']:
        model_path = os.path.join(load_dir, f"model_category_{category}.cbm")
        model = CatBoostRegressor()
        model.load_model(model_path)
        models[category] = model
        logger.info(f"Модель для категории {category} загружена из {model_path}")
    
    return {
        'models': models,
        'categories': metadata['categories'],
        'feature_names': metadata['feature_names'],
        'categorical_features': metadata['categorical_features']
    }
