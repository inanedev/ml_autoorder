import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import os
from dotenv import load_dotenv

# Импорт функций из sns_ml_add_features
from sns_ml_add_features import load_and_add_features

# Импорт функций для работы с тестовыми данными и сохранения предсказаний
from sns_ml_fetch_data import fetch_test_data, save_predictions_to_sql, check_and_cleanup_predictions_table




# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт CatBoost
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost не установлен. Обучение моделей будет недоступно.")

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
                       verbose: bool = True) -> Tuple:
    """
    Обучает единую модель CatBoost на всех категориях сразу с использованием Tweedie loss.
    
    Args:
        df: Исходный датафрейм с признаками и целевой переменной
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        exclude_cols: Список колонок для исключения из признаков
        verbose: Флаг для вывода подробной информации
        
    Returns:
        Кортеж (model, feature_names, categorical_features):
            - model: обученная модель CatBoostRegressor
            - feature_names: список имен признаков
            - categorical_features: список индексов категориальных признаков
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен. Установите его: pip install catboost")
    
    logger.info("Начало обучения единой модели на всех категориях...")
    
    # Подготавливаем данные для обучения
    X, y, cat_features, feat_names = prepare_data_for_training(
        df, 
        target_col=target_col,
        exclude_cols=exclude_cols
    )
    
    # Создаём пулы CatBoost
    train_pool = Pool(X, y, cat_features=cat_features)
    
    # Параметры модели CatBoost с Tweedie loss
    model_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Tweedie:tweedie_variance_power=1.5',
        'eval_metric': 'Tweedie',
        'random_seed': 42,
        'verbose': 100 if verbose else 0
    }
    
    # Создаём и обучаем модель
    logger.info("Обучение модели с функцией потерь Tweedie...")
    model = CatBoostRegressor(**model_params)
    model.fit(train_pool)
    
    logger.info("Модель успешно обучена")
    
    return model, feat_names, cat_features


def predict_with_single_model(model: CatBoostRegressor, 
                               df: pd.DataFrame,
                               feature_names: List[str],
                               categorical_features: List[int],
                               exclude_cols: Optional[List[str]] = None) -> pd.Series:
    """
    Делает предсказания используя единую обученную модель.
    
    Args:
        model: Обученная модель CatBoostRegressor
        df: Датафрейм с данными для предсказания
        feature_names: Список имен признаков, использованных при обучении
        categorical_features: Список индексов категориальных признаков
        exclude_cols: Список колонок для исключения из признаков
        
    Returns:
        Серия с предсказаниями
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен")
    
    logger.info("Выполнение предсказаний с использованием единой модели...")
    
    # Подготавливаем признаки
    X = df[feature_names].copy()
    
    # Преобразуем категориальные признаки в строковый тип
    for idx in categorical_features:
        col_name = feature_names[idx]
        X[col_name] = X[col_name].fillna('Unknown').astype(str)
    
    # Делаем предсказания
    predictions = model.predict(X)
    
    logger.info(f"Предсказания сделаны для {len(predictions)} записей")
    
    return predictions


def save_single_model(model: CatBoostRegressor, feature_names: List[str], 
                      categorical_features: List[int], save_path: str) -> None:
    """
    Сохраняет единую обученную модель в файл.
    
    Args:
        model: Обученная модель CatBoostRegressor
        feature_names: Список имен признаков
        categorical_features: Список индексов категориальных признаков
        save_path: Путь для сохранения модели (.cbm файл)
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен")
    
    # Сохраняем модель
    model.save_model(save_path)
    logger.info(f"Модель сохранена в {save_path}")
    
    # Сохраняем метаданные
    import json
    
    def convert_to_python_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        else:
            return obj
    
    metadata = {
        'feature_names': feature_names,
        'categorical_features': categorical_features
    }
    metadata = convert_to_python_types(metadata)
    
    metadata_path = save_path.replace('.cbm', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Метаданные модели сохранены в {metadata_path}")


def load_single_model(load_path: str) -> Tuple:
    """
    Загружает единую обученную модель из файла.
    
    Args:
        load_path: Путь к файлу модели (.cbm файл)
        
    Returns:
        Кортеж (model, feature_names, categorical_features)
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен")
    
    import json
    
    # Загружаем модель
    model = CatBoostRegressor()
    model.load_model(load_path)
    logger.info(f"Модель загружена из {load_path}")
    
    # Загружаем метаданные
    metadata_path = load_path.replace('.cbm', '_metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return model, metadata['feature_names'], metadata['categorical_features']


def run_full_pipeline(model_save_path: str = 'catboost_model_single.cbm'):
    """
    Запускает полный пайплайн ML: обучение единой модели, сохранение, загрузка тестовых данных,
    предсказание и сохранение результатов в таблицу SNS_ML_Predictions.
    
    Args:
        model_save_path: Путь для сохранения обученной модели (.cbm файл)
    """
    logger.info("=" * 60)
    logger.info("Запуск полного ML пайплайна")
    logger.info("=" * 60)
    
    # Шаг 1: Загружаем данные с фичами для обучения
    logger.info("Шаг 1: Загрузка данных для обучения...")
    today = date.today()
    end_date = today
    start_date = end_date - timedelta(days=395)
    
    df_train = load_and_add_features(start_date, end_date)
    logger.info(f"Загружено {len(df_train)} записей для обучения")
    
    # Шаг 2: Обучаем единую модель на всех категориях
    logger.info("Шаг 2: Обучение единой модели на всех категориях...")
    model, feature_names, categorical_features = train_single_model(
        df_train,
        target_col='SumRoubles',
        exclude_cols=['VisitDate'],
        verbose=True
    )
    logger.info("Модель успешно обучена")
    
    # Шаг 3: Сохраняем модель
    logger.info("Шаг 3: Сохранение обученной модели...")
    save_single_model(model, feature_names, categorical_features, model_save_path)
    logger.info(f"Модель сохранена в {model_save_path}")
    
    # Шаг 4: Загружаем тестовые данные на сегодняшний день
    logger.info("Шаг 4: Загрузка тестовых данных для предсказания...")
    test_df = fetch_test_data(today)
    logger.info(f"Загружено {len(test_df)} записей для предсказания")
    
    # Шаг 5: Проверяем и очищаем таблицу предсказаний от данных за сегодня
    logger.info("Шаг 5: Проверка и очистка таблицы SNS_ML_Predictions...")
    check_and_cleanup_predictions_table(today, table_name='SNS_ML_Predictions')
    
    # Шаг 6: Делаем предсказания с использованием единой модели
    logger.info("Шаг 6: Выполнение предсказаний...")
    predictions = predict_with_single_model(
        model,
        test_df,
        feature_names,
        categorical_features,
        exclude_cols=['VisitDate']
    )
    
    # Шаг 7: Добавляем предсказания в датафрейм
    logger.info("Шаг 7: Подготовка результатов к сохранению...")
    result_df = test_df.copy()
    result_df['predict_new'] = predictions
    result_df['CreatedAt'] = datetime.now()
    
    # Шаг 8: Сохраняем результаты в таблицу SNS_ML_Predictions
    logger.info("Шаг 8: Сохранение предсказаний в таблицу SNS_ML_Predictions...")
    save_predictions_to_sql(result_df, table_name='SNS_ML_Predictions', target_date=today)
    
    logger.info("=" * 60)
    logger.info("ML пайплайн успешно завершён!")
    logger.info(f"Предсказания сохранены в таблицу SNS_ML_Predictions")
    logger.info(f"Количество предсказаний: {len(predictions)}")
    logger.info(f"Среднее значение предсказания: {np.mean(predictions):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
