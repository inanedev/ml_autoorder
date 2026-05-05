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
    
    # Обучаем модель с использованием RMSE как функции потерь
    # RMSLE используется как метрика оценки в кросс-валидации
    model, metrics = train_catboost_model(
        X=X,
        y=y,
        categorical_features=categorical_features,
        n_splits=n_splits,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
        loss_function='RMSE',
        model_name="Единая модель (RMSLE)"
    )
    
    logger.info(f"Единая модель успешно обучена на {len(X)} записях")
    logger.info(f"Основная метрика (RMSLE): {metrics['rmsle_mean']:.4f} (+/- {metrics['rmsle_std']:.4f})")
    
    return model, metrics, categorical_features, feature_names






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
    Основная функция для обучения единой модели CatBoost для всех категорий
    с использованием метрики RMSLE и прогнозирования на тестовых данных.
    """
    logger.info("=" * 60)
    logger.info("Запуск обучения единой модели CatBoost для прогнозирования SumRoubles")
    logger.info("Используется единая модель для всех категорий (метрика: RMSLE)")
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
    logger.info(f"Путь сохранения модели: {model_path}")
    
    try:
        # Шаг 1: Загрузка данных с признаками
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 1: Загрузка данных с признаками")
        logger.info("=" * 60)
        
        df = load_and_add_features(start_date, end_date)
        
        logger.info(f"Загружено {len(df)} записей")
        logger.info(f"Колонки в датасете: {list(df.columns)}")
        
        # Шаг 2: Обучение единой модели для всех категорий
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 2: Обучение единой модели для всех категорий (RMSLE)")
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
        
        # Обучаем единую модель для всех категорий
        model, metrics, categorical_features, feature_names = train_single_model(
            df,
            target_col='SumRoubles',
            exclude_cols=exclude_cols
        )
        
        # Шаг 3: Сохранение модели
        logger.info("\n" + "=" * 60)
        logger.info("Шаг 3: Сохранение единой модели")
        logger.info("=" * 60)
        
        model.save_model(model_path)
        logger.info(f"Модель сохранена в файл: {model_path}")
        
        # Шаг 4: Вывод итоговой информации
        logger.info("\n" + "=" * 60)
        logger.info("Обучение модели завершено успешно!")
        logger.info(f"Основная метрика RMSLE: {metrics['rmsle_mean']:.4f} (+/- {metrics['rmsle_std']:.4f})")
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
        
        # Делаем предсказание
        predictions = model.predict(X_test)
        
        # Расчет уверенности прогноза
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        confidence = np.exp(-np.abs(predictions - mean_pred) / (std_pred + 1e-6))
        
        # Создаем датафрейм с предсказаниями
        result_df = test_df.copy()
        result_df['predict'] = predictions
        result_df['confidence'] = confidence
        
        # Логгируем статистику
        logger.info(f"Выполнено {len(predictions)} предсказаний")
        logger.info(f"  Среднее: {result_df['predict'].mean():.4f}")
        logger.info(f"  Std: {result_df['predict'].std():.4f}")
        
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
