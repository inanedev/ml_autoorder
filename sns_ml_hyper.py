import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import os
import json
from dotenv import load_dotenv

# Импорт функций из sns_ml_add_features
from sns_ml_add_features import load_and_add_features

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт CatBoost и Optuna
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost не установлен. Обучение моделей будет недоступно.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna не установлена. Подбор гиперпараметров будет недоступен.")


class BiasRatioMetric(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # Нам нужно, чтобы отношение было 1.0, поэтому максимизация не наш путь
        return False

    def evaluate(self, approxes, target, weight):
        # approxes — это список векторов (по одному на каждое предсказание)
        # Для регрессии берем [0]
        # Так как Tweedie использует log-link, approxes приходят в лог-шкале
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        
        preds = np.exp(approxes[0])
        sum_target = np.sum(target)
        sum_preds = np.sum(preds)
        
        # Считаем отношение. Если сумма предсказаний = 0, возвращаем 0
        ratio = sum_target / sum_preds if sum_preds != 0 else 0
        
        return ratio, 1 # Возвращаем значение и вес (обычно 1)


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


def objective(trial, X_train: pd.DataFrame, y_train: pd.Series, 
              cat_features: List[int], verbose: bool = True) -> float:
    """
    Функция цели для оптимизации Optuna.
    
    Args:
        trial: Объект испытания Optuna
        X_train: Данные для обучения (признаки)
        y_train: Целевая переменная для обучения
        cat_features: Список индексов категориальных признаков
        verbose: Флаг для вывода подробной информации
        
    Returns:
        Значение метрики RMSE для минимизации
    """
    # Предлагаемые гиперпараметры
    variance_power = trial.suggest_categorical('variance_power', [1.2, 1.3, 1.4])
    iterations = trial.suggest_categorical('iterations', [1000, 1500, 3000])
    learning_rate = trial.suggest_categorical('learning_rate', [0.03, 0.05, 0.1])
    depth = trial.suggest_categorical('depth', [4, 6, 8, 10])
    
    # Параметры модели CatBoost с Tweedie loss
    model_params = {
        'iterations': iterations,
        'learning_rate': learning_rate,
        'depth': depth,
        'loss_function': f'Tweedie:variance_power={variance_power}',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0,  # Отключаем вывод для каждого испытания
        'grow_policy': 'Lossguide',
        'l2_leaf_reg': 1,
        'max_leaves': 31
    }
    
    # Создаём пул CatBoost
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    
    # Создаём и обучаем модель
    model = CatBoostRegressor(**model_params)
    model.fit(train_pool)
    
    # Предсказания на обучающих данных для оценки качества
    predictions = model.predict(X_train)
    
    # Вычисляем RMSE
    rmse = np.sqrt(np.mean((predictions - y_train) ** 2))
    
    if verbose:
        logger.info(f"Trial {trial.number}: variance_power={variance_power}, "
                   f"iterations={iterations}, learning_rate={learning_rate}, "
                   f"depth={depth}, RMSE={rmse:.4f}")
    
    return rmse


def optimize_hyperparameters(df: pd.DataFrame,
                             target_col: str = 'SumRoubles',
                             exclude_cols: Optional[List[str]] = None,
                             n_trials: int = 50,
                             timeout: Optional[int] = None,
                             verbose: bool = True) -> Dict:
    """
    Подбирает лучшие гиперпараметры для CatBoost с помощью Optuna.
    
    Args:
        df: Исходный датафрейм с признаками и целевой переменной
        target_col: Название целевой колонки (по умолчанию 'SumRoubles')
        exclude_cols: Список колонок для исключения из признаков
        n_trials: Количество испытаний для оптимизации
        timeout: Таймаут оптимизации в секундах (None - без ограничений)
        verbose: Флаг для вывода подробной информации
        
    Returns:
        Словарь с лучшими гиперпараметрами
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost не установлен. Установите его: pip install catboost")
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna не установлена. Установите её: pip install optuna")
    
    logger.info("=" * 60)
    logger.info("Запуск оптимизации гиперпараметров CatBoost с помощью Optuna")
    logger.info("=" * 60)
    
    # Подготавливаем данные для обучения
    X, y, cat_features, feat_names = prepare_data_for_training(
        df, 
        target_col=target_col,
        exclude_cols=exclude_cols
    )
    
    # Создаем исследование Optuna
    study = optuna.create_study(
        direction='minimize',  # Минимизируем RMSE
        study_name='catboost_hyperparameter_optimization',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Запускаем оптимизацию
    logger.info(f"Запуск {n_trials} испытаний оптимизации...")
    
    study.optimize(
        lambda trial: objective(trial, X, y, cat_features, verbose),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )
    
    # Получаем лучшие параметры
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info("=" * 60)
    logger.info("Оптимизация завершена!")
    logger.info(f"Лучшее значение RMSE: {best_value:.4f}")
    logger.info("Лучшие гиперпараметры:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("=" * 60)
    
    return best_params


def save_best_params(best_params: Dict, save_path: str = 'best_model.json') -> None:
    """
    Сохраняет лучшие гиперпараметры в JSON файл.
    
    Args:
        best_params: Словарь с лучшими гиперпараметрами
        save_path: Путь для сохранения файла с параметрами
    """
    # Добавляем фиксированные параметры
    full_params = {
        'variance_power': best_params.get('variance_power', 1.4),
        'iterations': best_params.get('iterations', 3000),
        'learning_rate': best_params.get('learning_rate', 0.01),
        'depth': best_params.get('depth', 6),
        'random_seed': 42,
        'grow_policy': 'Lossguide',
        'l2_leaf_reg': 1,
        'max_leaves': 31
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(full_params, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Лучшие гиперпараметры сохранены в {save_path}")


def load_best_params(load_path: str = 'best_model.json') -> Optional[Dict]:
    """
    Загружает лучшие гиперпараметры из JSON файла.
    
    Args:
        load_path: Путь к файлу с параметрами
        
    Returns:
        Словарь с гиперпараметрами или None, если файл не найден
    """
    if not os.path.exists(load_path):
        logger.warning(f"Файл с лучшими параметрами {load_path} не найден")
        return None
    
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        logger.info(f"Лучшие гиперпараметры загружены из {load_path}")
        return params
    except Exception as e:
        logger.error(f"Ошибка при загрузке параметров из {load_path}: {e}")
        return None


def run_hyperparameter_optimization(model_save_path: str = 'best_model.json',
                                    n_trials: int = 50,
                                    timeout: Optional[int] = None):
    """
    Запускает полный процесс оптимизации гиперпараметров и сохраняет результат.
    
    Args:
        model_save_path: Путь для сохранения лучших параметров (.json файл)
        n_trials: Количество испытаний для оптимизации
        timeout: Таймаут оптимизации в секундах (None - без ограничений)
    """
    logger.info("=" * 60)
    logger.info("Запуск пайплайна оптимизации гиперпараметров")
    logger.info("=" * 60)
    
    # Шаг 1: Загружаем данные с фичами для обучения
    logger.info("Шаг 1: Загрузка данных для обучения...")
    today = date.today()
    end_date = today
    start_date = end_date - timedelta(days=395)
    
    df_train = load_and_add_features(start_date, end_date)
    logger.info(f"Загружено {len(df_train)} записей для обучения")
    
    # Шаг 2: Оптимизация гиперпараметров
    logger.info("Шаг 2: Оптимизация гиперпараметров CatBoost...")
    best_params = optimize_hyperparameters(
        df_train,
        target_col='SumRoubles',
        exclude_cols=['VisitDate'],
        n_trials=n_trials,
        timeout=timeout,
        verbose=True
    )
    
    # Шаг 3: Сохранение лучших параметров
    logger.info("Шаг 3: Сохранение лучших гиперпараметров...")
    save_best_params(best_params, model_save_path)
    logger.info(f"Лучшие параметры сохранены в {model_save_path}")
    
    logger.info("=" * 60)
    logger.info("Оптимизация гиперпараметров успешно завершена!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_hyperparameter_optimization()
