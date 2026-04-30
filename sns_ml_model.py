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
    # Использование: python sns_ml_model.py [start_date] [end_date] [model_path] [--predict]
    predict_mode = '--predict' in sys.argv or '-p' in sys.argv
    
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
    
    model_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ['--predict', '-p'] else 'catboost_model.cbm'
    
    logger.info(f"Период обучения: {start_date} - {end_date}")
    logger.info(f"Путь сохранения модели: {model_path}")
    logger.info(f"Режим прогнозирования: {predict_mode}")
    
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

        # Шаг 6: Вывод итоговой информации по обеим моделям
        logger.info("\n" + "=" * 60)
        logger.info("Итоговые метрики базовой модели (MAE)")
        logger.info("=" * 60)
        logger.info(f"MAE: {metrics_base['mae_mean']:.4f} (+/- {metrics_base['mae_std']:.4f})")
        logger.info(f"RMSE: {metrics_base['rmse_mean']:.4f} (+/- {metrics_base['rmse_std']:.4f})")
        logger.info(f"R²: {metrics_base['r2_mean']:.4f} (+/- {metrics_base['r2_std']:.4f})")

        logger.info("\n" + "=" * 60)
        logger.info("Итоговые метрики улучшенной модели (RMSE)")
        logger.info("=" * 60)
        logger.info(f"MAE: {metrics_new['mae_mean']:.4f} (+/- {metrics_new['mae_std']:.4f})")
        logger.info(f"RMSE: {metrics_new['rmse_mean']:.4f} (+/- {metrics_new['rmse_std']:.4f})")
        logger.info(f"R²: {metrics_new['r2_mean']:.4f} (+/- {metrics_new['r2_std']:.4f})")

        # Важность признаков для базовой модели
        logger.info("\n" + "=" * 60)
        logger.info("Важность признаков - Базовая модель (топ-15)")
        logger.info("=" * 60)

        feature_importance_base = model_base.get_feature_importance()
        importance_df_base = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance_base
        }).sort_values('Importance', ascending=False)

        print(importance_df_base.head(15).to_string(index=False))

        # Важность признаков для улучшенной модели
        logger.info("\n" + "=" * 60)
        logger.info("Важность признаков - Улучшенная модель (топ-15)")
        logger.info("=" * 60)

        feature_importance_new = model_new.get_feature_importance()
        importance_df_new = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance_new
        }).sort_values('Importance', ascending=False)

        print(importance_df_new.head(15).to_string(index=False))

        logger.info("\n" + "=" * 60)
        logger.info("Обучение обеих моделей завершено успешно!")
        logger.info("=" * 60)

        # Шаг 7: Прогнозирование на тестовых данных (если указан флаг --predict)
        if predict_mode:
            logger.info("\n" + "=" * 60)
            logger.info("Шаг 7: Прогнозирование на тестовых данных обеими моделями")
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

            # Прогнозирование базовой моделью
            logger.info("Выполнение прогнозирования базовой моделью...")
            predictions_base = model_base.predict(X_test)

            # Прогнозирование улучшенной моделью
            logger.info("Выполнение прогнозирования улучшенной моделью...")
            predictions_new = model_new.predict(X_test)

            # Расчет вероятности/уверенности прогноза для улучшенной модели
            mean_pred_new = np.mean(predictions_new)
            std_pred_new = np.std(predictions_new)
            confidence_new = np.exp(-np.abs(predictions_new - mean_pred_new) / (std_pred_new + 1e-6))

            # Добавляем предсказания от обеих моделей в тестовый датафрейм
            test_df['Predict_base'] = predictions_base
            test_df['Predict_new'] = predictions_new
            test_df['Prediction_Confidence'] = confidence_new

            # Добавляем служебные поля
            test_df['CreatedAt'] = datetime.now()
            test_df['ModelVersion'] = 'catboost_v2_dual_model'

            logger.info(f"Прогнозы выполнены. Статистика предсказаний:")
            logger.info(f"  Базовая модель - Среднее: {np.mean(predictions_base):.2f}, Стандартное отклонение: {np.std(predictions_base):.2f}")
            logger.info(f"  Улучшенная модель - Среднее: {np.mean(predictions_new):.2f}, Стандартное отклонение: {np.std(predictions_new):.2f}")
            logger.info(f"  Средняя уверенность: {np.mean(confidence_new):.4f}")

            # Сохранение результатов в базу данных
            logger.info("\n" + "=" * 60)
            logger.info("Шаг 8: Сохранение результатов в SNS_ML_Predictions")
            logger.info("=" * 60)

            # Выбираем все колонки для сохранения
            columns_to_save = list(test_df.columns)
            logger.info(f"Сохранение {len(test_df)} записей с {len(columns_to_save)} колонками")

            save_predictions_to_sql(test_df, table_name='SNS_ML_Predictions')

            logger.info("\n" + "=" * 60)
            logger.info("Прогнозирование и сохранение результатов завершены успешно!")
            logger.info("=" * 60)

        
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
