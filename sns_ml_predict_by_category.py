import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Optional, List
import logging
import os
import json
from dotenv import load_dotenv

# Импорты для CatBoost
from catboost import CatBoostRegressor, Pool

# Импорт функций для работы с тестовыми данными и сохранения предсказаний
from sns_ml_fetch_data import fetch_test_data, save_predictions_to_sql

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CategoryModelPredictor:
    """
    Класс для выполнения предсказаний с использованием моделей, 
    обученных отдельно для каждой категории.
    """
    
    def __init__(self, models_dir: str = 'category_models', 
                 params_file: str = 'category_best_params.json'):
        """
        Инициализация загрузчика моделей по категориям.
        
        Args:
            models_dir: Директория с сохраненными моделями
            params_file: Файл с лучшими гиперпараметрами по категориям
        """
        self.models_dir = models_dir
        self.params_file = params_file
        self.models: Dict[int, CatBoostRegressor] = {}
        self.best_params: Dict[int, dict] = {}
        self.feature_names: Optional[List[str]] = None
        self.categorical_features: Optional[List[int]] = None
        
        logger.info(f"Инициализация CategoryModelPredictor")
        logger.info(f"Директория моделей: {models_dir}")
        logger.info(f"Файл параметров: {params_file}")
    
    def load_params(self) -> bool:
        """
        Загружает лучшие гиперпараметры по категориям из JSON файла.
        
        Returns:
            True если загрузка успешна, иначе False
        """
        if not os.path.exists(self.params_file):
            logger.error(f"Файл с параметрами не найден: {self.params_file}")
            return False
        
        try:
            with open(self.params_file, 'r', encoding='utf-8') as f:
                self.best_params = json.load(f)
            
            # Конвертируем ключи из строк в int
            self.best_params = {int(k): v for k, v in self.best_params.items()}
            
            logger.info(f"Загружены параметры для {len(self.best_params)} категорий")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке параметров: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        Загружает модели для всех категорий из файлов.
        
        Returns:
            True если загрузка успешна, иначе False
        """
        if not os.path.exists(self.models_dir):
            logger.error(f"Директория с моделями не найдена: {self.models_dir}")
            return False
        
        if not self.best_params:
            logger.warning("Параметры не загружены. Сначала вызовите load_params()")
            return False
        
        loaded_count = 0
        for cat_id in self.best_params.keys():
            model_path = os.path.join(self.models_dir, f'catboost_model_cat_{cat_id}.cbm')
            
            if os.path.exists(model_path):
                try:
                    model = CatBoostRegressor()
                    model.load_model(model_path)
                    self.models[cat_id] = model
                    loaded_count += 1
                    logger.debug(f"Загружена модель для категории {cat_id}")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке модели для категории {cat_id}: {e}")
            else:
                logger.warning(f"Модель для категории {cat_id} не найдена: {model_path}")
        
        logger.info(f"Загружено {loaded_count} моделей из {len(self.best_params)}")
        return loaded_count > 0
    
    def set_feature_info(self, feature_names: List[str], categorical_features: List[int]):
        """
        Устанавливает информацию о признаках (имена и индексы категориальных).
        
        Args:
            feature_names: Список имен признаков
            categorical_features: Список индексов категориальных признаков
        """
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        logger.info(f"Установлена информация о признаках: {len(feature_names)} признаков, "
                   f"{len(categorical_features)} категориальных")
    
    def predict(self, test_df: pd.DataFrame, fallback_model: Optional[CatBoostRegressor] = None) -> np.ndarray:
        """
        Выполняет предсказания для тестовых данных, используя модели по категориям.
        
        Args:
            test_df: Датафрейм с тестовыми данными (должен содержать CategoryID и все признаки)
            fallback_model: Резервная модель для категорий без своей модели
            
        Returns:
            Массив предсказаний
        """
        if not self.models:
            logger.error("Модели не загружены. Вызовите load_params() и load_models()")
            raise ValueError("Модели не загружены")
        
        if self.feature_names is None:
            logger.error("Информация о признаках не установлена. Вызовите set_feature_info()")
            raise ValueError("Информация о признаках не установлена")
        
        predictions = np.zeros(len(test_df))
        uncovered_categories = set()
        
        # Проверяем наличие всех признаков
        missing_cols = set(self.feature_names) - set(test_df.columns)
        if missing_cols:
            logger.warning(f"Отсутствуют колонки в тестовых данных: {missing_cols}")
            for col in missing_cols:
                test_df[col] = 0
        
        # Отбираем признаки в правильном порядке
        X_test = test_df[self.feature_names].copy()
        
        # Преобразуем категориальные признаки в строковый тип
        for idx in self.categorical_features:
            col_name = self.feature_names[idx]
            X_test[col_name] = X_test[col_name].fillna('Unknown').astype(str)
        
        # Для каждой категории используем свою модель
        for cat_id, cat_model in self.models.items():
            cat_mask = test_df['CategoryID'] == cat_id
            cat_count = cat_mask.sum()
            
            if cat_count > 0:
                X_test_cat = X_test[cat_mask]
                pred_cat = cat_model.predict(X_test_cat)
                predictions[cat_mask] = pred_cat
        
        # Проверяем, остались ли записи без предсказания
        uncovered_mask = ~test_df['CategoryID'].isin(self.models.keys())
        uncovered_count = uncovered_mask.sum()
        
        if uncovered_count > 0:
            uncovered_categories = test_df.loc[uncovered_mask, 'CategoryID'].unique()
            logger.warning(f"{uncovered_count} записей имеют категории без обученной модели: {uncovered_categories}")
            
            if fallback_model is not None:
                logger.warning("Используем резервную модель для этих записей...")
                predictions[uncovered_mask] = fallback_model.predict(X_test[uncovered_mask])
            else:
                logger.warning("Резервная модель не предоставлена. Заполняем нулями.")
                predictions[uncovered_mask] = 0
        
        return predictions


def main():
    """
    Основная функция для выполнения предсказаний с использованием моделей по категориям.
    """
    logger.info("=" * 60)
    logger.info("Запуск предсказания с использованием моделей по категориям")
    logger.info("=" * 60)
    
    # Загружаем тестовые данные на текущую дату
    today = date.today()
    logger.info(f"Загрузка тестовых данных на дату: {today}")
    
    test_df = fetch_test_data(today)
    logger.info(f"Загружено {len(test_df)} строк тестовых данных")
    
    # Инициализация предсказателя
    predictor = CategoryModelPredictor(
        models_dir='category_models',
        params_file='category_best_params.json'
    )
    
    # Загрузка параметров и моделей
    if not predictor.load_params():
        logger.error("Не удалось загрузить параметры. Прерывание.")
        return
    
    if not predictor.load_models():
        logger.error("Не удалось загрузить модели. Прерывание.")
        return
    
    # Устанавливаем информацию о признаках
    # (эти данные должны быть сохранены при обучении или загружены из конфигурации)
    # Для примера используем стандартный набор признаков
    feature_names = [
        'PointID', 'CategoryID', 'BranchID', 'PointClass', 'PointType',
        'Lat', 'Lon', 'MicroRegionID', 'DayOfWeek', 'Quarter', 'Month',
        'WeekOfYear', 'IsFriday', 'IsMonday', 'IsPreHoliday', 'IsPostHoliday',
        'isEndOfMonth', 'LastSalesCategory'
    ]
    categorical_features = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    predictor.set_feature_info(feature_names, categorical_features)
    
    # Загружаем базовую модель как резервную
    fallback_model = None
    if os.path.exists('catboost_model_base.cbm'):
        fallback_model = CatBoostRegressor()
        fallback_model.load_model('catboost_model_base.cbm')
        logger.info("Загружена резервная базовая модель")
    
    # Выполняем предсказание
    logger.info("Выполнение предсказания...")
    predictions = predictor.predict(test_df, fallback_model=fallback_model)
    
    logger.info(f"Выполнено {len(predictions)} предсказаний")
    logger.info(f"Среднее: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")
    
    # Создаем результирующий датафрейм
    result_df = test_df.copy()
    result_df['predict_advanced'] = predictions
    result_df['CreatedAt'] = datetime.now()
    
    # Сохраняем результаты
    logger.info("Сохранение результатов в SNS_ML_Predictions...")
    save_predictions_to_sql(result_df, table_name='SNS_ML_Predictions')
    
    logger.info("=" * 60)
    logger.info("Предсказание завершено успешно!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
