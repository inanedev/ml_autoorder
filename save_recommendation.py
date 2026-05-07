"""
Модуль сохранения рекомендаций по брендам в SQL Server.

Использует таблицу SNS_ML_Brand_Recommendations для хранения результатов расчета.
"""

import pandas as pd
from typing import Optional, List, Dict
import logging
from datetime import datetime

from fetch_raw_data import get_connection

logger = logging.getLogger(__name__)


class RecommendationStorage:
    """Класс для сохранения и загрузки рекомендаций из SQL Server."""
    
    def __init__(self):
        """Инициализация хранилища."""
        pass
    
    def save_recommendation(
        self,
        recommendation_df: pd.DataFrame,
        point_id: int,
        category_id: int,
        forecast_amount: float,
        days_until_visit: int,
        reference_date: datetime,
        model_version: Optional[str] = None
    ) -> int:
        """
        Сохраняет рекомендации по брендам в таблицу SQL Server.
        
        ИСПОЛЬЗУЕТСЯ BATCH INSERT через executemany для производительности.
        
        Args:
            recommendation_df: DataFrame с результатами расчета из OrderRecommender
            point_id: ID торговой точки
            category_id: ID категории товара
            forecast_amount: Прогноз суммы продаж категории
            days_until_visit: Дней до следующего визита
            reference_date: Дата расчета рекомендации
            model_version: Версия ML модели (опционально)
            
        Returns:
            Количество сохраненных записей
            
        Raises:
            Exception: При ошибке записи в базу данных
        """
        if recommendation_df.empty:
            logger.warning("Пустой DataFrame рекомендаций, ничего не сохраняем")
            return 0
        
        conn = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Включаем режим быстрой вставки для ODBC
            cursor.fast_executemany = True
            
            # Подготовка данных для вставки
            target_dow = reference_date.isoweekday()
            
            insert_query = """
                INSERT INTO dbo.SNS_ML_Brand_Recommendations (
                    PointId, CategoryId, ForecastAmount, DaysUntilVisit,
                    ReferenceDate, TargetDayOfWeek,
                    BrandId, BrandName, BrandQuantum, ImportanceLabel,
                    PriorityWeight, IsTurboBrand, AvgPrice,
                    IsTopLocal, AvgDailySales, RawNeed,
                    Priority, RecommendedQty, EstimatedCost, Included,
                    ModelVersion, PredictedSum, ExtendedSum, Comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Формируем список кортежей для batch insert
            records = []
            for _, row in recommendation_df.iterrows():
                record = (
                    point_id,                                    # PointId
                    category_id,                                 # CategoryId
                    forecast_amount,                             # ForecastAmount
                    days_until_visit,                            # DaysUntilVisit
                    reference_date,                              # ReferenceDate
                    target_dow,                                  # TargetDayOfWeek
                    
                    row['brand_id'],                             # BrandId
                    row['brand_name'],                           # BrandName
                    row['quantum'],                              # BrandQuantum
                    row.get('importance_label', ''),             # ImportanceLabel
                    row.get('priority', 0) - (5 if row.get('is_turbo', 0) else 0) - (2 if row.get('is_top_local', False) else 0),  # PriorityWeight (базовый)
                    row.get('is_turbo', 0),                      # IsTurboBrand
                    row.get('avg_price', 0.0),                   # AvgPrice
                    
                    1 if row.get('is_top_local', False) else 0,  # IsTopLocal
                    row.get('avg_daily_sales', 0.0),             # AvgDailySales
                    row.get('raw_need', 0.0),                    # RawNeed
                    
                    row.get('priority', 0),                      # Priority (итоговый)
                    row.get('recommended_qty', 0),               # RecommendedQty
                    row.get('estimated_cost', 0.0),              # EstimatedCost
                    1 if row.get('included', False) else 0,      # Included
                    
                    model_version,                               # ModelVersion
                    row.get('predicted_sum', forecast_amount),   # PredictedSum
                    row.get('extended_sum', forecast_amount),    # ExtendedSum
                    row.get('comment', '')                       # Comment
                )
                records.append(record)
            
            # Batch insert всех записей одним вызовом
            if records:
                cursor.executemany(insert_query, records)
                conn.commit()
                logger.info(f"Сохранено {len(records)} записей рекомендаций для точки {point_id}, категория {category_id} (batch insert)")
                return len(records)
            else:
                return 0
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Ошибка при сохранении рекомендаций: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_recommendations_batch(
        self,
        recommendations_list: List[Dict],
        model_version: Optional[str] = None
    ) -> int:
        """
        Массовое сохранение рекомендаций для нескольких точек/категорий.
        
        Args:
            recommendations_list: Список словарей с параметрами:
                [
                    {
                        'recommendation_df': DataFrame,
                        'point_id': int,
                        'category_id': int,
                        'forecast_amount': float,
                        'days_until_visit': int,
                        'reference_date': datetime
                    },
                    ...
                ]
            model_version: Версия ML модели (применяется ко всем записям)
            
        Returns:
            Общее количество сохраненных записей
        """
        total_records = 0
        
        for rec_params in recommendations_list:
            try:
                count = self.save_recommendation(
                    recommendation_df=rec_params['recommendation_df'],
                    point_id=rec_params['point_id'],
                    category_id=rec_params['category_id'],
                    forecast_amount=rec_params['forecast_amount'],
                    days_until_visit=rec_params['days_until_visit'],
                    reference_date=rec_params['reference_date'],
                    model_version=model_version
                )
                total_records += count
            except Exception as e:
                logger.error(
                    f"Не удалось сохранить рекомендации для точки "
                    f"{rec_params.get('point_id')}, категория {rec_params.get('category_id')}: {e}"
                )
                continue
        
        return total_records
    
    def get_recommendations_by_point(
        self,
        point_id: int,
        category_id: Optional[int] = None,
        from_date: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Получает сохраненные рекомендации для торговой точки.
        
        Args:
            point_id: ID торговой точки
            category_id: ID категории (опционально)
            from_date: Начальная дата выборки (опционально)
            limit: Максимальное количество записей
            
        Returns:
            DataFrame с рекомендациями
        """
        conn = None
        try:
            conn = get_connection()
            
            query = """
                SELECT TOP (?) *
                FROM dbo.SNS_ML_Brand_Recommendations
                WHERE PointId = ?
            """
            params = [limit, point_id]
            
            if category_id is not None:
                query += " AND CategoryId = ?"
                params.append(category_id)
            
            if from_date is not None:
                query += " AND ReferenceDate >= ?"
                params.append(from_date)
            
            query += " ORDER BY ReferenceDate DESC, CategoryId, Priority DESC"
            
            df = pd.read_sql(query, conn, params=params)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при получении рекомендаций: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def get_latest_recommendations(
        self,
        category_id: Optional[int] = None,
        included_only: bool = True
    ) -> pd.DataFrame:
        """
        Получает последние рекомендации по каждой точке.
        
        Args:
            category_id: Фильтр по категории (опционально)
            included_only: Возвращать только включенные в заказ позиции
            
        Returns:
            DataFrame с последними рекомендациями
        """
        conn = None
        try:
            conn = get_connection()
            
            query = """
                WITH RankedRecs AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY PointId, CategoryId 
                               ORDER BY ReferenceDate DESC
                           ) AS rn
                    FROM dbo.SNS_ML_Brand_Recommendations
                    WHERE 1=1
            """
            params = []
            
            if category_id is not None:
                query += " AND CategoryId = ?"
                params.append(category_id)
            
            if included_only:
                query += " AND Included = 1"
            
            query += """
                )
                SELECT *
                FROM RankedRecs
                WHERE rn = 1
                ORDER BY PointId, CategoryId, Priority DESC
            """
            
            df = pd.read_sql(query, conn, params=params)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при получении последних рекомендаций: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    # Тестовый пример использования
    print("Тест модуля сохранения рекомендаций...")
    
    from order_recommendation import OrderRecommender
    
    recommender = OrderRecommender()
    storage = RecommendationStorage()
    
    TEST_POINT_ID = 12345
    TEST_CATEGORY_ID = 101
    TEST_FORECAST_AMOUNT = 50000.0
    TEST_DAYS_UNTIL_VISIT = 7
    TEST_MODEL_VERSION = "v1.0.0"
    
    try:
        # Генерация рекомендации
        recommendation = recommender.generate_recommendation(
            point_id=TEST_POINT_ID,
            category_id=TEST_CATEGORY_ID,
            forecast_amount=TEST_FORECAST_AMOUNT,
            days_until_visit=TEST_DAYS_UNTIL_VISIT
        )
        
        if not recommendation.empty:
            # Сохранение в БД
            saved_count = storage.save_recommendation(
                recommendation_df=recommendation,
                point_id=TEST_POINT_ID,
                category_id=TEST_CATEGORY_ID,
                forecast_amount=TEST_FORECAST_AMOUNT,
                days_until_visit=TEST_DAYS_UNTIL_VISIT,
                reference_date=datetime.now(),
                model_version=TEST_MODEL_VERSION
            )
            print(f"Сохранено {saved_count} записей")
            
            # Чтение обратно из БД
            retrieved = storage.get_recommendations_by_point(
                point_id=TEST_POINT_ID,
                category_id=TEST_CATEGORY_ID
            )
            print(f"Найдено {len(retrieved)} записей в БД")
            
            if not retrieved.empty:
                print("\nПоследние сохраненные рекомендации:")
                print(retrieved[['brand_id', 'brand_name', 'recommended_qty', 'estimated_cost', 'included']].to_string())
        else:
            print("Рекомендаций не сформировано.")
            
    except Exception as e:
        logger.error(f"Ошибка в тесте: {e}")
        import traceback
        traceback.print_exc()
