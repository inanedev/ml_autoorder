"""
Модуль формирования рекомендованного заказа на основе прогноза модели.

Использует реальные хранимые процедуры:
- SNS_ML_Get_Brand_Rules: возвращает ВСЕ бренды (без параметров)
  Колонки: CategoryID, GroupID, BrandQuantum, ImportanceLabel, PriorityWeight, IsTurboBrand, AvgPrice
  
- SNS_ML_Get_Brand_Sales: принимает @StartDate, возвращает продажи за 3 месяца до даты
  Колонки: PointId, VisitDate, BranchID, PointClass, PointType, Lat, Lon, MicroRegionID, 
           CategoryID, brand, brand_name, brand_amount

Логика:
1. Получает прогноз суммы категории от ML-модели.
2. Рассчитывает потребность в штуках для каждого бренда на основе:
   - Средних продаж в конкретный день недели в данной точке.
   - Количества дней до следующего визита.
   - Округления до кванта поставки (BrandQuantum).
3. Распределяет бюджет категории между брендами согласно приоритетам:
   - Turbo (+5), MustList (+4), DriveList (+3), NPI (+2), SPEED KPI (+1).
   - Локальная популярность в микрорегионе (+2).
4. Формирует итоговый список заказов.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging
from datetime import datetime
import math

# Импортируем функцию подключения из существующего модуля
from fetch_raw_data import get_connection

logger = logging.getLogger(__name__)


class OrderRecommender:
    """Класс для формирования рекомендаций заказа на основе прогноза и правил брендов."""
    
    def __init__(self):
        """Инициализация рекомендателя."""
        pass
    
    def get_brand_rules(self, category_id: Optional[int] = None) -> pd.DataFrame:
        """
        Получает правила брендов через хранимую процедуру SNS_ML_Get_Brand_Rules.
        
        Процедура не принимает параметров, возвращает все бренды.
        Если указан category_id, фильтруем результат в Python.
        """
        conn = None
        try:
            conn = get_connection()
            sql = "EXEC SNS_ML_Get_Brand_Rules"
            df = pd.read_sql(sql, conn)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                
                if 'brandquantum' in df.columns:
                    df['brandquantum'] = df['brandquantum'].fillna(1).astype(int)
                if 'priorityweight' in df.columns:
                    df['priorityweight'] = df['priorityweight'].fillna(0).astype(int)
                if 'isturbobrand' in df.columns:
                    df['isturbobrand'] = df['isturbobrand'].apply(
                        lambda x: 1 if str(x).lower() in ['да', 'true', '1', 'yes'] else 0
                    )
                if 'avgprice' in df.columns:
                    df['avgprice'] = df['avgprice'].fillna(0.0).astype(float)
                
                if category_id is not None and 'categoryid' in df.columns:
                    df = df[df['categoryid'] == category_id]
                    
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при получении правил брендов: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def get_brand_sales_history(
        self, 
        end_date: datetime,
        point_id: Optional[int] = None,
        category_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Получает историю продаж брендов через хранимую процедуру SNS_ML_Get_Brand_Sales.
        
        Процедура принимает @StartDate (дата начала периода),
        возвращает продажи за 3 месяца до этой даты.
        """
        conn = None
        try:
            conn = get_connection()
            date_str = end_date.strftime('%Y-%m-%d')
            sql = f"EXEC SNS_ML_Get_Brand_Sales @StartDate = '{date_str}'"
            df = pd.read_sql(sql, conn)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                
                if 'visitdate' in df.columns:
                    df['visitdate'] = pd.to_datetime(df['visitdate'])
                    df['dayofweek'] = df['visitdate'].dt.dayofweek
                
                if point_id is not None and 'pointid' in df.columns:
                    df = df[df['pointid'] == point_id]
                
                if category_id is not None and 'categoryid' in df.columns:
                    df = df[df['categoryid'] == category_id]
                    
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при получении истории продаж: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def _calculate_brand_priority(self, brand_row: pd.Series, is_top_in_microregion: bool = False) -> int:
        """Рассчитывает приоритет бренда по формуле."""
        priority = 1  # Base Regular
        priority += brand_row.get('priorityweight', 0)
        
        if brand_row.get('isturbobrand', 0) == 1:
            priority += 5
            
        if is_top_in_microregion:
            priority += 2
            
        return priority
    
    def _calculate_avg_daily_sales_by_dow(
        self, 
        sales_df: pd.DataFrame, 
        target_dow: int,
        brand_id: Optional[int] = None
    ) -> float:
        """Считает среднее количество продаж в штуках для конкретного дня недели."""
        df = sales_df.copy()
        
        if brand_id is not None:
            df = df[df['brand'] == brand_id]
            
        if df.empty:
            return 0.0
            
        dow_sales = df[df['dayofweek'] == target_dow]
        
        if dow_sales.empty:
            return df['brand_amount'].mean() if 'brand_amount' in df.columns else 0.0
            
        return dow_sales['brand_amount'].mean()
    
    def _get_top_brands_in_microregion(
        self, 
        sales_df: pd.DataFrame, 
        category_id: int,
        target_dow: int,
        top_percentile: float = 0.8
    ) -> Dict[int, bool]:
        """Определяет топовые бренды в микрорегионе для каждого дня недели."""
        if sales_df.empty:
            return {}
        
        filtered = sales_df[
            (sales_df['categoryid'] == category_id) & 
            (sales_df['dayofweek'] == target_dow)
        ]
        
        if filtered.empty:
            return {}
        
        brand_sales = filtered.groupby('brand')['brand_amount'].sum()
        
        if brand_sales.empty or brand_sales.max() == 0:
            return {}
        
        threshold = brand_sales.max() * top_percentile
        top_brands = brand_sales[brand_sales >= threshold].index.tolist()
        
        return {b: True for b in top_brands}
    
    def _round_to_quantum(self, value: float, quantum: int) -> int:
        """Округляет значение вверх до ближайшего кратного кванту."""
        if quantum <= 0:
            quantum = 1
        if value <= 0:
            return 0
        return math.ceil(value / quantum) * quantum
    
    def _distribute_forecast_budget(
        self, 
        candidates: pd.DataFrame, 
        total_budget: float,
        mandatory_priority_threshold: int = 9
    ) -> pd.DataFrame:
        """Жадное распределение бюджета категории по брендам."""
        if candidates.empty:
            return candidates
            
        df = candidates.sort_values('priority', ascending=False).copy()
        df['included'] = False
        df['final_cost'] = df['recommended_qty'] * df['avg_price']
        
        remaining_budget = total_budget
        
        mandatory_mask = df['priority'] >= mandatory_priority_threshold
        if mandatory_mask.any():
            mandatory_cost = df.loc[mandatory_mask, 'final_cost'].sum()
            df.loc[mandatory_mask, 'included'] = True
            remaining_budget -= mandatory_cost
            
            if remaining_budget < 0:
                logger.warning(
                    f"Стоимость обязательных брендов ({mandatory_cost:.2f}) "
                    f"превышает прогноз категории ({total_budget:.2f})"
                )
        
        if remaining_budget > 0:
            optional_df = df[~mandatory_mask].sort_values('priority', ascending=False)
            
            for idx in optional_df.index:
                cost = df.at[idx, 'final_cost']
                if cost <= remaining_budget:
                    df.at[idx, 'included'] = True
                    remaining_budget -= cost
        
        return df
    
    def generate_recommendation(
        self,
        point_id: int,
        category_id: int,
        forecast_amount: float,
        days_until_visit: int,
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Полный цикл формирования рекомендации заказа.
        
        Args:
            point_id: ID точки
            category_id: ID категории
            forecast_amount: Прогноз суммы продаж категории (от ML модели)
            days_until_visit: Дней до следующего визита мерчандайзера
            reference_date: Дата расчета (по умолчанию сегодня)
            
        Returns:
            DataFrame с рекомендованным заказом
        """
        if reference_date is None:
            reference_date = datetime.now()
            
        target_dow = reference_date.weekday()
        
        # 1. Получаем правила брендов для категории
        brand_rules = self.get_brand_rules(category_id)
        if brand_rules.empty:
            logger.warning(f"Нет правил брендов для категории {category_id}")
            return pd.DataFrame()
            
        # 2. Получаем историю продаж точки
        sales_history = self.get_brand_sales_history(
            end_date=reference_date,
            point_id=point_id,
            category_id=category_id
        )
        
        if sales_history.empty:
            logger.info(f"Нет истории продаж для точки {point_id}, категория {category_id}")
        
        # 3. Определяем топовые бренды в микрорегионе
        top_brands_map = self._get_top_brands_in_microregion(
            sales_history, category_id, target_dow
        )
        
        results = []
        
        # 4. Расчет по каждому бренду
        for _, brand in brand_rules.iterrows():
            brand_id = brand.get('groupid')
            if brand_id is None:
                continue
                
            avg_sales = self._calculate_avg_daily_sales_by_dow(
                sales_history, target_dow, brand_id=brand_id
            )
            
            is_top = top_brands_map.get(brand_id, False)
            priority = self._calculate_brand_priority(brand, is_top)
            
            quantum = int(brand.get('brandquantum', 1))
            price = float(brand.get('avgprice', 0.0))
            
            rec_qty = self._round_to_quantum(avg_sales * days_until_visit, quantum)
            est_cost = rec_qty * price
            
            results.append({
                'brand_id': brand_id,
                'brand_name': brand.get('brand_name', f'Brand_{brand_id}'),
                'priority': priority,
                'is_turbo': brand.get('isturbobrand', 0),
                'importance_label': brand.get('importancelabel', ''),
                'is_top_local': is_top,
                'avg_daily_sales': round(avg_sales, 2),
                'days_until_visit': days_until_visit,
                'raw_need': round(avg_sales * days_until_visit, 2),
                'quantum': quantum,
                'recommended_qty': rec_qty,
                'avg_price': price,
                'estimated_cost': round(est_cost, 2)
            })
            
        if not results:
            return pd.DataFrame()
            
        df_results = pd.DataFrame(results)
        final_df = self._distribute_forecast_budget(df_results, forecast_amount)
        
        return final_df[final_df['included']].sort_values('priority', ascending=False)


if __name__ == "__main__":
    print("Запуск теста генерации заказа...")
    
    recommender = OrderRecommender()
    
    TEST_POINT_ID = 12345
    TEST_CATEGORY_ID = 101
    TEST_FORECAST_AMOUNT = 50000.0
    TEST_DAYS_UNTIL_VISIT = 7
    
    try:
        recommendation = recommender.generate_recommendation(
            point_id=TEST_POINT_ID,
            category_id=TEST_CATEGORY_ID,
            forecast_amount=TEST_FORECAST_AMOUNT,
            days_until_visit=TEST_DAYS_UNTIL_VISIT
        )
        
        if not recommendation.empty:
            print(f"\nСформировано {len(recommendation)} позиций заказа:")
            print(recommendation[[
                'brand_id', 'brand_name', 'priority', 'recommended_qty', 
                'avg_price', 'estimated_cost'
            ]].to_string())
            
            total_cost = recommendation['estimated_cost'].sum()
            print(f"\nИтого сумма заказа: {total_cost:.2f} руб.")
            print(f"Прогноз категории: {TEST_FORECAST_AMOUNT:.2f} руб.")
            print(f"Утилизация бюджета: {(total_cost/TEST_FORECAST_AMOUNT)*100:.1f}%")
            
            # Пример сохранения в БД (раскомментировать при необходимости)
            # from save_recommendation import RecommendationStorage
            # from datetime import datetime
            # storage = RecommendationStorage()
            # saved_count = storage.save_recommendation(
            #     recommendation_df=recommendation,
            #     point_id=TEST_POINT_ID,
            #     category_id=TEST_CATEGORY_ID,
            #     forecast_amount=TEST_FORECAST_AMOUNT,
            #     days_until_visit=TEST_DAYS_UNTIL_VISIT,
            #     reference_date=datetime.now(),
            #     model_version="v1.0.0"
            # )
            # print(f"\nСохранено {saved_count} записей в таблицу SNS_ML_Brand_Recommendations")
        else:
            print("Рекомендаций не сформировано.")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
