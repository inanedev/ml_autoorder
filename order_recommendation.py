"""
Модуль формирования рекомендованного заказа на основе прогноза модели.

Оптимизированная версия для пакетной обработки всех точек и категорий за один раз.

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

ОПТИМИЗАЦИИ:
- Единовременная загрузка всех данных из БД
- Векторизованные вычисления через pandas merge/groupby
- Пакетная обработка всех пар точка-категория
- Минимизация циклов Python
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import math

# Импортируем функцию подключения из существующего модуля
from fetch_raw_data import get_connection

logger = logging.getLogger(__name__)


class OrderRecommender:
    """Класс для формирования рекомендаций заказа на основе прогноза и правил брендов."""
    
    def __init__(self):
        """Инициализация рекомендателя с кэшированием данных."""
        self._brand_rules_cache: Optional[pd.DataFrame] = None
        self._sales_history_cache: Optional[pd.DataFrame] = None
        self._reference_date: Optional[datetime] = None
    
    def _prepare_brand_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит типы данных и подготавливает DataFrame правил брендов."""
        if df.empty:
            return df
            
        # Нормализуем имена колонок в нижний регистр
        df.columns = [col.lower() for col in df.columns]
        
        # Явно переименовываем возможные вариации имен колонок
        rename_map = {}
        for col in df.columns:
            if col in ['categoryid', 'category_id', 'category']:
                rename_map[col] = 'categoryid'
            elif col in ['brandid', 'brand_id', 'brand']:
                rename_map[col] = 'brand'
            elif col in ['brandquantum', 'brand_quantum', 'quantum']:
                rename_map[col] = 'brandquantum'
            elif col in ['priorityweight', 'priority_weight', 'weight']:
                rename_map[col] = 'priorityweight'
            elif col in ['isturbobrand', 'is_turbo_brand', 'turbo']:
                rename_map[col] = 'isturbobrand'
            elif col in ['avgprice', 'avg_price', 'price']:
                rename_map[col] = 'avgprice'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
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
            
        return df
    
    def get_all_brand_rules(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Получает ВСЕ правила брендов через хранимую процедуру SNS_ML_Get_Brand_Rules.
        
        Процедура не принимает параметров, возвращает все бренды.
        Данные кэшируются после первого запроса.
        
        Args:
            force_refresh: Принудительное обновление кэша
            
        Returns:
            DataFrame со всеми правилами брендов
        """
        if self._brand_rules_cache is not None and not force_refresh:
            return self._brand_rules_cache.copy()
        
        conn = None
        try:
            logger.info("Загрузка всех правил брендов из БД...")
            conn = get_connection()
            sql = "EXEC SNS_ML_Get_Brand_Rules"
            df = pd.read_sql(sql, conn)
            
            if not df.empty:
                df = self._prepare_brand_rules(df)
                self._brand_rules_cache = df.copy()
                logger.info(f"Загружено {len(df)} правил брендов для {df['categoryid'].nunique() if 'categoryid' in df.columns else 0} категорий")
            else:
                logger.warning("Правила брендов пусты")
                
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при получении правил брендов: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def get_all_sales_history(
        self, 
        end_date: datetime,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Получает ВСЮ историю продаж через хранимую процедуру SNS_ML_Get_Brand_Sales.
        
        Процедура принимает @StartDate, возвращает продажи за 3 месяца до этой даты.
        Данные кэшируются для указанной даты.
        
        Args:
            end_date: Дата окончания периода продаж
            force_refresh: Принудительное обновление кэша
            
        Returns:
            DataFrame со всей историей продаж
        """
        # Проверяем кэш по дате
        if (self._sales_history_cache is not None and 
            self._reference_date == end_date and 
            not force_refresh):
            return self._sales_history_cache.copy()
        
        conn = None
        try:
            logger.info(f"Загрузка истории продаж до {end_date.strftime('%Y-%m-%d')}...")
            conn = get_connection()
            date_str = end_date.strftime('%Y-%m-%d')
            sql = f"EXEC SNS_ML_Get_Brand_Sales @StartDate = '{date_str}'"
            df = pd.read_sql(sql, conn)
            
            if not df.empty:
                df = self._prepare_sales_history(df)
                self._sales_history_cache = df.copy()
                self._reference_date = end_date
                logger.info(f"Загружено {len(df)} записей истории продаж")
            else:
                logger.warning("История продаж пуста")
                
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при получении истории продаж: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def clear_all_caches(self):
        """Очищает все кэши данных."""
        self._brand_rules_cache = None
        self._sales_history_cache = None
        self._reference_date = None
    
    def _prepare_sales_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит типы данных и подготавливает DataFrame истории продаж."""
        if df.empty:
            return df
            
        # Нормализуем имена колонок в нижний регистр
        df.columns = [col.lower() for col in df.columns]
        
        logger.info(f"Колонки в истории продаж после нормализации: {list(df.columns)}")
        
        # Явно переименовываем возможные вариации имен колонок
        rename_map = {}
        for col in df.columns:
            if col in ['pointid', 'point_id', 'point', 'mfid']:
                rename_map[col] = 'pointid'
            elif col in ['categoryid', 'category_id', 'category']:
                rename_map[col] = 'categoryid'
            elif col in ['brandamount', 'brand_amount', 'amount']:
                rename_map[col] = 'brand_amount'
            elif col in ['microregionid', 'micro_region_id', 'microregion']:
                rename_map[col] = 'microregionid'
            elif col in ['visitdate', 'visit_date']:
                rename_map[col] = 'visitdate'
            elif col in ['brand', 'brandid', 'brand_id']:
                rename_map[col] = 'brand'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        logger.info(f"Колонки после переименования: {list(df.columns)}")
        
        if 'visitdate' in df.columns:
            df['visitdate'] = pd.to_datetime(df['visitdate'])
            df['dayofweek'] = df['visitdate'].dt.dayofweek
            
        return df
    
    def generate_recommendations_batch(
        self,
        predictions_df: pd.DataFrame,
        days_until_visit: int = 7,
        reference_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> List[Tuple[int, int, float, pd.DataFrame]]:
        """
        Оптимизированная пакетная генерация рекомендаций для всех точек и категорий.
        
        За один раз загружает ВСЕ данные из БД и обрабатывает их векторизованно.
        
        Args:
            predictions_df: DataFrame с прогнозами (pointid, categoryid, forecast_amount)
            days_until_visit: Дней до следующего визита
            reference_date: Дата расчета (по умолчанию сегодня)
            force_refresh: Принудительное обновление кэша данных
            
        Returns:
            Список кортежей (point_id, category_id, forecast_amount, recommendation_df)
        """
        if reference_date is None:
            reference_date = datetime.now().date()
        
        target_dow = reference_date.weekday()
        logger.info(f"Пакетная генерация рекомендаций для {len(predictions_df)} пар точка-категория")
        logger.info(f"Целевой день недели: {target_dow} ({['Пн','Вт','Ср','Чт','Пт','Сб','Вс'][target_dow]})")
        
        # 1. Загружаем ВСЕ правила брендов один раз
        brand_rules = self.get_all_brand_rules(force_refresh=force_refresh)
        if brand_rules.empty:
            logger.error("Нет правил брендов")
            return []
        
        # 2. Загружаем ВСЮ историю продаж один раз
        sales_history = self.get_all_sales_history(reference_date, force_refresh=force_refresh)
        
        # Нормализуем имена колонок в predictions
        pred_df = predictions_df.copy()
        pred_df.columns = [col.lower() for col in pred_df.columns]
        
        # Находим колонку с суммой прогноза
        value_col = None
        for col in ['predicted_category_sum', 'sumroubles', 'forecast_amount', 'forecast']:
            if col in pred_df.columns:
                value_col = col
                break
        
        if value_col is None and len(pred_df.columns) >= 3:
            value_col = pred_df.columns[2]
        
        if value_col is None:
            logger.error("Не найдена колонка с прогнозом суммы")
            return []
        
        results = []
        
        # Группируем прогнозы по уникальным парам точка-категория
        if 'pointid' not in pred_df.columns or 'categoryid' not in pred_df.columns:
            logger.error("Отсутствуют колонки pointid/categoryid в прогнозах")
            return []
        
        grouped = pred_df.groupby(['pointid', 'categoryid'], as_index=False)[value_col].mean()
        if value_col != 'sumroubles':
            grouped = grouped.rename(columns={value_col: 'sumroubles'})
        
        total_pairs = len(grouped)
        logger.info(f"Обработка {total_pairs} уникальных пар точка-категория...")
        
        # Предварительно рассчитываем средние продажи по всем точкам/брендам/дням недели
        if not sales_history.empty and 'brand' in sales_history.columns:
            # Агрегируем продажи: точка x категория x бренд x день недели -> среднее количество
            sales_agg = sales_history.groupby(
                ['pointid', 'categoryid', 'brand', 'dayofweek'], 
                as_index=False
            )['brand_amount'].mean().rename(columns={'brand_amount': 'avg_sales'})
        else:
            sales_agg = pd.DataFrame()
        
        # Предварительно определяем топовые бренды по микрорегионам
        if not sales_history.empty and 'microregionid' in sales_history.columns:
            top_brands_by_region = self._calculate_top_brands_by_region(sales_history, target_dow)
        else:
            top_brands_by_region = {}
        
        # Обрабатываем каждую пару точка-категория
        for idx, row in grouped.iterrows():
            point_id = int(row['pointid'])
            category_id = int(row['categoryid'])
            forecast_amount = float(row['sumroubles'])
            
            try:
                # Фильтруем правила брендов для категории
                cat_rules = brand_rules[brand_rules['categoryid'] == category_id].copy()
                if cat_rules.empty:
                    continue
                
                # Фильтруем продажи для точки и категории
                if not sales_agg.empty:
                    point_sales = sales_agg[
                        (sales_agg['pointid'] == point_id) & 
                        (sales_agg['categoryid'] == category_id) &
                        (sales_agg['dayofweek'] == target_dow)
                    ]
                else:
                    point_sales = pd.DataFrame()
                
                # Получаем MicroRegionID для точки (если есть в истории)
                microregion_id = None
                if not sales_history.empty and 'microregionid' in sales_history.columns:
                    point_regions = sales_history[sales_history['pointid'] == point_id]['microregionid'].dropna()
                    if len(point_regions) > 0:
                        microregion_id = point_regions.iloc[0]
                
                # Векторизованный расчет для всех брендов категории
                rec_df = self._calculate_brand_recommendations_vectorized(
                    cat_rules, 
                    point_sales, 
                    days_until_visit,
                    microregion_id,
                    top_brands_by_region,
                    target_dow
                )
                
                if rec_df.empty:
                    continue
                
                # Распределяем бюджет
                final_df = self._distribute_forecast_budget(rec_df, forecast_amount)
                included_df = final_df[final_df['included']].sort_values('priority', ascending=False)
                
                if not included_df.empty:
                    results.append((point_id, category_id, forecast_amount, included_df))
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Обработано {idx + 1}/{total_pairs} пар")
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке точки {point_id}, категории {category_id}: {e}")
                continue
        
        logger.info(f"Сгенерировано рекомендаций для {len(results)} пар точка-категория из {total_pairs}")
        return results
    
    def _calculate_top_brands_by_region(
        self, 
        sales_history: pd.DataFrame, 
        target_dow: int,
        top_percentile: float = 0.8
    ) -> Dict[Tuple[int, int], set]:
        """
        Рассчитывает топовые бренды для каждого микрорегиона и категории.
        
        Returns:
            Dict[(microregion_id, category_id), set of top brand_ids]
        """
        if sales_history.empty or 'microregionid' not in sales_history.columns:
            return {}
        
        filtered = sales_history[sales_history['dayofweek'] == target_dow]
        if filtered.empty:
            return {}
        
        # Группируем по микрорегиону x категория x бренд
        brand_sales = filtered.groupby(
            ['microregionid', 'categoryid', 'brand']
        )['brand_amount'].sum().reset_index()
        
        top_brands = {}
        for (region, category), group in brand_sales.groupby(['microregionid', 'categoryid']):
            max_sales = group['brand_amount'].max()
            if max_sales > 0:
                threshold = max_sales * top_percentile
                top_in_group = group[group['brand_amount'] >= threshold]['brand'].tolist()
                top_brands[(region, category)] = set(top_in_group)
        
        return top_brands
    
    def _calculate_brand_recommendations_vectorized(
        self,
        cat_rules: pd.DataFrame,
        point_sales: pd.DataFrame,
        days_until_visit: int,
        microregion_id: Optional[int],
        top_brands_by_region: Dict[Tuple[int, int], set],
        target_dow: int
    ) -> pd.DataFrame:
        """
        Векторизованный расчет рекомендаций для всех брендов категории.
        """
        if cat_rules.empty:
            return pd.DataFrame()
        
        # Merge с продажами для получения avg_sales
        df = cat_rules.merge(
            point_sales[['brand', 'avg_sales']], 
            on='brand', 
            how='left'
        )
        
        # Заполняем нулями отсутствующие продажи
        df['avg_sales'] = df['avg_sales'].fillna(0.0)
        
        # Расчет raw потребности
        df['raw_need'] = df['avg_sales'] * days_until_visit
        
        # Округление до кванта (векторизованно)
        df['recommended_qty'] = np.ceil(df['raw_need'] / df['brandquantum']).astype(int) * df['brandquantum']
        df.loc[df['raw_need'] <= 0, 'recommended_qty'] = 0
        
        # Расчет стоимости
        df['estimated_cost'] = df['recommended_qty'] * df['avg_price']
        
        # Расчет приоритета
        df['priority'] = 1 + df['priorityweight']
        df.loc[df['isturbobrand'] == 1, 'priority'] += 5
        
        # Проверка на топ в микрорегионе
        if microregion_id is not None:
            top_brands_set = top_brands_by_region.get((microregion_id, cat_rules['categoryid'].iloc[0]), set())
            df['is_top_local'] = df['brand'].isin(top_brands_set)
            df.loc[df['is_top_local'], 'priority'] += 2
        else:
            df['is_top_local'] = False
        
        # Дополнительные поля
        df['brand_id'] = df['brand']
        df['brand_name'] = df.get('brand_name', df['brand'].astype(str))
        df['is_turbo'] = df['isturbobrand']
        df['importance_label'] = df.get('importancelabel', '')
        df['days_until_visit'] = days_until_visit
        df['quantum'] = df['brandquantum']
        df['avg_daily_sales'] = df['avg_sales'].round(2)
        df['avg_price'] = df['avgprice']
        
        return df
    
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
        reference_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Полный цикл формирования рекомендации заказа.
        
        Args:
            point_id: ID точки
            category_id: ID категории
            forecast_amount: Прогноз суммы продаж категории (от ML модели)
            days_until_visit: Дней до следующего визита мерчандайзера
            reference_date: Дата расчета (по умолчанию сегодня)
            use_cache: Использовать кэширование данных (по умолчанию True)
            
        Returns:
            DataFrame с рекомендованным заказом
        """
        if reference_date is None:
            reference_date = datetime.now()
            
        target_dow = reference_date.weekday()
        
        # 1. Получаем правила брендов для категории (с кэшированием)
        brand_rules = self.get_brand_rules(
            category_id=category_id, 
            force_refresh=not use_cache
        )
        if brand_rules.empty:
            logger.warning(f"Нет правил брендов для категории {category_id}")
            return pd.DataFrame()
            
        # 2. Получаем историю продаж точки (с кэшированием)
        sales_history = self.get_brand_sales_history(
            end_date=reference_date,
            point_id=point_id,
            category_id=category_id,
            force_refresh=not use_cache
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
    
    def clear_all_caches(self):
        """Очищает все кэши данных."""
        self._brand_rules_cache = None
        self._sales_history_cache = None
        self._reference_date = None
    
    # =====================================================================
    # СТАРЫЕ МЕТОДЫ (для обратной совместимости, не используются в новой версии)
    # =====================================================================
    
    def get_brand_rules(self, category_id: Optional[int] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Устаревший метод. Используйте get_all_brand_rules()."""
        df = self.get_all_brand_rules(force_refresh=force_refresh)
        if category_id is not None and 'categoryid' in df.columns:
            return df[df['categoryid'] == category_id]
        return df
    
    def get_brand_sales_history(
        self, 
        end_date: datetime,
        point_id: Optional[int] = None,
        category_id: Optional[int] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Устаревший метод. Используйте get_all_sales_history()."""
        df = self.get_all_sales_history(end_date, force_refresh=force_refresh)
        
        if point_id is not None and 'pointid' in df.columns:
            df = df[df['pointid'] == point_id]
        
        if category_id is not None and 'categoryid' in df.columns:
            df = df[df['categoryid'] == category_id]
            
        return df
    
    def clear_brand_rules_cache(self):
        """Устаревший метод. Используйте clear_all_caches()."""
        self._brand_rules_cache = None
    
    def clear_sales_history_cache(self):
        """Устаревший метод. Используйте clear_all_caches()."""
        self._sales_history_cache = None
        self._reference_date = None


if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТ ПАКЕТНОЙ ГЕНЕРАЦИИ РЕКОМЕНДАЦИЙ")
    print("=" * 80)
    
    recommender = OrderRecommender()
    
    # Создаем тестовые данные прогнозов
    test_predictions = pd.DataFrame({
        'pointid': [12345, 12345, 67890, 67890],
        'categoryid': [101, 102, 101, 102],
        'sumroubles': [50000.0, 30000.0, 45000.0, 35000.0]
    })
    
    try:
        print(f"\nТестовые прогнозы для {len(test_predictions)} пар точка-категория:")
        print(test_predictions.to_string())
        
        # Тестируем пакетную генерацию
        recommendations = recommender.generate_recommendations_batch(
            predictions_df=test_predictions,
            days_until_visit=7,
            force_refresh=True
        )
        
        if recommendations:
            print(f"\n✓ Сгенерировано рекомендаций для {len(recommendations)} пар точка-категория:")
            
            total_brands = 0
            total_cost = 0.0
            
            for point_id, category_id, forecast, rec_df in recommendations:
                n_brands = len(rec_df)
                cost = rec_df['estimated_cost'].sum()
                total_brands += n_brands
                total_cost += cost
                
                print(f"\n  Точка {point_id}, Категория {category_id}:")
                print(f"    Прогноз: {forecast:,.2f} руб.")
                print(f"    Рекомендовано брендов: {n_brands}")
                print(f"    Сумма заказа: {cost:,.2f} руб.")
                
                if n_brands > 0:
                    top3 = rec_df.head(3)
                    print(f"    Топ-3 бренда:")
                    for _, row in top3.iterrows():
                        print(f"      - {row.get('brand_name', row['brand_id'])}: {row['recommended_qty']} шт. ({row['estimated_cost']:,.2f} руб.)")
            
            print(f"\n{'=' * 80}")
            print(f"ИТОГО:")
            print(f"  Пар точка-категория: {len(recommendations)}")
            print(f"  Всего рекомендованных позиций: {total_brands}")
            print(f"  Общая сумма заказов: {total_cost:,.2f} руб.")
            print(f"{'=' * 80}")
        else:
            print("\n⚠ Рекомендации не сформированы (возможно, нет данных в БД)")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
