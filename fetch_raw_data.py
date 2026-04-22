import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional
import logging
import os
from dotenv import load_dotenv
import numpy as np

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация подключения к MSSQL из переменных окружения
SQL_CONFIG = {
    'driver': os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server'),
    'server': os.getenv('SQL_SERVER'),
    'database': os.getenv('SQL_DATABASE'),
    'user': os.getenv('SQL_USER'),
    'password': os.getenv('SQL_PASSWORD')
}

# Валидация конфигурации
required_config_keys = ['server', 'database', 'user', 'password']
missing_keys = [key for key in required_config_keys if not SQL_CONFIG.get(key)]
if missing_keys:
    raise ValueError(f"Отсутствуют обязательные параметры подключения: {', '.join(missing_keys)}")


def get_connection() -> pyodbc.Connection:
    """
    Создает и возвращает соединение с базой данных.
    
    Returns:
        pyodbc.Connection: Объект соединения с БД
        
    Raises:
        pyodbc.Error: Ошибка подключения к базе данных
    """
    conn_str = (
        f"DRIVER={{{SQL_CONFIG['driver']}}}; "
        f"SERVER={SQL_CONFIG['server']}; "
        f"DATABASE={SQL_CONFIG['database']}; "
        f"UID={SQL_CONFIG['user']}; "
        f"PWD={SQL_CONFIG['password']}"
    )
    logger.info("Установление соединения с базой данных...")
    return pyodbc.connect(conn_str)


def validate_dates(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> tuple:
    """
    Валидирует и преобразует даты в формат строки YYYY-MM-DD.
    
    Args:
        start_date: Начальная дата (строка или объект date/datetime)
        end_date: Конечная дата (строка или объект date/datetime)
        
    Returns:
        tuple: Кортеж (start_date_str, end_date_str) в формате 'YYYY-MM-DD'
        
    Raises:
        ValueError: Если даты некорректны или start_date > end_date
    """
    try:
        # Преобразование строк в объекты datetime если необходимо
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif hasattr(start_date, 'date'):
            start_dt = start_date.date() if callable(start_date.date) else start_date
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif hasattr(end_date, 'date'):
            end_dt = end_date.date() if callable(end_date.date) else end_date
        else:
            end_dt = end_date
        
        # Проверка корректности диапазона дат
        if start_dt > end_dt:
            raise ValueError(f"Начальная дата ({start_dt}) не может быть больше конечной ({end_dt})")
        
        return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
        
    except ValueError as e:
        logger.error(f"Ошибка валидации дат: {e}")
        raise


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет календарные фичи для подготовки модели машинного обучения.
    
    Аргументы:
        df: DataFrame с данными, содержащий колонку 'date' или 'Date'
        
    Возвращает:
        DataFrame с добавленными календарными фичами:
        - day_of_week: день недели (0-понедельник, 6-воскресенье)
        - is_weekend: флаг выходного дня (True/False)
        - is_monday: флаг понедельника
        - is_friday: флаг пятницы (день перед выходными)
        - is_holiday: флаг праздника
        - is_pre_holiday: флаг предпраздничного дня
        - is_post_holiday: флаг дня после праздника/выходных
        - days_to_holiday: количество дней до ближайшего праздника
        - days_from_holiday: количество дней от последнего праздника
        - month: месяц года
        - quarter: квартал года
        - week_of_year: неделя года
        - is_month_start: флаг начала месяца
        - is_month_end: флаг конца месяца
        
    Примечание:
        Функция ожидает, что в DataFrame есть колонка с датой ('date' или 'Date')
    """
    if df.empty:
        return df
    
    # Определение колонки с датой
    date_col = None
    for col_name in ['date', 'Date', 'DATE']:
        if col_name in df.columns:
            date_col = col_name
            break
    
    if date_col is None:
        logger.warning("Колонка с датой не найдена. Календарные фичи не добавлены.")
        return df
    
    # Создание копии DataFrame для предотвращения модификации оригинала
    df_result = df.copy()
    
    # Преобразование колонки с датой в datetime формат
    df_result[date_col] = pd.to_datetime(df_result[date_col])
    
    # Извлечение дня недели (0=понедельник, 6=воскресенье)
    df_result['day_of_week'] = df_result[date_col].dt.dayofweek
    
    # Флаг выходного дня (суббота=5, воскресенье=6)
    df_result['is_weekend'] = df_result['day_of_week'].isin([5, 6])
    
    # Флаги отдельных дней недели
    df_result['is_monday'] = df_result['day_of_week'] == 0
    df_result['is_friday'] = df_result['day_of_week'] == 4  # День перед выходными
    df_result['is_saturday'] = df_result['day_of_week'] == 5
    df_result['is_sunday'] = df_result['day_of_week'] == 6
    
    # Список праздников России (фиксированные даты)
    fixed_holidays = [
        (1, 1),    # Новый год
        (1, 2),    # Новогодние каникулы
        (1, 3),    # Новогодние каникулы
        (1, 4),    # Новогодние каникулы
        (1, 5),    # Новогодние каникулы
        (1, 6),    # Новогодние каникулы
        (1, 7),    # Рождество
        (1, 8),    # Новогодние каникулы
        (2, 23),   # День защитника Отечества
        (3, 8),    # Международный женский день
        (5, 1),    # Праздник Весны и Труда
        (5, 9),    # День Победы
        (6, 12),   # День России
        (11, 4),   # День народного единства
    ]
    
    # Функция для определения праздника
    def is_holiday_date(date):
        """Проверяет, является ли дата праздником"""
        return (date.month, date.day) in fixed_holidays
    
    # Функция для определения предпраздничного дня
    def is_pre_holiday_date(date):
        """Проверяет, является ли дата предпраздничным днем"""
        next_day = date + timedelta(days=1)
        return is_holiday_date(next_day) or next_day.weekday() >= 5
    
    # Функция для определения дня после праздника/выходных
    def is_post_holiday_date(date):
        """Проверяет, является ли дата днем после праздника или выходных"""
        prev_day = date - timedelta(days=1)
        return is_holiday_date(prev_day) or prev_day.weekday() >= 5
    
    # Функция для расчета дней до ближайшего праздника
    def days_until_holiday(date):
        """Возвращает количество дней до следующего праздника"""
        for i in range(1, 366):  # Ищем в пределах года
            future_date = date + timedelta(days=i)
            if is_holiday_date(future_date):
                return i
        return 365  # Если праздников нет в пределах года
    
    # Функция для расчета дней от последнего праздника
    def days_since_holiday(date):
        """Возвращает количество дней от последнего праздника"""
        for i in range(1, 366):  # Ищем в пределах года
            past_date = date - timedelta(days=i)
            if is_holiday_date(past_date):
                return i
        return 365  # Если праздников нет в пределах года
    
    # Применение функций для создания фичей
    df_result['is_holiday'] = df_result[date_col].apply(is_holiday_date)
    df_result['is_pre_holiday'] = df_result[date_col].apply(is_pre_holiday_date)
    df_result['is_post_holiday'] = df_result[date_col].apply(is_post_holiday_date)
    df_result['days_to_holiday'] = df_result[date_col].apply(days_until_holiday)
    df_result['days_from_holiday'] = df_result[date_col].apply(days_since_holiday)
    
    # Дополнительные календарные фичи
    df_result['month'] = df_result[date_col].dt.month
    df_result['quarter'] = df_result[date_col].dt.quarter
    df_result['week_of_year'] = df_result[date_col].dt.isocalendar().week.astype(int)
    df_result['is_month_start'] = df_result[date_col].dt.is_month_start
    df_result['is_month_end'] = df_result[date_col].dt.is_month_end
    df_result['day_of_month'] = df_result[date_col].dt.day
    df_result['day_of_year'] = df_result[date_col].dt.dayofyear
    
    logger.info(f"Добавлены календарные фичи: {len(df_result.columns)} колонок всего")
    
    return df_result


def add_order_history_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Добавляет фичи истории заказов для подготовки модели машинного обучения.
    
    Аргументы:
        df: DataFrame с данными продаж
        start_date: Начальная дата периода выгрузки (строка YYYY-MM-DD)
        end_date: Конечная дата периода выгрузки (строка YYYY-MM-DD)
        
    Возвращает:
        DataFrame с добавленными фичами:
        - Days_Since_Last_Order_Category: дней назад точка брала эту категорию
        - Days_Since_Last_Order_Total: дней назад был любой заказ от точки
        - Average_Interval_Category: средний интервал между закупками категории
        
    Примечание:
        Требуется подключение к БД для получения исторических данных
        Ожидает наличие колонок: 'date', 'outlet_id' (или 'OutletID'), 'category_id' (или 'CategoryID')
    """
    if df.empty:
        return df
    
    # Определение имен колонок
    date_col = None
    for col_name in ['date', 'Date', 'DATE']:
        if col_name in df.columns:
            date_col = col_name
            break
    
    outlet_col = None
    for col_name in ['outlet_id', 'OutletID', 'OUTLET_ID', 'outletId']:
        if col_name in df.columns:
            outlet_col = col_name
            break
    
    category_col = None
    for col_name in ['category_id', 'CategoryID', 'CATEGORY_ID', 'categoryId']:
        if col_name in df.columns:
            category_col = col_name
            break
    
    # Проверка наличия обязательных колонок
    missing_cols = []
    if date_col is None:
        missing_cols.append('date')
    if outlet_col is None:
        missing_cols.append('outlet_id')
    
    if missing_cols:
        logger.warning(f"Отсутствуют обязательные колонки для history features: {', '.join(missing_cols)}. Фичи не добавлены.")
        return df
    
    # Создание копии DataFrame
    df_result = df.copy()
    
    # Преобразование даты
    df_result[date_col] = pd.to_datetime(df_result[date_col])
    
    # Получение исторических данных из БД
    conn = None
    try:
        logger.info("Загрузка исторических данных для расчета фичей заказов...")
        
        # Запрос на получение всей истории заказов до start_date
        historical_query = f"""
        SELECT 
            t.{date_col} as order_date,
            t.{outlet_col} as outlet_id,
            t.{category_col if category_col else 'NULL'} as category_id
        FROM OPENQUERY(
            OPTIMUM_LEPETSK, 
            'SELECT TOP 1000000 
                CAST(OrderDate AS DATE) as order_date,
                OutletID,
                CategoryID
             FROM Sales.vw_ML_Raw_Data
             WHERE OrderDate < '''{start_date}'''
             ORDER BY OrderDate DESC'
        ) t
        """
        
        # Если нет category_id в исходных данных, упрощаем запрос
        if category_col is None:
            historical_query = f"""
            SELECT 
                t.{date_col} as order_date,
                t.{outlet_col} as outlet_id,
                NULL as category_id
            FROM OPENQUERY(
                OPTIMUM_LEPETSK, 
                'SELECT TOP 1000000 
                    CAST(OrderDate AS DATE) as order_date,
                    OutletID,
                    NULL as CategoryID
                 FROM Sales.vw_ML_Raw_Data
                 WHERE OrderDate < '''{start_date}'''
                 ORDER BY OrderDate DESC'
            ) t
            """
        
        conn = get_connection()
        df_history = pd.read_sql(historical_query, conn)
        
        if df_history.empty:
            logger.warning("Исторические данные не найдены. Фичи истории заказов не добавлены.")
            df_result['Days_Since_Last_Order_Category'] = np.nan
            df_result['Days_Since_Last_Order_Total'] = np.nan
            df_result['Average_Interval_Category'] = np.nan
            return df_result
        
        # Преобразование даты в истории
        df_history['order_date'] = pd.to_datetime(df_history['order_date'])
        
        # Инициализация колонок
        df_result['Days_Since_Last_Order_Category'] = np.nan
        df_result['Days_Since_Last_Order_Total'] = np.nan
        df_result['Average_Interval_Category'] = np.nan
        
        # Расчет для каждой уникальной комбинации outlet + date
        unique_outlets = df_result[outlet_col].unique()
        
        logger.info(f"Расчет фичей для {len(unique_outlets)} точек продаж...")
        
        for outlet_id in unique_outlets:
            # Фильтрация истории по точке
            outlet_history = df_history[df_history['outlet_id'] == outlet_id].copy()
            
            if outlet_history.empty:
                continue
            
            # Сортировка по дате
            outlet_history = outlet_history.sort_values('order_date')
            
            # Расчет Days_Since_Last_Order_Total для каждой даты в df_result
            outlet_mask = df_result[outlet_col] == outlet_id
            outlet_dates = df_result.loc[outlet_mask, date_col]
            
            for idx in outlet_dates.index:
                current_date = outlet_dates.loc[idx]
                
                # Days_Since_Last_Order_Total
                past_orders = outlet_history[outlet_history['order_date'] < current_date]
                if not past_orders.empty:
                    last_order_date = past_orders['order_date'].max()
                    days_since = (current_date - last_order_date).days
                    df_result.loc[idx, 'Days_Since_Last_Order_Total'] = days_since
                
                # Days_Since_Last_Order_Category (если есть category_id)
                if category_col is not None:
                    current_category = df_result.loc[idx, category_col]
                    if pd.notna(current_category):
                        category_history = outlet_history[
                            (outlet_history['order_date'] < current_date) & 
                            (outlet_history['category_id'] == current_category)
                        ]
                        if not category_history.empty:
                            last_category_date = category_history['order_date'].max()
                            days_since_cat = (current_date - last_category_date).days
                            df_result.loc[idx, 'Days_Since_Last_Order_Category'] = days_since_cat
            
            # Расчет Average_Interval_Category (средний интервал между заказами категории)
            if category_col is not None and not outlet_history[category_col].isna().all():
                for category_id in outlet_history[category_col].dropna().unique():
                    cat_history = outlet_history[
                        outlet_history[category_col] == category_id
                    ].sort_values('order_date')
                    
                    if len(cat_history) > 1:
                        # Расчет интервалов между заказами
                        intervals = cat_history['order_date'].diff().dt.days.dropna()
                        avg_interval = intervals.mean() if len(intervals) > 0 else np.nan
                        
                        # Присвоение среднего интервала всем записям этой категории в этой точке
                        mask = (df_result[outlet_col] == outlet_id) & \
                               (df_result[category_col] == category_id)
                        df_result.loc[mask, 'Average_Interval_Category'] = avg_interval
        
        logger.info(f"Добавлены фичи истории заказов: {len(df_result.columns)} колонок всего")
        
        return df_result
        
    except Exception as e:
        logger.error(f"Ошибка при расчете фичей истории заказов: {e}")
        df_result['Days_Since_Last_Order_Category'] = np.nan
        df_result['Days_Since_Last_Order_Total'] = np.nan
        df_result['Average_Interval_Category'] = np.nan
        return df_result
    finally:
        if conn:
            conn.close()


def add_sales_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Добавляет фичи продаж для подготовки модели машинного обучения.
    
    Аргументы:
        df: DataFrame с данными продаж
        start_date: Начальная дата периода выгрузки (строка YYYY-MM-DD)
        end_date: Конечная дата периода выгрузки (строка YYYY-MM-DD)
        
    Возвращает:
        DataFrame с добавленными фичами:
        - Prev_Order_Amount_Category: сумма предыдущего заказа по категории
        - SMA_3_Category: средняя сумма закупки категории за последние 3 дня
        - SMA_7_Category: средняя сумма закупки категории за последние 7 дней
        - SMA_30_Category: средняя сумма закупки категории за последние 30 дней
        - Momentum_Category: отношение среднего чека за неделю к среднему чеку за месяц
        - StdDev_Category: скользящее стандартное отклонение для категории (стабильность спроса)
        
    Примечание:
        Требуется подключение к БД для получения исторических данных
        Ожидает наличие колонок: 'date', 'outlet_id', 'category_id', 'sales_amount' (или аналог)
    """
    if df.empty:
        return df
    
    # Определение имен колонок
    date_col = None
    for col_name in ['date', 'Date', 'DATE']:
        if col_name in df.columns:
            date_col = col_name
            break
    
    outlet_col = None
    for col_name in ['outlet_id', 'OutletID', 'OUTLET_ID', 'outletId']:
        if col_name in df.columns:
            outlet_col = col_name
            break
    
    category_col = None
    for col_name in ['category_id', 'CategoryID', 'CATEGORY_ID', 'categoryId']:
        if col_name in df.columns:
            category_col = col_name
            break
    
    amount_col = None
    for col_name in ['sales_amount', 'SalesAmount', 'SALES_AMOUNT', 'amount', 'Amount', 'sum', 'Sum']:
        if col_name in df.columns:
            amount_col = col_name
            break
    
    # Проверка наличия обязательных колонок
    missing_cols = []
    if date_col is None:
        missing_cols.append('date')
    if outlet_col is None:
        missing_cols.append('outlet_id')
    if amount_col is None:
        missing_cols.append('sales_amount')
    
    if missing_cols:
        logger.warning(f"Отсутствуют обязательные колонки для sales features: {', '.join(missing_cols)}. Фичи не добавлены.")
        return df
    
    # Создание копии DataFrame
    df_result = df.copy()
    
    # Преобразование даты
    df_result[date_col] = pd.to_datetime(df_result[date_col])
    
    # Получение исторических данных из БД
    conn = None
    try:
        logger.info("Загрузка исторических данных для расчета фичей продаж...")
        
        # Запрос на получение всей истории продаж до start_date
        historical_query = f"""
        SELECT 
            t.{date_col} as order_date,
            t.{outlet_col} as outlet_id,
            t.{category_col if category_col else 'NULL'} as category_id,
            t.{amount_col} as sales_amount
        FROM OPENQUERY(
            OPTIMUM_LEPETSK, 
            'SELECT TOP 1000000 
                CAST(OrderDate AS DATE) as order_date,
                OutletID,
                CategoryID,
                SalesAmount
             FROM Sales.vw_ML_Raw_Data
             WHERE OrderDate < '''{start_date}'''
             ORDER BY OrderDate DESC'
        ) t
        """
        
        # Если нет category_id в исходных данных, упрощаем запрос
        if category_col is None:
            historical_query = f"""
            SELECT 
                t.{date_col} as order_date,
                t.{outlet_col} as outlet_id,
                NULL as category_id,
                t.{amount_col} as sales_amount
            FROM OPENQUERY(
                OPTIMUM_LEPETSK, 
                'SELECT TOP 1000000 
                    CAST(OrderDate AS DATE) as order_date,
                    OutletID,
                    NULL as CategoryID,
                    SalesAmount
                 FROM Sales.vw_ML_Raw_Data
                 WHERE OrderDate < '''{start_date}'''
                 ORDER BY OrderDate DESC'
            ) t
            """
        
        conn = get_connection()
        df_history = pd.read_sql(historical_query, conn)
        
        if df_history.empty:
            logger.warning("Исторические данные по продажам не найдены. Фичи продаж не добавлены.")
            df_result['Prev_Order_Amount_Category'] = np.nan
            df_result['SMA_3_Category'] = np.nan
            df_result['SMA_7_Category'] = np.nan
            df_result['SMA_30_Category'] = np.nan
            df_result['Momentum_Category'] = np.nan
            df_result['StdDev_Category'] = np.nan
            return df_result
        
        # Преобразование даты в истории
        df_history['order_date'] = pd.to_datetime(df_history['order_date'])
        df_history['sales_amount'] = pd.to_numeric(df_history['sales_amount'], errors='coerce')
        
        # Инициализация колонок
        df_result['Prev_Order_Amount_Category'] = np.nan
        df_result['SMA_3_Category'] = np.nan
        df_result['SMA_7_Category'] = np.nan
        df_result['SMA_30_Category'] = np.nan
        df_result['Momentum_Category'] = np.nan
        df_result['StdDev_Category'] = np.nan
        
        # Расчет для каждой уникальной комбинации outlet + category
        if category_col is not None:
            unique_groups = df_result[[outlet_col, category_col]].drop_duplicates()
            logger.info(f"Расчет фичей продаж для {len(unique_groups)} групп (точка x категория)...")
            
            for _, row in unique_groups.iterrows():
                outlet_id = row[outlet_col]
                category_id = row[category_col]
                
                if pd.isna(category_id):
                    continue
                
                # Фильтрация истории по точке и категории
                group_history = df_history[
                    (df_history['outlet_id'] == outlet_id) & 
                    (df_history['category_id'] == category_id)
                ].sort_values('order_date')
                
                if group_history.empty:
                    continue
                
                # Фильтрация данных для текущей группы в df_result
                group_mask = (df_result[outlet_col] == outlet_id) & \
                            (df_result[category_col] == category_id)
                group_dates = df_result.loc[group_mask, date_col].sort_values()
                
                for idx in group_dates.index:
                    current_date = group_dates.loc[idx]
                    
                    # Прошлые заказы для этой группы
                    past_orders = group_history[group_history['order_date'] < current_date]
                    
                    if past_orders.empty:
                        continue
                    
                    # 1. Сумма предыдущего заказа по категории
                    last_order = past_orders[past_orders['order_date'] == past_orders['order_date'].max()]
                    prev_amount = last_order['sales_amount'].sum()
                    df_result.loc[idx, 'Prev_Order_Amount_Category'] = prev_amount
                    
                    # 2. SMA (Simple Moving Average) за 3, 7, 30 дней
                    # SMA_3: среднее за последние 3 дня
                    last_3_days = past_orders[
                        past_orders['order_date'] >= (current_date - timedelta(days=3))
                    ]
                    if not last_3_days.empty:
                        df_result.loc[idx, 'SMA_3_Category'] = last_3_days.groupby('order_date')['sales_amount'].sum().mean()
                    
                    # SMA_7: среднее за последние 7 дней
                    last_7_days = past_orders[
                        past_orders['order_date'] >= (current_date - timedelta(days=7))
                    ]
                    if not last_7_days.empty:
                        df_result.loc[idx, 'SMA_7_Category'] = last_7_days.groupby('order_date')['sales_amount'].sum().mean()
                    
                    # SMA_30: среднее за последние 30 дней
                    last_30_days = past_orders[
                        past_orders['order_date'] >= (current_date - timedelta(days=30))
                    ]
                    if not last_30_days.empty:
                        df_result.loc[idx, 'SMA_30_Category'] = last_30_days.groupby('order_date')['sales_amount'].sum().mean()
                    
                    # 3. Импульс (Momentum): отношение среднего за неделю к среднему за месяц
                    week_avg = last_7_days.groupby('order_date')['sales_amount'].sum().mean() if not last_7_days.empty else np.nan
                    month_avg = last_30_days.groupby('order_date')['sales_amount'].sum().mean() if not last_30_days.empty else np.nan
                    
                    if pd.notna(week_avg) and pd.notna(month_avg) and month_avg > 0:
                        df_result.loc[idx, 'Momentum_Category'] = week_avg / month_avg
                    
                    # 4. Скользящее стандартное отклонение (за 30 дней)
                    if len(last_30_days) > 1:
                        daily_sums = last_30_days.groupby('order_date')['sales_amount'].sum()
                        df_result.loc[idx, 'StdDev_Category'] = daily_sums.std()
        else:
            # Если нет category_id, считаем только по точкам
            unique_outlets = df_result[outlet_col].unique()
            logger.info(f"Расчет фичей продаж для {len(unique_outlets)} точек продаж...")
            
            for outlet_id in unique_outlets:
                outlet_history = df_history[
                    df_history['outlet_id'] == outlet_id
                ].sort_values('order_date')
                
                if outlet_history.empty:
                    continue
                
                outlet_mask = df_result[outlet_col] == outlet_id
                outlet_dates = df_result.loc[outlet_mask, date_col].sort_values()
                
                for idx in outlet_dates.index:
                    current_date = outlet_dates.loc[idx]
                    past_orders = outlet_history[outlet_history['order_date'] < current_date]
                    
                    if past_orders.empty:
                        continue
                    
                    # 1. Сумма предыдущего заказа
                    last_order = past_orders[past_orders['order_date'] == past_orders['order_date'].max()]
                    prev_amount = last_order['sales_amount'].sum()
                    df_result.loc[idx, 'Prev_Order_Amount_Category'] = prev_amount
                    
                    # 2. SMA за 3, 7, 30 дней
                    last_3_days = past_orders[past_orders['order_date'] >= (current_date - timedelta(days=3))]
                    if not last_3_days.empty:
                        df_result.loc[idx, 'SMA_3_Category'] = last_3_days.groupby('order_date')['sales_amount'].sum().mean()
                    
                    last_7_days = past_orders[past_orders['order_date'] >= (current_date - timedelta(days=7))]
                    if not last_7_days.empty:
                        df_result.loc[idx, 'SMA_7_Category'] = last_7_days.groupby('order_date')['sales_amount'].sum().mean()
                    
                    last_30_days = past_orders[past_orders['order_date'] >= (current_date - timedelta(days=30))]
                    if not last_30_days.empty:
                        df_result.loc[idx, 'SMA_30_Category'] = last_30_days.groupby('order_date')['sales_amount'].sum().mean()
                    
                    # 3. Импульс
                    week_avg = last_7_days.groupby('order_date')['sales_amount'].sum().mean() if not last_7_days.empty else np.nan
                    month_avg = last_30_days.groupby('order_date')['sales_amount'].sum().mean() if not last_30_days.empty else np.nan
                    
                    if pd.notna(week_avg) and pd.notna(month_avg) and month_avg > 0:
                        df_result.loc[idx, 'Momentum_Category'] = week_avg / month_avg
                    
                    # 4. Стандартное отклонение
                    if len(last_30_days) > 1:
                        daily_sums = last_30_days.groupby('order_date')['sales_amount'].sum()
                        df_result.loc[idx, 'StdDev_Category'] = daily_sums.std()
        
        logger.info(f"Добавлены фичи продаж: {len(df_result.columns)} колонок всего")
        
        return df_result
    
    except Exception as e:
        logger.error(f"Ошибка при расчете фичей продаж: {e}")
        df_result['Prev_Order_Amount_Category'] = np.nan
        df_result['SMA_3_Category'] = np.nan
        df_result['SMA_7_Category'] = np.nan
        df_result['SMA_30_Category'] = np.nan
        df_result['Momentum_Category'] = np.nan
        df_result['StdDev_Category'] = np.nan
        return df_result
    finally:
        if conn:
            conn.close()


def fetch_raw_data(start_date: Union[str, datetime], end_date: Union[str, datetime], 
                   add_features: bool = False, add_history_features: bool = False, 
                   add_sales_features_flag: bool = False) -> Optional[pd.DataFrame]:
    """
    Вызывает хранимую процедуру SNS_ML_Get_Raw_Data и возвращает данные в DataFrame.
    
    Args:
        start_date: Начальная дата (YYYY-MM-DD или объект date/datetime)
        end_date: Конечная дата (YYYY-MM-DD или объект date/datetime)
        add_features: Если True, добавляет календарные фичи для ML
        add_history_features: Если True, добавляет фичи истории заказов (Days_Since_Last_Order_Category,
                              Days_Since_Last_Order_Total, Average_Interval_Category)
        add_sales_features_flag: Если True, добавляет фичи продаж (Prev_Order_Amount_Category,
                                 SMA_3_Category, SMA_7_Category, SMA_30_Category, 
                                 Momentum_Category, StdDev_Category)
        
    Returns:
        pd.DataFrame: DataFrame с результатами выполнения хранимой процедуры
                     или None если данные не получены
                     
    Raises:
        ValueError: Ошибка валидации входных параметров
        pyodbc.Error: Ошибка выполнения запроса к базе данных
        Exception: Другие ошибки выполнения
    """
    # Валидация дат
    start_date_str, end_date_str = validate_dates(start_date, end_date)
    
    query = """
    EXEC dbo.SNS_ML_Get_Raw_Data 
        @StartDate = ?, 
        @EndDate = ?
    """
    
    conn = None
    try:
        logger.info(f"Загрузка данных за период с {start_date_str} по {end_date_str}")
        conn = get_connection()
        cursor = conn.cursor()
        
        # Выполнение хранимой процедуры
        cursor.execute(query, (start_date_str, end_date_str))
        
        # Проверка наличия результатов
        if not cursor.description:
            logger.warning("Хранимая процедура не вернула данные")
            return None
        
        # Получение имен колонок из описания курсора
        columns = [column[0] for column in cursor.description]
        
        # Извлечение всех строк результата
        rows = cursor.fetchall()
        
        # Проверка на пустой результат
        if not rows:
            logger.info(f"Данные за период с {start_date_str} по {end_date_str} отсутствуют")
            return pd.DataFrame(columns=columns)
        
        # Преобразование в pandas DataFrame
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Обработка NULL значений
        logger.info(f"Успешно загружено {len(df)} записей за период с {start_date_str} по {end_date_str}")
        logger.debug(f"Колонки: {', '.join(columns)}")
        
        # Добавление календарных фичей если запрошено
        if add_features:
            df = add_calendar_features(df)
        
        # Добавление фичей истории заказов если запрошено
        if add_history_features:
            df = add_order_history_features(df, start_date_str, end_date_str)
        
        # Добавление фичей продаж если запрошено
        if add_sales_features_flag:
            df = add_sales_features(df, start_date_str, end_date_str)
        
        return df
        
    except pyodbc.Error as db_error:
        logger.error(f"Ошибка базы данных: {db_error}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении данных: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Соединение с базой данных закрыто")


def main():
    """
    Основная функция для демонстрации использования модуля.
    Загружает данные за последние 30 дней и выводит статистику.
    """
    import argparse
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Выгрузка сырых данных из базы данных optimum_lipetsk'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 30 дней назад)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        help='Конечная дата в формате YYYY-MM-DD (по умолчанию: сегодня)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help='Путь для сохранения результата в CSV (опционально)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Включить подробный режим логгирования'
    )
    parser.add_argument(
        '--add-features', 
        action='store_true', 
        help='Добавить календарные фичи для ML модели'
    )
    parser.add_argument(
        '--add-history-features', 
        action='store_true', 
        help='Добавить фичи истории заказов (Days_Since_Last_Order_Category, Days_Since_Last_Order_Total, Average_Interval_Category)'
    )
    parser.add_argument(
        '--add-sales-features', 
        action='store_true', 
        help='Добавить фичи продаж (Prev_Order_Amount_Category, SMA_3/7/30_Category, Momentum_Category, StdDev_Category)'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логгирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Определение дат
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else datetime.now().date()
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else end_date - timedelta(weeks=52)
    
    logger.info(f"Загрузка данных с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        df = fetch_raw_data(start_date, end_date, add_features=args.add_features, 
                           add_history_features=args.add_history_features, 
                           add_sales_features_flag=args.add_sales_features)
        
        if df is not None and len(df) > 0:
            # Вывод статистики
            logger.info("\n=== Статистика данных ===")
            logger.info(f"Всего записей: {len(df)}")
            logger.info(f"\nПервые 5 строк данных:\n{df.head()}")
            logger.info(f"\nИнформация о типах данных:")
            logger.info(f"Колонки: {list(df.columns)}")
            logger.info(f"\nСтатистика числовых колонок:\n{df.describe()}")
            
            # Сохранение в файл если указан путь
            if args.output:
                df.to_csv(args.output, index=False)
                logger.info(f"\nДанные сохранены в файл: {args.output}")
            
            return df
        else:
            logger.warning("Данные не получены или пустой результат")
            return None
        
    except Exception as e:
        logger.error(f"Критическая ошибка выполнения: {e}")
        raise


if __name__ == "__main__":
    main()
