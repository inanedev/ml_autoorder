import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional
import logging
import os
from dotenv import load_dotenv
import numpy as np
from catboost import CatBoostRegressor, Pool, cv
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from functools import lru_cache
import hashlib
import warnings
warnings.filterwarnings('ignore')

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
    for col_name in ['date', 'Date', 'DATE', 'VisitDate', 'visitdate']:
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
    
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: использует векторизованные операции pandas вместо циклов.
    
    Аргументы:
        df: DataFrame с данными продаж (должен содержать историю за 6 месяцев)
        start_date: Начальная дата периода выгрузки (строка YYYY-MM-DD)
        end_date: Конечная дата периода выгрузки (строка YYYY-MM-DD)
        
    Возвращает:
        DataFrame с добавленными фичами:
        - Days_Since_Last_Order_Category: дней назад точка брала эту категорию
        - Days_Since_Last_Order_Total: дней назад был любой заказ от точки
        - Average_Interval_Category: средний интервал между закупками категории
        - Days_Until_Next_Visit: дней до следующего визита точки (для последнего визита 
          используется значение из визита недельной давности, или 7 если такого визита нет)
        
    Примечание:
        Все расчеты производятся только на основе предоставленного DataFrame.
        Для корректного расчета рекомендуется передавать данные за 6 месяцев.
        Ожидает наличие колонок: 'date', 'outlet_id' (или 'OutletID'), 'category_id' (или 'CategoryID')
    """
    if df.empty:
        return df
    
    # Определение имен колонок
    date_col = None
    for col_name in ['date', 'Date', 'DATE', 'VisitDate', 'visitdate']:
        if col_name in df.columns:
            date_col = col_name
            break
    
    outlet_col = None
    for col_name in ['outlet_id', 'OutletID', 'OUTLET_ID', 'outletId', 'PointID', 'pointid']:
        if col_name in df.columns:
            outlet_col = col_name
            break
    
    category_col = None
    for col_name in ['category_id', 'CategoryID', 'CATEGORY_ID', 'categoryId', 'Category_Id']:
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
    
    # Инициализация колонок
    df_result['Days_Since_Last_Order_Category'] = np.nan
    df_result['Days_Since_Last_Order_Total'] = np.nan
    df_result['Average_Interval_Category'] = np.nan
    df_result['Days_Until_Next_Visit'] = np.nan
    
    logger.info(f"Расчет фичей истории заказов (векторизованный метод)...")
    
    # Сортировка по точке и дате для корректного расчета diff
    df_result = df_result.sort_values([outlet_col, date_col]).reset_index(drop=True)
    
    # ========== 1. Days_Since_Last_Order_Total (векторизованно) ==========
    # Используем groupby + shift для получения предыдущей даты заказа
    df_result['prev_order_date'] = df_result.groupby(outlet_col)[date_col].shift(1)
    df_result['Days_Since_Last_Order_Total'] = (
        df_result[date_col] - df_result['prev_order_date']
    ).dt.days
    
    # ========== 2. Days_Until_Next_Visit (векторизованно) ==========
    # Используем groupby + shift(-1) для получения следующей даты заказа
    df_result['next_order_date'] = df_result.groupby(outlet_col)[date_col].shift(-1)
    df_result['Days_Until_Next_Visit'] = (
        df_result['next_order_date'] - df_result[date_col]
    ).dt.days
    
    # Для последней записи в каждой точке заполняем значением 7
    last_visit_mask = df_result['Days_Until_Next_Visit'].isna()
    df_result.loc[last_visit_mask, 'Days_Until_Next_Visit'] = 7
    
    # Удаляем временные колонки
    df_result.drop(columns=['prev_order_date', 'next_order_date'], inplace=True)
    
    # ========== 3. Days_Since_Last_Order_Category (векторизованно) ==========
    if category_col is not None:
        # Группируем по точке + категория
        df_result['prev_category_date'] = df_result.groupby(
            [outlet_col, category_col]
        )[date_col].shift(1)
        
        df_result['Days_Since_Last_Order_Category'] = (
            df_result[date_col] - df_result['prev_category_date']
        ).dt.days
        
        # Удаляем временную колонку
        df_result.drop(columns=['prev_category_date'], inplace=True)
    
    # ========== 4. Average_Interval_Category (векторизованно) ==========
    if category_col is not None:
        # Рассчитываем интервалы между заказами внутри каждой группы точка-категория
        df_result['category_interval'] = df_result.groupby(
            [outlet_col, category_col]
        )[date_col].diff().dt.days
        
        # Средний интервал по каждой группе
        avg_intervals = df_result.groupby(
            [outlet_col, category_col], as_index=False
        )['category_interval'].mean().rename(columns={'category_interval': 'avg_interval'})
        
        # Merge среднего интервала обратно в основной DataFrame
        df_result = df_result.merge(
            avg_intervals,
            on=[outlet_col, category_col],
            how='left'
        )
        df_result['Average_Interval_Category'] = df_result['avg_interval']
        df_result.drop(columns=['avg_interval', 'category_interval'], inplace=True)
    
    logger.info(f"Добавлены фичи истории заказов: {len(df_result.columns)} колонок всего")
    
    return df_result


def add_sales_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Добавляет фичи продаж для подготовки модели машинного обучения.
    
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: использует векторизованные операции pandas вместо циклов.
    Время выполнения сокращается с часов/минут до секунд.
    
    Аргументы:
        df: DataFrame с данными продаж (должен содержать историю за 6 месяцев)
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
        Все расчеты производятся только на основе предоставленного DataFrame.
        Для корректного расчета рекомендуется передавать данные за 6 месяцев.
        Ожидает наличие колонок: 'date', 'outlet_id', 'category_id', 'sales_amount' (или аналог)
    """
    if df.empty:
        return df
    
    # Определение имен колонок
    date_col = None
    for col_name in ['date', 'Date', 'DATE', 'VisitDate', 'visitdate']:
        if col_name in df.columns:
            date_col = col_name
            break
    
    outlet_col = None
    for col_name in ['outlet_id', 'OutletID', 'OUTLET_ID', 'outletId', 'PointID', 'pointid']:
        if col_name in df.columns:
            outlet_col = col_name
            break
    
    category_col = None
    for col_name in ['category_id', 'CategoryID', 'CATEGORY_ID', 'categoryId', 'Category_Id']:
        if col_name in df.columns:
            category_col = col_name
            break
    
    amount_col = None
    for col_name in ['sales_amount', 'SalesAmount', 'SALES_AMOUNT', 'amount', 'Amount', 'sum', 'Sum', 'SumRoubles']:
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
    
    # Преобразование даты и суммы
    df_result[date_col] = pd.to_datetime(df_result[date_col])
    df_result[amount_col] = pd.to_numeric(df_result[amount_col], errors='coerce')
    
    # Инициализация колонок
    df_result['Prev_Order_Amount_Category'] = np.nan
    df_result['SMA_3_Category'] = np.nan
    df_result['SMA_7_Category'] = np.nan
    df_result['SMA_30_Category'] = np.nan
    df_result['Momentum_Category'] = np.nan
    df_result['StdDev_Category'] = np.nan
    
    # Сортировка по точке, категории и дате для корректного расчета
    sort_cols = [outlet_col]
    if category_col is not None:
        sort_cols.append(category_col)
    sort_cols.append(date_col)
    
    df_result = df_result.sort_values(sort_cols).reset_index(drop=True)
    
    logger.info(f"Расчет фичей продаж (векторизованный метод)...")
    
    # Определение группы для группировки
    group_cols = [outlet_col]
    if category_col is not None:
        group_cols.append(category_col)
    
    # ========== 1. Prev_Order_Amount_Category (векторизованно) ==========
    # Используем groupby + shift для получения предыдущей суммы заказа
    df_result['Prev_Order_Amount_Category'] = df_result.groupby(group_cols)[amount_col].shift(1)
    
    # ========== 2-6. Rolling statistics (векторизованно) ==========
    # Сначала агрегируем данные по дате (сумма продаж по каждой дате в группе)
    daily_agg = df_result.groupby(group_cols + [date_col])[amount_col].sum().reset_index()
    daily_agg = daily_agg.sort_values(sort_cols)
    
    logger.info(f"Агрегировано {len(daily_agg)} записей для rolling расчетов")
    
    # Для rolling окон используем count-based window (количество наблюдений, а не календарных дней)
    # Это быстрее и работает с неравномерными данными
    
    # Устанавливаем дату как индекс для rolling операций
    daily_agg = daily_agg.set_index(date_col)
    
    # Группируем по outlet/category для rolling операций
    grouped = daily_agg.groupby(group_cols)
    
    # ========== SMA_3 (последние 3 наблюдения) ==========
    sma_3 = grouped[amount_col].rolling(window=3, min_periods=1).mean()
    daily_agg['SMA_3_Category'] = sma_3.reset_index(level=[0, 1], drop=True).values
    
    # ========== SMA_7 (последние 7 наблюдений) ==========
    sma_7 = grouped[amount_col].rolling(window=7, min_periods=1).mean()
    daily_agg['SMA_7_Category'] = sma_7.reset_index(level=[0, 1], drop=True).values
    
    # ========== SMA_30 (последние 30 наблюдений) ==========
    sma_30 = grouped[amount_col].rolling(window=30, min_periods=1).mean()
    daily_agg['SMA_30_Category'] = sma_30.reset_index(level=[0, 1], drop=True).values
    
    # ========== StdDev_Category (последние 30 наблюдений) ==========
    std_dev = grouped[amount_col].rolling(window=30, min_periods=2).std()
    daily_agg['StdDev_Category'] = std_dev.reset_index(level=[0, 1], drop=True).values
    
    # ========== Momentum_Category ==========
    # Отношение SMA_7 к SMA_30
    daily_agg['Momentum_Category'] = daily_agg['SMA_7_Category'] / daily_agg['SMA_30_Category']
    daily_agg.loc[daily_agg['SMA_30_Category'] == 0, 'Momentum_Category'] = np.nan
    
    # Сбрасываем индекс
    daily_agg = daily_agg.reset_index()
    
    # Merge рассчитанных фичей обратно в основной DataFrame
    merge_cols = group_cols + [date_col]
    feature_cols = ['SMA_3_Category', 'SMA_7_Category', 'SMA_30_Category', 'Momentum_Category', 'StdDev_Category']
    
    df_result = df_result.merge(
        daily_agg[merge_cols + feature_cols],
        on=merge_cols,
        how='left'
    )
    
    logger.info(f"Добавлены фичи продаж: {len(df_result.columns)} колонок всего")
    
    return df_result

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
                              Days_Since_Last_Order_Total, Average_Interval_Category, Days_Until_Next_Visit)
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
    Загружает данные за последние 30 дней, обучает модель CatBoost,
    прогнозирует суммы для тестовых данных и выкладывает результат в SNS_ML_Predictions.
    """
    import argparse
    
    # Проверка свободного места перед установкой библиотек
    def check_disk_space(min_gb=5):
        """Проверяет свободное место на диске"""
        total, used, free = shutil.disk_usage('/')
        free_gb = free / (1024 ** 3)
        logger.info(f"Свободно места на диске: {free_gb:.2f} GB")
        if free_gb < min_gb:
            logger.warning(f"Недостаточно свободного места (требуется минимум {min_gb} GB)")
            return False
        return True
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Выгрузка сырых данных из базы данных optimum_lipetsk, обучение модели и прогнозирование'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 365 дней назад)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        help='Конечная дата в формате YYYY-MM-DD (по умолчанию: вчера)'
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
        '--skip-install-check',
        action='store_true',
        help='Пропустить проверку свободного места перед установкой библиотек'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логгирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Проверка свободного места (если не пропущено)
    if not args.skip_install_check:
        if not check_disk_space(min_gb=5):
            logger.error("Недостаточно свободного места для продолжения работы")
            return None
    
    # Определение дат для обучающей выборки
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else (datetime.now().date() - timedelta(days=1))
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else end_date - timedelta(days=365)
    
    # Сегодняшняя дата для тестовых данных
    today_date = datetime.now().date()
    
    logger.info(f"Загрузка обучающих данных с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        # Шаг 1: Загрузка обучающих данных со всеми фичами
        logger.info("Загрузка данных для обучения модели...")
        df_train = fetch_raw_data(
            start_date, end_date, 
            add_features=True, 
            add_history_features=True, 
            add_sales_features_flag=True
        )
        
        if df_train is None or len(df_train) == 0:
            logger.error("Не удалось загрузить данные для обучения модели")
            return None
        
        logger.info(f"Загружено {len(df_train)} записей для обучения")
        
        # Шаг 2: Подготовка данных для обучения CatBoost
        # Определение целевой переменной и признаков
        target_col = 'SumRoubles'
        
        # Колонки которые не являются признаками
        exclude_cols = [target_col, 'VisitDate', 'date', 'Date', 'DATE']
        
        # Определение колонок-признаков
        feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        
        logger.info(f"Используется {len(feature_cols)} признаков для обучения")
        logger.info(f"Признаки: {feature_cols}")
        
        # Проверка наличия целевой переменной
        if target_col not in df_train.columns:
            logger.error(f"Целевая переменная '{target_col}' не найдена в данных")
            return None
        
        # Удаление строк с NaN в целевой переменной
        df_train_clean = df_train.dropna(subset=[target_col])
        
        if len(df_train_clean) == 0:
            logger.error("После удаления NaN не осталось данных для обучения")
            return None
        
        # Определение категориальных признаков
        categorical_features = []
        for col in df_train_clean.columns:
            if col == target_col or col.lower() in ['visitdate', 'date']:
                continue
            if df_train_clean[col].dtype == 'object' or df_train_clean[col].dtype == 'bool':
                categorical_features.append(col)
        
        logger.info(f"Категориальные признаки: {categorical_features}")
        
        # Заполнение пропусков в категориальных признаках значением 'Unknown'
        for col in categorical_features:
            df_train_clean[col] = df_train_clean[col].fillna('Unknown')
        
        # Заполнение пропусков в числовых признаках нулем или медианой
        numeric_features = [col for col in df_train_clean.columns if col not in [target_col] + categorical_features and col.lower() not in ['visitdate', 'date']]
        for col in numeric_features:
            # Принудительное преобразование к числовому типу, заменяя нечисловые значения на NaN
            df_train_clean[col] = pd.to_numeric(df_train_clean[col], errors='coerce')
            # Замена бесконечных значений на NaN
            df_train_clean[col] = df_train_clean[col].replace([np.inf, -np.inf], np.nan)
            if df_train_clean[col].isna().any():
                if df_train_clean[col].dtype in ['float64', 'int64']:
                    df_train_clean[col] = df_train_clean[col].fillna(0)
        
        # Преобразование target_col к числовому типу для предотвращения ошибок с decimal.Decimal
        df_train_clean[target_col] = pd.to_numeric(df_train_clean[target_col], errors='coerce')
        # Замена бесконечных значений в целевой переменной на NaN и последующее удаление строк
        df_train_clean[target_col] = df_train_clean[target_col].replace([np.inf, -np.inf], np.nan)
        df_train_clean = df_train_clean.dropna(subset=[target_col])
        
        # ==================== 1. ОБРАБОТКА ВЫБРОСОВ В ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ====================
        logger.info("Обработка выбросов в целевой переменной...")
        
        # Работаем напрямую с оригинальной целевой переменной (без логарифмирования!)
        # Логарифмирование приводило к систематическому занижению прогнозов
        
        # Расчет границ для удаления выбросов (метод IQR) на оригинальных данных
        Q1 = df_train_clean[target_col].quantile(0.25)
        Q3 = df_train_clean[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Фильтрация выбросов
        outliers_mask = (df_train_clean[target_col] < lower_bound) | (df_train_clean[target_col] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            logger.info(f"Удалено {n_outliers} выбросов ({100*n_outliers/len(df_train_clean):.2f}% данных)")
            df_train_clean = df_train_clean[~outliers_mask]
        
        X_train = df_train_clean[feature_cols]
        y_train = df_train_clean[target_col]  # Используем оригинальную целевую переменную (БЕЗ логарифма!)
        # y_train_original больше не нужен, используем напрямую y_train
        
        # ==================== 2. КРОСС-ВАЛИДАЦИЯ С ВРЕМЕННЫМИ РЯДАМИ И АНСАМБЛИРОВАНИЕ ====================
        logger.info("Настройка кросс-валидации с временными рядами и ансамблирования...")
        
        # Параметры для CatBoost с MAE вместо RMSE
        base_params = {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.1,
            'loss_function': 'MAE',  # MAE более устойчив к выбросам
            'eval_metric': 'MAE',
            'verbose': 100,
            'cat_features': categorical_features if categorical_features else None,
            'random_seed': 42,
            'early_stopping_rounds': 50
        }
        
        # Разные конфигурации для ансамбля (разная глубина и learning rate)
        ensemble_configs = [
            {**base_params, 'depth': 6, 'learning_rate': 0.1, 'random_seed': 42},
            {**base_params, 'depth': 8, 'learning_rate': 0.05, 'random_seed': 123},
            {**base_params, 'depth': 4, 'learning_rate': 0.15, 'random_seed': 456},
            {**base_params, 'depth': 7, 'learning_rate': 0.08, 'random_seed': 789},
            {**base_params, 'depth': 5, 'learning_rate': 0.12, 'random_seed': 101112},
        ]
        
        # ==================== 2.1 КРОСС-ВАЛИДАЦИЯ С ВРЕМЕННЫМИ РЯДАМИ ====================
        # Используем TimeSeriesSplit для корректной валидации на временных данных
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores_mae = []
        cv_scores_rmse = []
        cv_scores_r2 = []
        
        logger.info(f"Проведение кросс-валидации с временными рядами ({n_splits} folds)...")
        logger.info("TimeSeriesSplit гарантирует, что тестовые данные всегда идут после обучающих")
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Создаем Pool для CatBoost
            train_pool = Pool(X_fold_train, y_fold_train, cat_features=categorical_features if categorical_features else None)
            val_pool = Pool(X_fold_val, y_fold_val, cat_features=categorical_features if categorical_features else None)
            
            model_cv = CatBoostRegressor(**base_params)
            model_cv.fit(train_pool, eval_set=val_pool, verbose=False)
            
            # Предсказания на валидации
            y_pred_fold = model_cv.predict(val_pool)
            
            # Метрики (в логарифмическом пространстве)
            mae_fold = mean_absolute_error(y_fold_val, y_pred_fold)
            rmse_fold = np.sqrt(mean_squared_error(y_fold_val, y_pred_fold))
            r2_fold = r2_score(y_fold_val, y_pred_fold)
            
            cv_scores_mae.append(mae_fold)
            cv_scores_rmse.append(rmse_fold)
            cv_scores_r2.append(r2_fold)
            
            logger.info(f"Fold {fold_idx+1}: MAE={mae_fold:.4f}, RMSE={rmse_fold:.4f}, R²={r2_fold:.4f}")
        
        logger.info(f"Кросс-валидация с временными рядами завершена. Средние метрики:")
        logger.info(f"  MAE: {np.mean(cv_scores_mae):.4f} (+/- {np.std(cv_scores_mae):.4f})")
        logger.info(f"  RMSE: {np.mean(cv_scores_rmse):.4f} (+/- {np.std(cv_scores_rmse):.4f})")
        logger.info(f"  R²: {np.mean(cv_scores_r2):.4f} (+/- {np.std(cv_scores_r2):.4f})")
        
        # ==================== 2.2 ДОПОЛНИТЕЛЬНАЯ КРОСС-ВАЛИДАЦИЯ K-Fold ДЛЯ СРАВНЕНИЯ ====================
        logger.info("\nДополнительная K-Fold кросс-валидация (shuffle=True) для сравнения...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        kf_scores_mae = []
        kf_scores_rmse = []
        kf_scores_r2 = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            train_pool = Pool(X_fold_train, y_fold_train, cat_features=categorical_features if categorical_features else None)
            val_pool = Pool(X_fold_val, y_fold_val, cat_features=categorical_features if categorical_features else None)
            
            model_kf = CatBoostRegressor(**base_params)
            model_kf.fit(train_pool, eval_set=val_pool, verbose=False)
            
            y_pred_fold = model_kf.predict(val_pool)
            
            mae_fold = mean_absolute_error(y_fold_val, y_pred_fold)
            rmse_fold = np.sqrt(mean_squared_error(y_fold_val, y_pred_fold))
            r2_fold = r2_score(y_fold_val, y_pred_fold)
            
            kf_scores_mae.append(mae_fold)
            kf_scores_rmse.append(rmse_fold)
            kf_scores_r2.append(r2_fold)
        
        logger.info(f"K-Fold средние метрики:")
        logger.info(f"  MAE: {np.mean(kf_scores_mae):.4f} (+/- {np.std(kf_scores_mae):.4f})")
        logger.info(f"  RMSE: {np.mean(kf_scores_rmse):.4f} (+/- {np.std(kf_scores_rmse):.4f})")
        logger.info(f"  R²: {np.mean(kf_scores_r2):.4f} (+/- {np.std(kf_scores_r2):.4f})")
        
        # Выбираем лучшую схему кросс-валидации по R²
        ts_r2_mean = np.mean(cv_scores_r2)
        kf_r2_mean = np.mean(kf_scores_r2)
        
        if ts_r2_mean >= kf_r2_mean:
            logger.info(f"\n✓ TimeSeriesSplit показал лучший результат (R²={ts_r2_mean:.4f}), используем его для финальной оценки")
        else:
            logger.info(f"\n✓ K-Fold показал лучший результат (R²={kf_r2_mean:.4f}), но для временных рядов рекомендуем TimeSeriesSplit")
        
        # ==================== 3. ОБУЧЕНИЕ АНСАМБЛЯ МОДЕЛЕЙ ====================
        logger.info("Обучение ансамбля моделей...")
        
        ensemble_models = []
        
        for idx, config in enumerate(ensemble_configs):
            logger.info(f"Обучение модели {idx+1}/{len(ensemble_configs)}...")
            
            # Создаем Pool с использованием всей обучающей выборки
            train_pool = Pool(X_train, y_train, cat_features=categorical_features if categorical_features else None)
            
            model = CatBoostRegressor(**config)
            model.fit(train_pool, verbose=False)
            
            ensemble_models.append(model)
            logger.info(f"Модель {idx+1} обучена")
        
        logger.info(f"Ансамбль из {len(ensemble_models)} моделей успешно обучен!")
        
        # Функция для предсказания ансамблем
        def ensemble_predict(X):
            """Усреднение предсказаний всех моделей ансамбля"""
            predictions = [model.predict(X) for model in ensemble_models]
            return np.mean(predictions, axis=0)
        
        # Оценка качества на тренировочных данных
        train_predictions = ensemble_predict(X_train)
        
        # Метрики в оригинальном пространстве (модель обучена на MAE с оригинальными значениями)
        mae_train = mean_absolute_error(y_train, train_predictions)
        rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
        r2_train = r2_score(y_train, train_predictions)
        
        logger.info(f"Метрики на тренировочных данных (оригинальное пространство):")
        logger.info(f"  MAE: {mae_train:.4f}")
        logger.info(f"  RMSE: {rmse_train:.4f}")
        logger.info(f"  R²: {r2_train:.4f}")
        
        # Шаг 4: Загрузка тестовых данных с сегодняшней датой
        logger.info(f"Загрузка тестовых данных для даты {today_date.strftime('%Y-%m-%d')}...")
        
        test_query = """
        EXEC dbo.SNS_ML_Get_Test_Data @TargetDate = ?
        """
        
        conn = get_connection()
        df_test = pd.read_sql(test_query, conn, params=[today_date.strftime('%Y-%m-%d')])
        conn.close()
        
        if df_test is None or len(df_test) == 0:
            logger.error("Не удалось загрузить тестовые данные")
            return None
        
        logger.info(f"Загружено {len(df_test)} записей для прогнозирования")
        
        # Шаг 5: Прогнозирование для тестовых данных
        logger.info("Выполнение прогнозирования...")
        
        # Подготовка тестовых данных (те же признаки)
        # Добавление отсутствующих признаков со значением 0 для совместимости с моделью
        missing_features = set(feature_cols) - set(df_test.columns)
        if missing_features:
            logger.warning(f"Отсутствуют следующие признаки в тестовых данных: {missing_features}")
            logger.info(f"Добавление {len(missing_features)} отсутствующих признаков со значением 0")
            for col in missing_features:
                df_test[col] = 0
        
        # Проверка наличия всех признаков
        available_feature_cols = [col for col in feature_cols if col in df_test.columns]
        if len(available_feature_cols) != len(feature_cols):
            logger.error(f"Не удалось добавить все признаки. Ожидается {len(feature_cols)}, доступно {len(available_feature_cols)}")
            raise ValueError("Несоответствие признаков между обучением и прогнозированием")
        
        X_test = df_test[feature_cols].copy()
        
        # Заполнение пропусков в категориальных признаках значением 'Unknown'
        for col in categorical_features:
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna('Unknown')
        
        # Заполнение пропусков в числовых признаках нулем - с предварительным преобразованием к numeric
        for col in numeric_features:
            if col in X_test.columns:
                # Принудительное преобразование к числовому типу, заменяя нечисловые значения на NaN
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                # Замена бесконечных значений на NaN
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
                if X_test[col].isna().any():
                    X_test[col] = X_test[col].fillna(0)
        
        # Предсказание с использованием ансамбля моделей
        predictions = ensemble_predict(X_test)
        
        # Обработка бесконечных и слишком больших значений
        # Замена inf на максимальное конечное значение
        max_finite = np.finfo(np.float64).max
        predictions = np.clip(predictions, -1, max_finite)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=max_finite, neginf=-1.0)

        # Дополнительная проверка и замена выбросов на разумные значения
        # Если есть значения > 1e100, заменяем их на 99-й перцентиль
        if np.any(predictions > 1e100):
            finite_predictions = predictions[np.isfinite(predictions) & (predictions < 1e100)]
            if len(finite_predictions) > 0:
                percentile_99 = np.percentile(finite_predictions, 99)
                logger.warning(f"Обнаружены экстремальные значения (>1e100). Замена на 99-й перцентиль: {percentile_99:.2f}")
                predictions = np.where(predictions > 1e100, percentile_99, predictions)
            else:
                logger.warning("Обнаружены экстремальные значения (>1e100). Замена на медиану")
                predictions = np.where(predictions > 1e100, np.median(predictions[predictions < 1e100]), predictions)
        
        # Дополнительная защита от переполнения: ограничение сверху разумным значением
        # На основе 99.9-го перцентиля для предотвращения проблем при расчете std
        finite_mask = np.isfinite(predictions)
        if np.sum(finite_mask) > 0:
            percentile_999 = np.percentile(predictions[finite_mask], 99.9)
            upper_limit = max(percentile_999, 1e6)  # Не меньше 1 миллиона
            if np.max(predictions) > upper_limit:
                logger.warning(f"Ограничение сверху значений на уровне {upper_limit:.2f} (99.9-й перцентиль)")
                predictions = np.clip(predictions, None, upper_limit)

        
        # Добавление предсказаний в DataFrame
        df_test['Predicted_Category_Sum'] = predictions
        
        logger.info(f"Прогнозы выполнены. Статистика предсказаний:")
        logger.info(f"  Мин: {predictions.min():.2f}")
        logger.info(f"  Макс: {predictions.max():.2f}")
        logger.info(f"  Среднее: {predictions.mean():.2f}")
        
        # Расчет Prediction_Confidence на основе исторической точности модели
        logger.info("Расчет метрик уверенности предсказаний...")
        
        # Метрики уже рассчитаны выше (mae_train, rmse_train, r2_train)
        # Используем их напрямую без повторного расчета
        
        logger.info(f"Метрики на тренировочных данных (оригинальное пространство):")
        logger.info(f"  MAE: {mae_train:.2f}")
        logger.info(f"  RMSE: {rmse_train:.2f}")
        logger.info(f"  R²: {r2_train:.4f}")
        
        # Расчет confidence как обратной величины от относительной ошибки
        # Confidence будет в диапазоне [0, 1], где 1 - максимальная уверенность
        mean_actual = y_train.mean()
        relative_error = mae_train / mean_actual if mean_actual > 0 else 1
        
        # Базовая уверенность модели (чем меньше ошибка, тем выше уверенность)
        base_confidence = max(0, min(1, 1 - relative_error))
        
        logger.info(f"  Базовая уверенность модели: {base_confidence:.3f}")
        
        # Для каждого предсказания рассчитываем индивидуальную уверенность
        # на основе расстояния от среднего значения предсказаний
        pred_std = predictions.std()
        pred_mean = predictions.mean()
        
        # Защита от бесконечных/NaN значений при расчете std и mean
        if not np.isfinite(pred_std) or not np.isfinite(pred_mean):
            logger.warning("Обнаружены некорректные значения pred_std или pred_mean. Используем безопасные значения.")
            finite_predictions = predictions[np.isfinite(predictions)]
            if len(finite_predictions) > 0:
                pred_std = finite_predictions.std()
                pred_mean = finite_predictions.mean()
            else:
                pred_std = 1.0
                pred_mean = 0.0
        
        # Confidence уменьшается для выбросов (далеких от среднего)
        if pred_std > 0 and np.isfinite(pred_std) and np.isfinite(pred_mean):
            z_scores = np.abs((predictions - pred_mean) / pred_std)
            # Замена бесконечных z_scores на большие конечные значения
            z_scores = np.nan_to_num(z_scores, nan=10.0, posinf=10.0, neginf=10.0)
            # Преобразуем z-score в confidence (чем больше отклонение, тем меньше уверенность)
            individual_confidence = np.exp(-0.5 * (z_scores ** 2))
        else:
            individual_confidence = np.ones(len(predictions))
        
        # Итоговая уверенность = базовая уверенность * индивидуальная уверенность
        df_test['Prediction_Confidence'] = base_confidence * individual_confidence
        
        logger.info(f"  Min Confidence: {df_test['Prediction_Confidence'].min():.3f}")
        logger.info(f"  Max Confidence: {df_test['Prediction_Confidence'].max():.3f}")
        logger.info(f"  Mean Confidence: {df_test['Prediction_Confidence'].mean():.3f}")
        
        # Шаг 6: Выгрузка результатов в таблицу SNS_ML_Predictions
        logger.info("Загрузка результатов в таблицу SNS_ML_Predictions...")
        
        # Подготовка данных для вставки
        columns_to_insert = [
            'VisitDate', 'PointID', 'CategoryID', 'BranchID', 'PointClass', 'PointType',
            'Lat', 'Lon', 'MicroRegionID',
            'day_of_week', 'is_weekend', 'is_monday', 'is_friday', 'is_saturday', 'is_sunday',
            'is_holiday', 'is_pre_holiday', 'is_post_holiday',
            'month', 'quarter', 'week_of_year', 'is_month_start', 'is_month_end',
            'day_of_month', 'day_of_year', 'days_to_holiday', 'days_from_holiday',
            'Days_Since_Last_Order_Category', 'Days_Since_Last_Order_Total', 'Average_Interval_Category',
            'Days_Until_Next_Visit',
            'Prev_Order_Amount_Category', 'SMA_3_Category', 'SMA_7_Category', 'SMA_30_Category',
            'Momentum_Category', 'StdDev_Category',
            'Predicted_Category_Sum', 'Prediction_Confidence'
        ]
        
        # Фильтрация только существующих колонок
        available_columns = [col for col in columns_to_insert if col in df_test.columns]
        df_to_insert = df_test[available_columns].copy()
        
        # Добавление служебных полей
        df_to_insert['CreatedAt'] = datetime.now()
        df_to_insert['ModelVersion'] = 'catboost_v1'
        
        # Вставка данных в базу
        conn = get_connection()
        cursor = conn.cursor()
        
        # Включаем режим быстрой пакетной вставки для pyodbc
        cursor.fast_executemany = True
        
        # Формирование SQL запроса для вставки
        insert_query = f"""
        INSERT INTO dbo.SNS_ML_Predictions (
            {', '.join(available_columns)}, CreatedAt, ModelVersion
        ) VALUES (
            {', '.join(['?' for _ in available_columns])}, ?, ?
        )
        """
        
        logger.info(f"Вставка {len(df_to_insert)} записей в SNS_ML_Predictions...")
        
        # Пакетная вставка - подготовка всех данных
        all_values = []
        for idx, row in df_to_insert.iterrows():
            values = [row[col] if pd.notna(row[col]) else None for col in available_columns]
            values.extend([row['CreatedAt'], row['ModelVersion']])
            all_values.append(values)
        
        # Единая пакетная вставка всех записей
        cursor.executemany(insert_query, all_values)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Успешно загружено {len(df_to_insert)} прогнозов в таблицу SNS_ML_Predictions")
        
        # Сохранение в файл если указан путь
        if args.output:
            df_test.to_csv(args.output, index=False)
            logger.info(f"Результаты сохранены в файл: {args.output}")
        
        return df_test
        
    except Exception as e:
        logger.error(f"Критическая ошибка выполнения: {e}")
        raise


if __name__ == "__main__":
    main()
