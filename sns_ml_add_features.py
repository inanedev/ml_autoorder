import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional
import logging
from sns_ml_fetch_data import fetch_raw_data, get_connection
import pyodbc
import os
from dotenv import load_dotenv
import holidays

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_russia_holidays(year: int) -> List[date]:
    """
    Возвращает список национальных праздников России для указанного года.
    Использует библиотеку holidays для получения точных дат с учётом переносов выходных.
    
    Args:
        year: Год, для которого нужно получить праздники
        
    Returns:
        Список дат праздников
    """
    ru_holidays = holidays.Russia(years=year)
    return sorted(list(ru_holidays.keys()))


def get_extended_russia_holidays(start_year: int, end_year: int) -> List[date]:
    """
    Возвращает расширенный список праздников России за период лет.
    Использует библиотеку holidays для получения точных дат с учётом переносов выходных.
    
    Args:
        start_year: Начальный год
        end_year: Конечный год
        
    Returns:
        Отсортированный список дат всех праздников
    """
    ru_holidays = holidays.Russia(years=range(start_year, end_year + 1))
    return sorted(list(ru_holidays.keys()))


def find_nearest_holidays(target_date: date, holidays: List[date]) -> Tuple[Optional[date], Optional[date]]:
    """
    Находит ближайший прошедший и ближайший будущий праздник относительно target_date.
    
    Args:
        target_date: Целевая дата
        holidays: Отсортированный список праздников
        
    Returns:
        Кортеж (прошедший_праздник, будущий_праздник)
        Если праздника нет в прошлом/будущем в пределах диапазона, возвращается None
    """
    past_holiday = None
    future_holiday = None
    
    for holiday in holidays:
        if holiday < target_date:
            past_holiday = holiday
        elif holiday > target_date and future_holiday is None:
            future_holiday = holiday
            break
    
    return past_holiday, future_holiday


def add_calendar_features(df: pd.DataFrame, visit_date_col: str = 'VisitDate') -> pd.DataFrame:
    """
    Добавляет календарные признаки к датафрейму на основе даты в указанной колонке.
    
    Добавляемые признаки:
    1. DayOfWeek - день недели (1-понедельник, 2-вторник, ..., 7-воскресенье)
    2. IsFriday - признак пятницы (1/0)
    3. IsMonday - признак понедельника (1/0)
    4. DaysToNextHoliday - количество дней до начала ближайшего национального праздника
    5. DaysSinceLastHoliday - количество дней с окончания ближайшего национального праздника
    6. IsPreHoliday - является ли день предпраздничным (до праздника 3 и менее дней)
    7. IsPostHoliday - является ли день постпраздничным (после праздника прошло 3 и менее дней)
    8. Quarter - номер квартала (1-4)
    9. Month - номер месяца (1-12)
    10. WeekOfYear - номер недели в году (1-53)
    11. DayOfMonth - день месяца (1-31)
    12. DayOfYear - день года (1-366)
    13. isEndOfMonth - бинарная фича: последние 3 дня месяца или первые 2 дня месяца, 
        если до конца недели <= 2 дней (пятница, суббота, воскресенье), 
        либо является последней субботой месяца
    14. DayOfYear_sin - синус дня года (циклический признак для сезонности)
    15. DayOfYear_cos - косинус дня года (циклический признак для сезонности)
    16. Month_sin - синус месяца (циклический признак для месячной сезонности)
    17. Month_cos - косинус месяца (циклический признак для месячной сезонности)
    18. DayOfWeek_sin - синус дня недели (циклический признак для недельной сезонности)
    19. DayOfWeek_cos - косинус дня недели (циклический признак для недельной сезонности)
    20. WeekOfYear_sin - синус недели года (циклический признак для годовой сезонности)
    21. WeekOfYear_cos - косинус недели года (циклический признак для годовой сезонности)
    22. Quarter_sin - синус квартала (циклический признак для квартальной сезонности)
    23. Quarter_cos - косинус квартала (циклический признак для квартальной сезонности)
    
    Циклические признаки (синус/косинус) кодируют циклическую природу времени,
    чтобы модель понимала близость значений на границах циклов (например, 
    воскресенье близко к понедельнику, декабрь близок к январю).
    
    Args:
        df: Исходный датафрейм
        visit_date_col: Название колонки с датой (по умолчанию 'VisitDate')
        
    Returns:
        Датафрейм с добавленными признаками
    """
    logger.info(f"Добавление календарных признаков на основе колонки '{visit_date_col}'")
    
    # Создаем копию датафрейма, чтобы не модифицировать исходный
    result_df = df.copy()
    
    # Преобразуем колонку с датой в формат datetime, если это еще не сделано
    if not pd.api.types.is_datetime64_any_dtype(result_df[visit_date_col]):
        result_df[visit_date_col] = pd.to_datetime(result_df[visit_date_col])
    
    # Извлекаем дату (без времени) для расчетов
    dates = result_df[visit_date_col].dt.date
    
    # Определяем диапазон лет для загрузки праздников
    min_year = dates.min().year - 1 if dates.min() else datetime.now().year - 1
    max_year = dates.max().year + 1 if dates.max() else datetime.now().year + 1
    
    # Загружаем праздники для расширенного диапазона
    logger.info(f"Загрузка праздников России за период {min_year}-{max_year}")
    holidays = get_extended_russia_holidays(min_year, max_year)
    holidays_set = set(holidays)
    
    logger.info(f"Всего загружено {len(holidays)} праздничных дней")
    
    # 2.1 День недели (1-понедельник, 2-вторник, ..., 7-воскресенье)
    # В pandas weekday(): понедельник=0, воскресенье=6, поэтому добавляем 1
    result_df['DayOfWeek'] = result_df[visit_date_col].dt.weekday + 1
    
    # 2.2 Признак пятницы
    result_df['IsFriday'] = (result_df['DayOfWeek'] == 5).astype(int)
    
    # 2.3 Признак понедельника
    result_df['IsMonday'] = (result_df['DayOfWeek'] == 1).astype(int)
    
    # 2.8 Номер квартала
    result_df['Quarter'] = result_df[visit_date_col].dt.quarter
    
    # 2.9 Номер месяца
    result_df['Month'] = result_df[visit_date_col].dt.month
    
    # 2.10 Номер недели в году
    result_df['WeekOfYear'] = result_df[visit_date_col].dt.isocalendar().week.astype(int)
    
    # 2.11 День месяца
    result_df['DayOfMonth'] = result_df[visit_date_col].dt.day
    
    # 2.12 День года
    result_df['DayOfYear'] = result_df[visit_date_col].dt.dayofyear
    
    # 2.13 isEndOfMonth - бинарная фича, которая показывает, что visitdate попадает 
    # на последние 3 дня месяца или на первые 2 дня месяца, но только если до конца 
    # недели меньше или равно 2 дня (конец недели - воскресенье, 7-й день),
    # либо является последней субботой месяца
    day_of_month = result_df['DayOfMonth']
    day_of_week = result_df['DayOfWeek']
    days_in_month = result_df[visit_date_col].dt.daysinmonth
    
    # Последние 3 дня месяца
    is_last_3_days = day_of_month >= days_in_month - 2
    
    # Первые 2 дня месяца И до конца недели <= 2 дней (DayOfWeek >= 5)
    is_first_2_days = day_of_month <= 2
    is_end_of_week = day_of_week >= 5  # пятница(5), суббота(6), воскресенье(7)
    
    # Последняя суббота месяца: DayOfWeek == 6 (суббота) И день + 7 > days_in_month
    is_last_saturday = (day_of_week == 6) & (day_of_month + 7 > days_in_month)
    
    result_df['isEndOfMonth'] = ((is_last_3_days) | (is_first_2_days & is_end_of_week) | is_last_saturday).astype(int)
    
    # Расчет признаков, связанных с праздниками
    days_to_next_holiday = []
    days_since_last_holiday = []
    is_pre_holiday = []
    is_post_holiday = []
    
    for d in dates:
        if d is None or pd.isna(d):
            days_to_next_holiday.append(np.nan)
            days_since_last_holiday.append(np.nan)
            is_pre_holiday.append(0)
            is_post_holiday.append(0)
            continue
        
        # Находим ближайший прошедший и будущий праздник
        past_holiday, future_holiday = find_nearest_holidays(d, holidays)
        
        # 2.4 Количество дней до начала ближайшего праздника
        if future_holiday:
            delta_to_next = (future_holiday - d).days
            days_to_next_holiday.append(delta_to_next)
        else:
            days_to_next_holiday.append(np.nan)
        
        # 2.5 Количество дней с окончания ближайшего праздника
        if past_holiday:
            delta_since_last = (d - past_holiday).days
            days_since_last_holiday.append(delta_since_last)
        else:
            days_since_last_holiday.append(np.nan)
        
        # 2.6 Предпраздничный день (до праздника 3 и менее дней)
        if future_holiday and (future_holiday - d).days <= 3:
            is_pre_holiday.append(1)
        else:
            is_pre_holiday.append(0)
        
        # 2.7 Постпраздничный день (после праздника прошло 3 и менее дней)
        if past_holiday and (d - past_holiday).days <= 3:
            is_post_holiday.append(1)
        else:
            is_post_holiday.append(0)
    
    result_df['DaysToNextHoliday'] = days_to_next_holiday
    result_df['DaysSinceLastHoliday'] = days_since_last_holiday
    result_df['IsPreHoliday'] = is_pre_holiday
    result_df['IsPostHoliday'] = is_post_holiday
    
    # 2.14-2.23 Циклические временные признаки (синус/косинус)
    # Эти признаки кодируют циклическую природу времени, чтобы модель понимала,
    # что например воскресенье близко к понедельнику, а декабрь близок к январю
    
    # День года (1-366) -> синус и косинус
    day_of_year = result_df['DayOfYear'].astype(float)
    result_df['DayOfYear_sin'] = np.sin(2 * np.pi * day_of_year / 366)
    result_df['DayOfYear_cos'] = np.cos(2 * np.pi * day_of_year / 366)
    
    # Месяц (1-12) -> синус и косинус
    month = result_df['Month'].astype(float)
    result_df['Month_sin'] = np.sin(2 * np.pi * month / 12)
    result_df['Month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # День недели (1-7) -> синус и косинус
    day_of_week = result_df['DayOfWeek'].astype(float)
    result_df['DayOfWeek_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    result_df['DayOfWeek_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Неделя года (1-53) -> синус и косинус (циклический признак для годовой сезонности)
    week_of_year = result_df['WeekOfYear'].astype(float)
    result_df['WeekOfYear_sin'] = np.sin(2 * np.pi * week_of_year / 53)
    result_df['WeekOfYear_cos'] = np.cos(2 * np.pi * week_of_year / 53)
    
    # Квартал (1-4) -> синус и косинус (циклический признак для квартальной сезонности)
    quarter = result_df['Quarter'].astype(float)
    result_df['Quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    result_df['Quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    logger.info(f"Успешно добавлено 23 календарных признаков")
    logger.info(f"Новые колонки: {['DayOfWeek', 'IsFriday', 'IsMonday', 'DaysToNextHoliday', 'DaysSinceLastHoliday', 'IsPreHoliday', 'IsPostHoliday', 'Quarter', 'Month', 'WeekOfYear', 'DayOfMonth', 'DayOfYear', 'isEndOfMonth', 'DayOfYear_sin', 'DayOfYear_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'WeekOfYear_sin', 'WeekOfYear_cos', 'Quarter_sin', 'Quarter_cos']}")
    
    return result_df


def add_visit_features(df: pd.DataFrame, visit_date_col: str = 'VisitDate', point_id_col: str = 'PointID') -> pd.DataFrame:
    """
    Добавляет признаки, связанные с посещениями точек.
    
    Добавляемые признаки:
    1. DaysLastVisit - количество дней с предыдущего VisitDate для PointID. 
       Если определить не удалось (первое посещение), то = 7
    2. DaysNextVisit - количество дней до следующего VisitDate для PointID. 
       Если определить не удалось (последнее посещение), то = 7
    3. isHolidayNextVisit - признак того, что следующий визит попадет на праздничный 
       или выходной день (суббота/воскресенье). 1 - если праздник/выходной, 0 - иначе.
       Для последнего посещения расчет ведется от даты визита + 7 дней (по умолчанию DaysNextVisit)
    
    Расчет ведется по уникальным датам для каждой точки, чтобы корректно обрабатывать
    случаи, когда для одной точки есть несколько записей с одинаковым VisitDate.
    
    Args:
        df: Исходный датафрейм
        visit_date_col: Название колонки с датой посещения (по умолчанию 'VisitDate')
        point_id_col: Название колонки с идентификатором точки (по умолчанию 'PointID')
        
    Returns:
        Датафрейм с добавленными признаками
    """
    logger.info(f"Добавление признаков посещений на основе колонок '{visit_date_col}' и '{point_id_col}'")
    
    # Создаем копию датафрейма, чтобы не модифицировать исходный
    result_df = df.copy()
    
    # Преобразуем колонку с датой в формат datetime, если это еще не сделано
    if not pd.api.types.is_datetime64_any_dtype(result_df[visit_date_col]):
        result_df[visit_date_col] = pd.to_datetime(result_df[visit_date_col])
    
    # Создаем датафрейм с уникальными комбинациями PointID и VisitDate
    unique_visits = result_df[[point_id_col, visit_date_col]].drop_duplicates().copy()
    
    # Сортируем по PointID и VisitDate для корректного расчета shift
    unique_visits = unique_visits.sort_values([point_id_col, visit_date_col]).reset_index(drop=True)
    
    # Рассчитываем предыдущую и следующую дату для каждой точки с помощью groupby + shift
    unique_visits['PrevVisitDate'] = unique_visits.groupby(point_id_col)[visit_date_col].shift(1)
    unique_visits['NextVisitDate'] = unique_visits.groupby(point_id_col)[visit_date_col].shift(-1)
    
    # Вычисляем разницу в днях (векторизированно)
    unique_visits['DaysLastVisit'] = (unique_visits[visit_date_col] - unique_visits['PrevVisitDate']).dt.days
    unique_visits['DaysNextVisit'] = (unique_visits['NextVisitDate'] - unique_visits[visit_date_col]).dt.days
    
    # Заполняем NaN значениями по умолчанию (7) для первого и последнего визита
    unique_visits['DaysLastVisit'] = unique_visits['DaysLastVisit'].fillna(7).astype(int)
    unique_visits['DaysNextVisit'] = unique_visits['DaysNextVisit'].fillna(7).astype(int)
    
    # 3. isHolidayNextVisit - проверяем, попадает ли следующий визит на праздник или выходной
    # Определяем диапазон лет для загрузки праздников
    min_year = unique_visits[visit_date_col].min().year - 1 if pd.notna(unique_visits[visit_date_col].min()) else datetime.now().year - 1
    max_year = unique_visits[visit_date_col].max().year + 1 if pd.notna(unique_visits[visit_date_col].max()) else datetime.now().year + 1
    
    # Загружаем праздники для расширенного диапазона
    ru_holidays = holidays.Russia(years=range(min_year, max_year + 1))
    holidays_set = set(ru_holidays.keys())
    
    def check_next_visit_holiday(row):
        """Проверяет, попадает ли следующий визит на праздник или выходной"""
        current_date = row[visit_date_col]
        days_next = row['DaysNextVisit']
        
        if pd.isna(current_date) or pd.isna(days_next):
            return 0
        
        # Вычисляем дату следующего визита
        next_visit_date = current_date + timedelta(days=days_next)
        next_visit_date_only = next_visit_date.date() if hasattr(next_visit_date, 'date') else next_visit_date
        
        # Проверяем: праздник ИЛИ суббота (5) ИЛИ воскресенье (6)
        # weekday(): понедельник=0, ..., суббота=5, воскресенье=6
        is_weekend = next_visit_date.weekday() >= 5  # 5=суббота, 6=воскресенье
        is_holiday = next_visit_date_only in holidays_set
        
        return 1 if (is_weekend or is_holiday) else 0
    
    # Применяем проверку для всех строк, включая последние визиты (где DaysNextVisit=7)
    unique_visits['isHolidayNextVisit'] = unique_visits.apply(check_next_visit_holiday, axis=1)
    
    # Удаляем вспомогательные колонки
    unique_visits = unique_visits.drop(columns=['PrevVisitDate', 'NextVisitDate'])
    
    # Merge с исходным датафреймом для присваивания значений всем строкам
    result_df = result_df.merge(unique_visits, on=[point_id_col, visit_date_col], how='left')
    
    logger.info(f"Успешно добавлено 3 признака посещений")
    logger.info(f"Новые колонки: ['DaysLastVisit', 'DaysNextVisit', 'isHolidayNextVisit']")
    
    return result_df


def add_category_sales_features(df: pd.DataFrame, visit_date_col: str = 'VisitDate', 
                                 point_id_col: str = 'PointID', 
                                 category_id_col: str = 'CategoryID') -> pd.DataFrame:
    """
    Добавляет признаки, связанные с продажами категорий в точках.
    
    Добавляемые признаки:
    1. DaysLastSalesCategory - количество дней с момента последней продажи категории в точку.
       Рассчитывается для комбинации VisitDate - PointID - CategoryID как дней с момента 
       последней продажи этой категории в эту точку.
       Если определить не удалось (первая продажа категории в точку), то = 7
    2. LastSalesCategory - сумма последней продажи категории в точку.
       Рассчитывается для комбинации VisitDate - PointID - CategoryID как сумма 
       последней продажи этой категории в эту точку.
       Если определить не удалось (первая продажа категории в точку), то = 0
    
    Расчет ведется по уникальным датам для каждой комбинации PointID-CategoryID, чтобы 
    корректно обрабатывать случаи, когда для одной точки и категории есть несколько записей 
    с одинаковым VisitDate.
    
    Args:
        df: Исходный датафрейм
        visit_date_col: Название колонки с датой посещения (по умолчанию 'VisitDate')
        point_id_col: Название колонки с идентификатором точки (по умолчанию 'PointID')
        category_id_col: Название колонки с идентификатором категории (по умолчанию 'CategoryID')
        
    Returns:
        Датафрейм с добавленными признаками
    """
    logger.info(f"Добавление признаков продаж категорий на основе колонок '{visit_date_col}', '{point_id_col}' и '{category_id_col}'")
    
    # Создаем копию датафрейма, чтобы не модифицировать исходный
    result_df = df.copy()
    
    # Преобразуем колонку с датой в формат datetime, если это еще не сделано
    if not pd.api.types.is_datetime64_any_dtype(result_df[visit_date_col]):
        result_df[visit_date_col] = pd.to_datetime(result_df[visit_date_col])
    
    # Создаем датафрейм с уникальными комбинациями PointID, CategoryID и VisitDate
    unique_sales = result_df[[point_id_col, category_id_col, visit_date_col]].drop_duplicates().copy()
    
    # Сортируем по PointID, CategoryID и VisitDate для корректного расчета shift
    unique_sales = unique_sales.sort_values([point_id_col, category_id_col, visit_date_col]).reset_index(drop=True)
    
    # Рассчитываем предыдущую дату продажи для каждой комбинации PointID-CategoryID с помощью groupby + shift
    unique_sales['PrevSaleDate'] = unique_sales.groupby([point_id_col, category_id_col])[visit_date_col].shift(1)
    
    # Вычисляем разницу в днях (векторизированно)
    unique_sales['DaysLastSalesCategory'] = (unique_sales[visit_date_col] - unique_sales['PrevSaleDate']).dt.days
    
    # Заполняем NaN значениями по умолчанию (7) для первой продажи категории в точку
    unique_sales['DaysLastSalesCategory'] = unique_sales['DaysLastSalesCategory'].fillna(7).astype(int)
    
    # Удаляем вспомогательные колонки
    unique_sales = unique_sales.drop(columns=['PrevSaleDate'])
    
    # Merge с исходным датафреймом для присваивания значений всем строкам
    result_df = result_df.merge(unique_sales, on=[point_id_col, category_id_col, visit_date_col], how='left')
    
    logger.info(f"Успешно добавлен 1 признак продаж категорий")
    logger.info(f"Новые колонки: ['DaysLastSalesCategory']")
    
    return result_df


def add_last_sales_category_feature(df: pd.DataFrame, visit_date_col: str = 'VisitDate', 
                                     point_id_col: str = 'PointID', 
                                     category_id_col: str = 'CategoryID',
                                     sum_col: str = 'SumRoubles') -> pd.DataFrame:
    """
    Добавляет признак суммы последней продажи категории в точку.
    
    Добавляемые признаки:
    1. LastSalesCategory - сумма последней продажи категории в точку.
       Рассчитывается для комбинации VisitDate - PointID - CategoryID как сумма 
       последней продажи этой категории в эту точку (предыдущая продажа по времени).
       Если определить не удалось (первая продажа категории в точку), то = 0
    
    Расчет ведется по уникальным датам для каждой комбинации PointID-CategoryID, чтобы 
    корректно обрабатывать случаи, когда для одной точки и категории есть несколько записей 
    с одинаковым VisitDate. В таком случае берется максимальная сумма продажи за день.
    
    Args:
        df: Исходный датафрейм
        visit_date_col: Название колонки с датой посещения (по умолчанию 'VisitDate')
        point_id_col: Название колонки с идентификатором точки (по умолчанию 'PointID')
        category_id_col: Название колонки с идентификатором категории (по умолчанию 'CategoryID')
        sum_col: Название колонки с суммой продажи (по умолчанию 'SumRoubles')
        
    Returns:
        Датафрейм с добавленным признаком
    """
    logger.info(f"Добавление признака LastSalesCategory на основе колонок '{visit_date_col}', '{point_id_col}', '{category_id_col}' и '{sum_col}'")
    
    # Создаем копию датафрейма, чтобы не модифицировать исходный
    result_df = df.copy()
    
    # Преобразуем колонку с датой в формат datetime, если это еще не сделано
    if not pd.api.types.is_datetime64_any_dtype(result_df[visit_date_col]):
        result_df[visit_date_col] = pd.to_datetime(result_df[visit_date_col])
    
    # Создаем датафрейм с уникальными комбинациями PointID, CategoryID, VisitDate и SumRoubles
    # Для случаев, когда в один день было несколько продаж одной категории в одну точку,
    # агрегируем сумму за день (суммируем)
    daily_sales = result_df.groupby([point_id_col, category_id_col, visit_date_col])[sum_col].sum().reset_index()
    
    # Переименовываем агрегированную колонку, чтобы избежать конфликта имен при merge
    daily_sales = daily_sales.rename(columns={sum_col: 'DailySum'})
    
    # Сортируем по PointID, CategoryID и VisitDate для корректного расчета shift
    daily_sales = daily_sales.sort_values([point_id_col, category_id_col, visit_date_col]).reset_index(drop=True)
    
    # Рассчитываем сумму предыдущей продажи для каждой комбинации PointID-CategoryID с помощью groupby + shift
    daily_sales['LastSalesCategory'] = daily_sales.groupby([point_id_col, category_id_col])['DailySum'].shift(1)
    
    # Заполняем NaN значениями по умолчанию (0) для первой продажи категории в точку
    daily_sales['LastSalesCategory'] = daily_sales['LastSalesCategory'].fillna(0)
    
    # Удаляем вспомогательную колонку DailySum
    daily_sales = daily_sales.drop(columns=['DailySum'])
    
    # Merge с исходным датафреймом для присваивания значений всем строкам
    result_df = result_df.merge(daily_sales, on=[point_id_col, category_id_col, visit_date_col], how='left')
    
    logger.info(f"Успешно добавлен признак LastSalesCategory")
    logger.info(f"Новые колонки: ['LastSalesCategory']")
    
    return result_df


def save_to_sql_server(df: pd.DataFrame, table_name: str = 'SNS_ML_features') -> None:
    """
    Сохраняет датафрейм в таблицу SQL Server 2012.
    Каждый раз таблица пересоздаётся через DROP/CREATE.
    
    Args:
        df: Датафрейм для сохранения
        table_name: Имя таблицы в БД (по умолчанию 'SNS_ML_features')
        
    Raises:
        Exception: При ошибке записи в БД
    """
    logger.info(f"Сохранение данных в таблицу {table_name} на SQL Server")
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # DROP существующей таблицы
        drop_query = f"IF OBJECT_ID('dbo.{table_name}', 'U') IS NOT NULL DROP TABLE dbo.{table_name}"
        logger.info(f"Удаление существующей таблицы: {drop_query}")
        cursor.execute(drop_query)
        conn.commit()
        
        # CREATE новой таблицы на основе типов данных DataFrame
        # Определяем типы колонок для SQL Server
        column_definitions = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "FLOAT"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "DATETIME"
            else:
                # Для строк и остальных типов используем NVARCHAR(MAX)
                sql_type = "NVARCHAR(MAX)"
            column_definitions.append(f"[{col}] {sql_type}")
        
        create_query = f"CREATE TABLE dbo.{table_name} (\n    " + ",\n    ".join(column_definitions) + "\n)"
        logger.info(f"Создание новой таблицы: {create_query}")
        cursor.execute(create_query)
        conn.commit()
        
        # Вставка данных
        logger.info(f"Вставка {len(df)} записей в таблицу {table_name}")
        
        # Формируем список колонок для вставки
        columns = list(df.columns)
        columns_str = ", ".join([f"[{col}]" for col in columns])
        placeholders = ", ".join(["?" for _ in columns])
        insert_query = f"INSERT INTO dbo.{table_name} ({columns_str}) VALUES ({placeholders})"
        
        # Вставляем данные batches
        batch_size = 10000
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df.iloc[start_idx:end_idx]
            
            # Преобразуем NaN в None для совместимости с pyodbc
            batch_data = []
            for _, row in batch.iterrows():
                row_data = tuple([None if pd.isna(val) else val for val in row.values])
                batch_data.append(row_data)
            
            cursor.executemany(insert_query, batch_data)
            conn.commit()
            logger.info(f"Вставлено записей: {end_idx}/{total_rows}")
        
        logger.info(f"Данные успешно сохранены в таблицу {table_name}")
        
    except pyodbc.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Соединение с базой данных закрыто")


def load_and_add_features(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Загружает сырые данные из БД с помощью fetch_raw_data и добавляет к ним календарные признаки, 
    признаки посещений и признаки продаж категорий.
    
    Args:
        start_date: Начальная дата периода выгрузки (включительно)
        end_date: Конечная дата периода выгрузки (не включительно)
        
    Returns:
        pd.DataFrame: Датафрейм с сырыми данными и добавленными календарными признаками, 
                      признаками посещений и признаками продаж категорий
        
    Raises:
        Exception: При ошибке загрузки данных или добавления признаков
    """
    logger.info(f"Загрузка данных за период {start_date} - {end_date}")
    
    # Получаем сырые данные через sns_ml_fetch_data
    df = fetch_raw_data(start_date, end_date)
    
    logger.info(f"Загружено {len(df)} записей")
    
    # Добавляем календарные признаки
    df_with_features = add_calendar_features(df)
    
    # Добавляем признаки посещений
    df_with_features = add_visit_features(df_with_features)
    
    # Добавляем признаки продаж категорий
    df_with_features = add_category_sales_features(df_with_features)
    
    # Добавляем признак LastSalesCategory
    df_with_features = add_last_sales_category_feature(df_with_features)
    
    logger.info("Данные успешно загружены и обогащены признаками")
    
    return df_with_features


# Пример использования
if __name__ == "__main__":
    import sys
    
    # Установка дат: end_date = текущая дата - 1 день, start_date = end_date - 1 год
    today = date.today()
    end_date = today
    start_date = end_date - timedelta(days=365)
    
    # Имя таблицы для сохранения в SQL Server (по умолчанию 'SNS_ML_features')
    table_name = sys.argv[1] if len(sys.argv) > 1 else 'SNS_ML_features'
    
    try:
        print(f"Загрузка продуктивных данных за период {start_date} - {end_date}...")
        
        # Загружаем продуктивные данные и добавляем признаки через load_and_add_features
        result = load_and_add_features(start_date, end_date)
        
        print("\nПервые 5 строк датафрейма с календарными признаками:")
        print(result.head())
        
        print(f"\nРазмер датафрейма: {result.shape}")
        print(f"\nТипы данных:\n{result.dtypes}")
        
        # Сохраняем полный датафрейм со всеми исходными и рассчитанными данными в SQL Server
        print(f"\nСохранение данных в таблицу {table_name} на SQL Server...")
        save_to_sql_server(result, table_name)
        print(f"Данные успешно сохранены в таблицу {table_name}")
        print(f"Всего записей: {len(result)}, всего колонок: {len(result.columns)}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
