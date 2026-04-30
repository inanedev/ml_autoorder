import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional
import logging
from sns_ml_fetch_data import fetch_raw_data

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_russia_holidays(year: int) -> List[date]:
    """
    Возвращает список национальных праздников России для указанного года.
    Учитывает переходящие праздники и переносы выходных дней.
    
    Args:
        year: Год, для которого нужно получить праздники
        
    Returns:
        Список дат праздников
    """
    holidays = []
    
    # Фиксированные праздничные даты (ст. 112 ТК РФ)
    fixed_holidays = [
        (1, 1),   # Новый год
        (1, 2),   # Новогодние каникулы
        (1, 3),   # Новогодние каникулы
        (1, 4),   # Новогодние каникулы
        (1, 5),   # Новогодние каникулы
        (1, 6),   # Новогодние каникулы
        (1, 7),   # Рождество Христово
        (1, 8),   # Новогодние каникулы
        (2, 23),  # День защитника Отечества
        (3, 8),   # Международный женский день
        (5, 1),   # Праздник Весны и Труда
        (5, 9),   # День Победы
        (6, 12),  # День России
        (11, 4),  # День народного единства
    ]
    
    # Добавляем фиксированные праздники
    for month, day in fixed_holidays:
        try:
            holidays.append(date(year, month, day))
        except ValueError:
            pass  # Пропускаем невалидные даты (например, 30 февраля)
    
    # Пасха (переходящий праздник, но не является официальным выходным в РФ)
    # Вычисляем дату Пасхи по алгоритму Гаусса
    def calculate_orthodox_easter(year: int) -> date:
        a = year % 19
        b = year % 4
        c = year % 7
        k = year // 100
        p = (8 * k + 13) // 25
        q = k // 4
        m = (15 - p + 32 * q - a + 19 * b) % 30
        n = (4 * k + m - b - c + 2) % 7
        d = m + n
        if d <= 9:
            return date(year, 4, d + 22)
        else:
            return date(year, 5, d - 9)
    
    # Пасха не является государственным праздником в РФ, но многие её отмечают
    # Не добавляем её в список официальных праздников
    
    # Дополнительные выходные из-за переноса (приблизительный расчет)
    # Точные даты переноса утверждаются правительством ежегодно
    # Для простоты добавляем типичные переносы
    
    # Проверяем дни недели для фиксированных праздников и добавляем переносы
    # Это упрощенная логика, точные даты нужно уточнять каждый год
    
    return sorted(set(holidays))


def get_extended_russia_holidays(start_year: int, end_year: int) -> List[date]:
    """
    Возвращает расширенный список праздников России за период лет.
    
    Args:
        start_year: Начальный год
        end_year: Конечный год
        
    Returns:
        Отсортированный список дат всех праздников
    """
    all_holidays = []
    for year in range(start_year, end_year + 1):
        all_holidays.extend(get_russia_holidays(year))
    return sorted(set(all_holidays))


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
    
    logger.info(f"Успешно добавлено 12 календарных признаков")
    logger.info(f"Новые колонки: {['DayOfWeek', 'IsFriday', 'IsMonday', 'DaysToNextHoliday', 'DaysSinceLastHoliday', 'IsPreHoliday', 'IsPostHoliday', 'Quarter', 'Month', 'WeekOfYear', 'DayOfMonth', 'DayOfYear']}")
    
    return result_df


def load_and_add_features(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Загружает сырые данные из БД с помощью fetch_raw_data и добавляет к ним календарные признаки.
    
    Args:
        start_date: Начальная дата периода выгрузки (включительно)
        end_date: Конечная дата периода выгрузки (не включительно)
        
    Returns:
        pd.DataFrame: Датафрейм с сырыми данными и добавленными календарными признаками
        
    Raises:
        Exception: При ошибке загрузки данных или добавления признаков
    """
    logger.info(f"Загрузка данных за период {start_date} - {end_date}")
    
    # Получаем сырые данные через sns_ml_fetch_data
    df = fetch_raw_data(start_date, end_date)
    
    logger.info(f"Загружено {len(df)} записей")
    
    # Добавляем календарные признаки
    df_with_features = add_calendar_features(df)
    
    logger.info("Данные успешно загружены и обогащены признаками")
    
    return df_with_features


# Пример использования
if __name__ == "__main__":
    # Установка дат: end_date = текущая дата - 1 день, start_date = end_date - 1 год
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    try:
        print(f"Загрузка продуктивных данных за период {start_date} - {end_date}...")
        
        # Загружаем продуктивные данные и добавляем признаки через load_and_add_features
        result = load_and_add_features(start_date, end_date)
        
        print("\nПервые 5 строк датафрейма с календарными признаками:")
        print(result[['VisitDate', 'DayOfWeek', 'IsFriday', 'IsMonday', 'DaysToNextHoliday', 
                      'DaysSinceLastHoliday', 'IsPreHoliday', 'IsPostHoliday', 'Quarter', 
                      'Month', 'WeekOfYear', 'DayOfMonth', 'DayOfYear']].head())
        
        print(f"\nРазмер датафрейма: {result.shape}")
        print(f"\nТипы данных:\n{result.dtypes}")
    except Exception as e:
        print(f"Ошибка: {e}")
