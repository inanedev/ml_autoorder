import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional
import logging
import os
from dotenv import load_dotenv

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


def fetch_raw_data(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> Optional[pd.DataFrame]:
    """
    Вызывает хранимую процедуру SNS_ML_Get_Raw_Data и возвращает данные в DataFrame.
    
    Args:
        start_date: Начальная дата (YYYY-MM-DD или объект date/datetime)
        end_date: Конечная дата (YYYY-MM-DD или объект date/datetime)
        
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
    
    args = parser.parse_args()
    
    # Настройка уровня логгирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Определение дат
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else datetime.now().date()
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else end_date - timedelta(days=30)
    
    logger.info(f"Загрузка данных с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        df = fetch_raw_data(start_date, end_date)
        
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
