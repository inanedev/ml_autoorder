import pyodbc
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
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
        pyodbc.Error: При ошибке подключения
    """
    conn_str = (
        f"DRIVER={{{SQL_CONFIG['driver']}}};"
        f"SERVER={SQL_CONFIG['server']};"
        f"DATABASE={SQL_CONFIG['database']};"
        f"UID={SQL_CONFIG['user']};"
        f"PWD={SQL_CONFIG['password']}"
    )
    logger.info(f"Подключение к серверу {SQL_CONFIG['server']}, база данных {SQL_CONFIG['database']}")
    return pyodbc.connect(conn_str)


def fetch_raw_data(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Загружает сырые данные о продажах из хранимой процедуры SNS_ML_Get_Raw_Data.
    
    Args:
        start_date: Начальная дата периода выгрузки (включительно)
        end_date: Конечная дата периода выгрузки (не включительно)
        
    Returns:
        pd.DataFrame: Датафрейм с колонками:
            - VisitDate: Дата визита/продажи
            - PointID: Идентификатор точки продаж
            - CategoryID: Идентификатор категории товара
            - BranchID: Идентификатор филиала/дистрибьютора
            - PointClass: Класс точки продаж
            - PointType: Тип точки продаж
            - Lat: Широта точки продаж
            - Lon: Долгота точки продаж
            - MicroRegionID: Идентификатор микрорегиона (сетка 3x3 км)
            - SumRoubles: Сумма продаж в рублях за день
            
    Raises:
        Exception: При ошибке выполнения хранимой процедуры
    """
    query = """
    EXEC dbo.SNS_ML_Get_Raw_Data 
        @StartDate = ?, 
        @EndDate = ?
    """
    
    logger.info(f"Вызов хранимой процедуры SNS_ML_Get_Raw_Data с датами: {start_date} - {end_date}")
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Выполнение хранимой процедуры с параметрами
        cursor.execute(query, (start_date, end_date))
        
        # Получение колонок из описания курсора
        columns = [column[0] for column in cursor.description]
        
        # Fetch всех строк
        rows = cursor.fetchall()
        
        # Создание DataFrame
        sns_ml_raw_data = pd.DataFrame.from_records(rows, columns=columns)
        
        # Выбираем только нужные колонки, исключая OrderCountDocs
        required_columns = [
            'VisitDate', 'PointID', 'CategoryID', 'BranchID',
            'PointClass', 'PointType', 'Lat', 'Lon',
            'MicroRegionID', 'SumRoubles'
        ]
        sns_ml_raw_data = sns_ml_raw_data[required_columns]
        
        logger.info(f"Успешно загружено {len(sns_ml_raw_data)} записей")
        logger.info(f"Колонки датафрейма: {list(sns_ml_raw_data.columns)}")
        
        return sns_ml_raw_data
        
    except pyodbc.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Соединение с базой данных закрыто")


# Пример использования
if __name__ == "__main__":
    # Установка дат: @EndDate = текущая дата - 1 день, @StartDate = @EndDate - 1 год
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    try:
        sns_ml_raw_data = fetch_raw_data(start_date, end_date)
        print("\nПервые 5 строк датафрейма:")
        print(sns_ml_raw_data.head())
        print(f"\nРазмер датафрейма: {sns_ml_raw_data.shape}")
        print(f"\nТипы данных:\n{sns_ml_raw_data.dtypes}")
    except Exception as e:
        print(f"Ошибка: {e}")
