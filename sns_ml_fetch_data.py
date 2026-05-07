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


def fetch_test_data(target_date: date) -> pd.DataFrame:
    """
    Загружает тестовые данные из хранимой процедуры SNS_ML_Get_Test_Data.
    
    Args:
        target_date: Дата, на которую загружаются тестовые данные
        
    Returns:
        pd.DataFrame: Датафрейм с колонками:
            - VisitDate: Дата прогноза (TargetDate)
            - PointID: Идентификатор точки продаж
            - CategoryID: Идентификатор категории товара
            - BranchID: Идентификатор филиала/дистрибьютора
            - PointClass: Класс точки продаж
            - PointType: Тип точки продаж
            - Lat: Широта точки продаж
            - Lon: Долгота точки продаж
            - MicroRegionID: Идентификатор микрорегиона (сетка 3x3 км)
            - DayOfWeek: День недели (1-понедельник, 7-воскресенье)
            - IsFriday: Признак пятницы (1/0)
            - IsMonday: Признак понедельника (1/0)
            - DaysToNextHoliday: Дней до ближайшего праздника
            - DaysSinceLastHoliday: Дней от последнего праздника
            - IsPreHoliday: Признак предпраздничного дня (1/0)
            - IsPostHoliday: Признак постпраздничного дня (1/0)
            - Quarter: Номер квартала
            - Month: Номер месяца
            - WeekOfYear: Номер недели в году
            - DayOfMonth: День месяца
            - DayOfYear: День года
            - isEndOfMonth: Признак конца месяца (1/0)
            - DaysLastVisit: Дней с последнего визита
            - DaysNextVisit: Дней до следующего визита
            - DaysLastSalesCategory: Дней с последней продажи категории
            - LastSalesCategory: Сумма последней продажи категории
            
    Raises:
        Exception: При ошибке выполнения хранимой процедуры
    """
    query = """
    EXEC dbo.SNS_ML_Get_Test_Data 
        @TargetDate = ?
    """
    
    logger.info(f"Вызов хранимой процедуры SNS_ML_Get_Test_Data с датой: {target_date}")
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Выполнение хранимой процедуры с параметром
        cursor.execute(query, (target_date,))
        
        # Получение колонок из описания курсора
        columns = [column[0] for column in cursor.description]
        
        # Fetch всех строк
        rows = cursor.fetchall()
        
        # Создание DataFrame
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        logger.info(f"Успешно загружено {len(df)} записей")
        logger.info(f"Колонки датафрейма: {list(df.columns)}")
        
        return df
        
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


def check_and_cleanup_predictions_table(target_date: date, table_name: str = 'SNS_ML_Predictions') -> None:
    """
    Проверяет существование таблицы SNS_ML_Predictions и её структуру.
    Если таблица существует и в ней есть данные с датой прогноза target_date,
    удаляет эти данные.
    
    Args:
        target_date: Дата прогноза для проверки и очистки
        table_name: Имя таблицы для проверки
    
    Raises:
        Exception: При ошибке работы с БД
    """
    logger.info(f"Проверка таблицы {table_name} на наличие данных за {target_date}")
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Проверяем существование таблицы
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            logger.info(f"Таблица {table_name} не существует")
            return
        
        # Проверяем структуру таблицы - наличие колонки predict_new
        check_column_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'predict_new'
        """
        cursor.execute(check_column_query)
        column_exists = cursor.fetchone()[0] > 0
        
        if not column_exists:
            logger.info(f"Таблица {table_name} существует, но не имеет колонки predict_new. Требуется пересоздание.")
            drop_query = f"DROP TABLE dbo.{table_name}"
            cursor.execute(drop_query)
            conn.commit()
            logger.info(f"Таблица {table_name} удалена")
            return
        
        # Проверяем наличие колонки VisitDate для фильтрации по дате
        check_date_column_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}' AND COLUMN_NAME = 'VisitDate'
        """
        cursor.execute(check_date_column_query)
        date_column_exists = cursor.fetchone()[0] > 0
        
        if date_column_exists:
            # Удаляем данные с указанной датой прогноза
            delete_query = f"DELETE FROM dbo.{table_name} WHERE VisitDate = ?"
            cursor.execute(delete_query, (target_date,))
            deleted_rows = cursor.rowcount
            conn.commit()
            if deleted_rows > 0:
                logger.info(f"Удалено {deleted_rows} записей с датой прогноза {target_date}")
            else:
                logger.info(f"Данных с датой прогноза {target_date} не найдено")
        else:
            logger.warning(f"Колонка VisitDate отсутствует в таблице {table_name}. Очистка по дате невозможна.")
        
    except pyodbc.Error as e:
        logger.error(f"Ошибка базы данных: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при проверке таблицы: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Соединение с базой данных закрыто")


def save_predictions_to_sql(df: pd.DataFrame, table_name: str = 'SNS_ML_Predictions', 
                            target_date: Optional[date] = None, 
                            recreate_if_structure_mismatch: bool = True) -> None:
    """
    Сохраняет датафрейм с предсказаниями в таблицу SQL Server.
    Если таблица не существует — создаёт её.
    Если таблица существует и структура совпадает — удаляет данные за target_date и вставляет новые.
    Если таблица существует но структура не совпадает — пересоздаёт таблицу (если флаг установлен).
    
    Args:
        df: Датафрейм с данными для сохранения (включая predict_new)
        table_name: Имя таблицы в БД (по умолчанию 'SNS_ML_Predictions')
        target_date: Дата прогноза для очистки данных перед вставкой
        recreate_if_structure_mismatch: Флаг пересоздания таблицы при несовпадении структуры
    
    Raises:
        Exception: При ошибке записи в БД
    """
    logger.info(f"Сохранение предсказаний в таблицу {table_name} на SQL Server")
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Проверяем существование таблицы и её структуру
        need_create = False
        
        check_table_query = f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            logger.info(f"Таблица {table_name} не существует. Будет создана.")
            need_create = True
        else:
            # Проверяем структуру - наличие всех необходимых колонок
            required_columns = set(df.columns)
            existing_columns = set()
            
            check_columns_query = f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
            """
            cursor.execute(check_columns_query)
            for row in cursor.fetchall():
                existing_columns.add(row[0])
            
            # Проверяем совпадение структуры
            if required_columns != existing_columns:
                missing_cols = required_columns - existing_columns
                extra_cols = existing_columns - required_columns
                logger.warning(f"Структура таблицы не совпадает:")
                if missing_cols:
                    logger.warning(f"  Отсутствуют колонки: {missing_cols}")
                if extra_cols:
                    logger.warning(f"  Лишние колонки: {extra_cols}")
                
                if recreate_if_structure_mismatch:
                    logger.info(f"Таблица будет пересоздана")
                    # DROP существующей таблицы
                    drop_query = f"IF OBJECT_ID('dbo.{table_name}', 'U') IS NOT NULL DROP TABLE dbo.{table_name}"
                    logger.info(f"Удаление существующей таблицы: {drop_query}")
                    cursor.execute(drop_query)
                    conn.commit()
                    need_create = True
                else:
                    logger.warning(f"Пересоздание таблицы отключено. Попытка вставки в существующую таблицу.")
            else:
                # Структура совпадает - удаляем данные за дату прогноза
                if target_date is not None and 'VisitDate' in existing_columns:
                    delete_query = f"DELETE FROM dbo.{table_name} WHERE VisitDate = ?"
                    cursor.execute(delete_query, (target_date,))
                    deleted_rows = cursor.rowcount
                    conn.commit()
                    logger.info(f"Удалено {deleted_rows} записей с датой прогноза {target_date}")
                elif target_date is not None:
                    logger.warning(f"Колонка VisitDate отсутствует в таблице {table_name}. Очистка по дате невозможна.")
        
        # CREATE таблицы если нужно
        if need_create:
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
                    sql_type = "NVARCHAR(MAX)"
                column_definitions.append(f"[{col}] {sql_type}")
            
            create_query = f"CREATE TABLE dbo.{table_name} (\n    " + ",\n    ".join(column_definitions) + "\n)"
            logger.info(f"Создание новой таблицы: {create_query}")
            cursor.execute(create_query)
            conn.commit()
        
        # Вставка данных
        logger.info(f"Вставка {len(df)} записей в таблицу {table_name}")
        
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
        
        logger.info(f"Предсказания успешно сохранены в таблицу {table_name}")
        
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


# Пример использования
if __name__ == "__main__":
    # Установка дат: @EndDate = текущая дата - 1 день, @StartDate = @EndDate - 1 год
    today = date.today()
    end_date = today 
    start_date = end_date - timedelta(days=365)
    
    try:
        sns_ml_raw_data = fetch_raw_data(start_date, end_date)
        print("\nПервые 5 строк датафрейма:")
        print(sns_ml_raw_data.head())
        print(f"\nРазмер датафрейма: {sns_ml_raw_data.shape}")
        print(f"\nТипы данных:\n{sns_ml_raw_data.dtypes}")
    except Exception as e:
        print(f"Ошибка: {e}")
