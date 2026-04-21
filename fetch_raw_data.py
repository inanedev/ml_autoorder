import pyodbc
import pandas as pd
from datetime import datetime, timedelta

# Конфигурация подключения к MSSQL
SQL_CONFIG = {
    'driver': 'SQL Server',
    'server': '10.0.0.163',
    'database': 'optimum_lipetsk',
    'user': 'optimum',
    'password': '123456qqq'
}

def get_connection():
    """Создает и возвращает соединение с базой данных."""
    conn_str = (
        f"DRIVER={{{SQL_CONFIG['driver']}}}; "
        f"SERVER={SQL_CONFIG['server']}; "
        f"DATABASE={SQL_CONFIG['database']}; "
        f"UID={SQL_CONFIG['user']}; "
        f"PWD={SQL_CONFIG['password']}"
    )
    return pyodbc.connect(conn_str)

def fetch_raw_data(start_date, end_date):
    """
    Вызывает хранимую процедуру SNS_ML_Get_Raw_Data и возвращает данные в DataFrame.
    
    Параметры:
    start_date (str or date): Начальная дата (YYYY-MM-DD)
    end_date (str or date): Конечная дата (YYYY-MM-DD)
    """
    query = """
    EXEC dbo.SNS_ML_Get_Raw_Data 
        @StartDate = ?, 
        @EndDate = ?
    """
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Выполнение хранимой процедуры
        cursor.execute(query, (start_date, end_date))
        
        # Получение имен колонок из описания курсора
        columns = [column[0] for column in cursor.description]
        
        # Извлечение всех строк результата
        rows = cursor.fetchall()
        
        # Преобразование в pandas DataFrame
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        print(f"Успешно загружено {len(df)} записей за период с {start_date} по {end_date}")
        return df
        
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Пример использования: выгрузка данных за последние 30 дней
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    # Форматирование дат в строку для передачи в процедуру (если требуется)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Загрузка данных с {start_date_str} по {end_date_str}...")
    
    try:
        df = fetch_raw_data(start_date_str, end_date_str)
        
        # Вывод первых строк для проверки
        print("\nПервые 5 строк данных:")
        print(df.head())
        
        print("\nИнформация о типах данных:")
        print(df.info())
        
        # Здесь можно добавить логику сохранения в файл или дальнейшей обработки
        # df.to_csv('raw_data.csv', index=False)
        
    except Exception as e:
        print(f"Критическая ошибка выполнения: {e}")
