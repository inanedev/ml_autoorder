# ETL для выгрузки сырых данных из базы данных Optimum Lipetsk

## Описание проекта

Модуль для извлечения данных о продажах из базы данных MSSQL с использованием хранимой процедуры `SNS_ML_Get_Raw_Data`. Данные агрегируются на уровень дня с обогащением атрибутами точек продаж и расчетом микрорегионов (сетка 3x3 км).

## Структура проекта

```
/workspace/
├── fetch_raw_data.py       # Основной Python модуль для выгрузки данных
├── SNS_ML_Get_Raw_Data.sql # SQL скрипт хранимой процедуры
├── .env                    # Файл с переменными окружения (не добавлять в git!)
├── requirements.txt        # Зависимости Python
└── README.md              # Этот файл
```

## Установка

### 1. Установка зависимостей Python

```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корне проекта и настройте параметры подключения:

```env
SQL_DRIVER=ODBC Driver 17 for SQL Server
SQL_SERVER=10.0.0.163
SQL_DATABASE=optimum_lipetsk
SQL_USER=optimum
SQL_PASSWORD=ваш_пароль
```

**Важно:** 
- Не коммитьте файл `.env` в систему контроля версий
- Используйте надежные пароли
- Файл `.env` уже добавлен в `.gitignore`

### 3. Установка драйвера ODBC для SQL Server

**Windows:** Драйвер обычно установлен по умолчанию

**Linux (Ubuntu/Debian):**
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev
```

**macOS:**
```bash
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
ACCEPT_EULA=Y brew install msodbcsql17 mssql-tools
```

## Использование

### Как библиотека

```python
from fetch_raw_data import fetch_raw_data
from datetime import datetime, timedelta

# Выгрузка данных за последние 30 дней
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)

df = fetch_raw_data(start_date, end_date)

# Работа с данными
print(f"Загружено {len(df)} записей")
print(df.head())

# Сохранение в CSV
df.to_csv('raw_data.csv', index=False)
```

### Как CLI утилита

```bash
# Выгрузка данных за последние 30 дней (по умолчанию)
python fetch_raw_data.py

# Выгрузка за указанный период
python fetch_raw_data.py --start-date 2024-01-01 --end-date 2024-02-01

# С сохранением в файл
python fetch_raw_data.py --start-date 2024-01-01 --end-date 2024-02-01 --output data.csv

# Подробный режим логгирования
python fetch_raw_data.py --verbose
```

### Аргументы командной строки

| Аргумент | Описание | По умолчанию |
|----------|----------|--------------|
| `--start-date` | Начальная дата в формате YYYY-MM-DD | 30 дней назад |
| `--end-date` | Конечная дата в формате YYYY-MM-DD | Сегодня |
| `--output` | Путь для сохранения результата в CSV | Не сохраняется |
| `--verbose` | Включить подробный режим логгирования | Выключен |

## Возвращаемые данные

Хранимая процедура возвращает следующие колонки:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `VisitDate` | DATE | Дата визита/продажи |
| `PointID` | INT | Идентификатор точки продаж |
| `CategoryID` | INT | Идентификатор категории товара |
| `BranchID` | INT | Идентификатор филиала/дистрибьютора |
| `PointClass` | VARCHAR | Класс точки продаж (атрибут 602) |
| `PointType` | VARCHAR | Тип точки продаж (атрибут 555) |
| `Lat` | FLOAT | Широта точки продаж |
| `Lon` | FLOAT | Долгота точки продаж |
| `MicroRegionID` | VARCHAR | Идентификатор микрорегиона (сетка 3x3 км) |
| `SumRoubles` | FLOAT | Сумма продаж в рублях за день |

## Логика работы

1. **Формирование справочника SKU** - выбираются активные товары с категориями
2. **Расчет признаков точек** - извлекаются координаты и атрибуты точек продаж
3. **Вычисление микрорегионов** - создается сетка 3x3 км на основе координат
4. **Агрегация продаж** - данные агрегируются на уровень дня с соединением всех справочников

## Обработка ошибок

Модуль включает комплексную обработку ошибок:

- Валидация входных дат (start_date <= end_date)
- Проверка наличия обязательных параметров подключения
- Обработка ошибок базы данных
- Логирование всех этапов выполнения
- Очистка ресурсов (закрытие соединений)

## Логирование

Модуль использует стандартный модуль `logging` Python:

- **INFO** - основные этапы выполнения
- **DEBUG** - детальная информация (при включенном verbose)
- **WARNING** - предупреждения (пустой результат)
- **ERROR** - ошибки выполнения

## Безопасность

### Рекомендации по безопасности

1. **Никогда не храните пароли в коде** - используйте переменные окружения
2. **Ограничьте доступ к файлу .env** - установите права 600 (chmod 600 .env)
3. **Используйте безопасное хранение секретов** в production:
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
4. **Регулярно меняйте пароли** баз данных
5. **Используйте минимально необходимые привилегии** для пользователя БД

## Требования к данным

- Драйвер ODBC для SQL Server
- Доступ к базе данных `optimum_lipetsk`
- Права на выполнение хранимой процедуры `SNS_ML_Get_Raw_Data`
- Таблицы: `DS_ITEMS`, `DS_ORDERS`, `DS_ORDERS_ITEMS`, `DS_FACES`, `DS_FACESATTRIBUTES`

## Производительность

Для оптимизации производительности:

- Хранимая процедура использует временные таблицы с индексами
- Фильтрация только активных записей
- Агрегация на уровне СУБД
- Параметризованные запросы для предотвращения SQL injection

## Примеры использования

### Ежедневная выгрузка

```python
from fetch_raw_data import fetch_raw_data
from datetime import datetime, timedelta

# Данные за вчера
yesterday = datetime.now().date() - timedelta(days=1)
df = fetch_raw_data(yesterday, yesterday + timedelta(days=1))
```

### Выгрузка за месяц

```python
# Данные за январь 2024
df = fetch_raw_data('2024-01-01', '2024-02-01')
```

### Интеграция с pandas

```python
import pandas as pd
from fetch_raw_data import fetch_raw_data

df = fetch_raw_data('2024-01-01', '2024-01-31')

# Анализ по микрорегионам
region_stats = df.groupby('MicroRegionID').agg({
    'SumRoubles': 'sum',
    'PointID': 'nunique'
}).reset_index()

# Анализ по категориям
category_stats = df.groupby('CategoryID')['SumRoubles'].sum().sort_values(ascending=False)
```

## Расширение функциональности

### Добавление новых источников данных

1. Создайте новую хранимую процедуру в SQL
2. Добавьте функцию-обертку в `fetch_raw_data.py`
3. Обновите документацию

### Интеграция с ETL-конвейером

```python
from fetch_raw_data import fetch_raw_data
import pandas as pd

def etl_pipeline():
    # Extract
    df = fetch_raw_data('2024-01-01', '2024-01-31')
    
    # Transform
    df['Revenue'] = df['SumRoubles'] * 1.2  # Пример трансформации
    
    # Load
    df.to_parquet('processed_data.parquet')
    
    return df
```

## Тестирование

Для тестирования подключений и функциональности:

```bash
# Проверка загрузки модуля
python -c "import fetch_raw_data; print('OK')"

# Проверка конфигурации
python -c "from fetch_raw_data import SQL_CONFIG; print(SQL_CONFIG)"

# Тестовая выгрузка (1 день)
python fetch_raw_data.py --start-date 2024-01-01 --end-date 2024-01-02 --verbose
```

## Устранение неполадок

### Ошибка подключения к БД

Проверьте:
- Корректность параметров в `.env`
- Доступность сервера БД
- Наличие драйвера ODBC
- Сетевые настройки (firewall)

### Ошибка "Модуль не найден"

```bash
pip install -r requirements.txt
```

### Ошибка кодировки

Убедитесь, что файлы сохранены в UTF-8:
```bash
file fetch_raw_data.py
```

## Лицензия

Внутренний проект компании. Все права защищены.

## Контакты

По вопросам обращайтесь к команде разработки аналитических систем.
