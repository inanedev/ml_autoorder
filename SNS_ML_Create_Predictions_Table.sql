-- ============================================================================
-- Скрипт создания таблицы для хранения результатов ML прогнозирования
-- Таблица включает все признаки из тестового набора данных + поле с предсказанием
-- ============================================================================

-- Удаляем таблицу если она существует
IF OBJECT_ID('dbo.SNS_ML_Predictions', 'U') IS NOT NULL
    DROP TABLE dbo.SNS_ML_Predictions;
GO

-- Создание таблицы
CREATE TABLE dbo.SNS_ML_Predictions
(
    -- ========================================================================
    -- БАЗОВЫЕ ДАННЫЕ
    -- ========================================================================
    
    -- Дата прогноза
    VisitDate DATE NOT NULL,
    
    -- Идентификатор точки продаж
    PointID INT NOT NULL,
    
    -- Идентификатор категории товара
    CategoryID INT NOT NULL,
    
    -- Идентификатор филиала/дистрибьютора
    BranchID INT NULL,
    
    -- Класс точки продаж (из атрибута 602)
    PointClass NVARCHAR(255) NULL,
    
    -- Тип точки продаж (из атрибута 555)
    PointType NVARCHAR(255) NULL,
    
    -- Широта точки продаж
    Lat FLOAT NULL,
    
    -- Долгота точки продаж
    Lon FLOAT NULL,
    
    -- Идентификатор микрорегиона (сетка 3x3 км)
    MicroRegionID NVARCHAR(50) NULL,
    
    -- ========================================================================
    -- КАЛЕНДАРНЫЕ ФИЧИ
    -- ========================================================================
    
    -- День недели (0=понедельник, 6=воскресенье)
    day_of_week TINYINT NULL,
    
    -- Флаг выходного дня
    is_weekend BIT NULL,
    
    -- Флаг понедельника
    is_monday BIT NULL,
    
    -- Флаг пятницы
    is_friday BIT NULL,
    
    -- Флаг субботы
    is_saturday BIT NULL,
    
    -- Флаг воскресенья
    is_sunday BIT NULL,
    
    -- Флаг праздника России
    is_holiday BIT NULL,
    
    -- Флаг предпраздничного дня
    is_pre_holiday BIT NULL,
    
    -- Флаг дня после праздника
    is_post_holiday BIT NULL,
    
    -- Месяц года
    month TINYINT NULL,
    
    -- Квартал года
    quarter TINYINT NULL,
    
    -- Неделя года
    week_of_year TINYINT NULL,
    
    -- Флаг начала месяца
    is_month_start BIT NULL,
    
    -- Флаг конца месяца
    is_month_end BIT NULL,
    
    -- День месяца
    day_of_month TINYINT NULL,
    
    -- День года
    day_of_year SMALLINT NULL,
    
    -- Дней до ближайшего праздника
    days_to_holiday SMALLINT NULL,
    
    -- Дней от последнего праздника
    days_from_holiday SMALLINT NULL,
    
    -- ========================================================================
    -- ФИЧИ ИСТОРИИ ЗАКАЗОВ
    -- ========================================================================
    
    -- Дней назад точка брала эту категорию
    Days_Since_Last_Order_Category INT NULL,
    
    -- Дней назад был любой заказ от точки
    Days_Since_Last_Order_Total INT NULL,
    
    -- Средний интервал между закупками категории
    Average_Interval_Category FLOAT NULL,
    
    -- Дней до следующего визита точки продаж
    Days_Until_Next_Visit INT NULL,
    
    -- ========================================================================
    -- ФИЧИ ПРОДАЖ
    -- ========================================================================
    
    -- Сумма предыдущего заказа по категории
    Prev_Order_Amount_Category DECIMAL(18, 2) NULL,
    
    -- Скользящее среднее за 3 дня по категории
    SMA_3_Category DECIMAL(18, 2) NULL,
    
    -- Скользящее среднее за 7 дней по категории
    SMA_7_Category DECIMAL(18, 2) NULL,
    
    -- Скользящее среднее за 30 дней по категории
    SMA_30_Category DECIMAL(18, 2) NULL,
    
    -- Отношение среднего чека за неделю к месяцу (моментум)
    Momentum_Category FLOAT NULL,
    
    -- Скользящее стандартное отклонение
    StdDev_Category FLOAT NULL,
    
    -- ========================================================================
    -- ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (ПРЕДСКАЗАНИЕ)
    -- ========================================================================
    
    -- Предсказанная сумма категории (результат работы ML модели)
    Predicted_Category_Sum DECIMAL(18, 2) NULL,
    
    -- ========================================================================
    -- СЛУЖЕБНЫЕ ПОЛЯ
    -- ========================================================================
    
    -- Дата и время записи прогноза в таблицу
    CreatedAt DATETIME NOT NULL CONSTRAINT DF_SNS_ML_Predictions_CreatedAt DEFAULT GETDATE(),
    
    -- Идентификатор версии модели, сделавшей прогноз
    ModelVersion NVARCHAR(50) NULL,
    
    -- Уверенность модели в предсказании (если применимо)
    Prediction_Confidence FLOAT NULL

);
GO

-- ============================================================================
-- СОЗДАНИЕ ИНДЕКСОВ ДЛЯ ОПТИМИЗАЦИИ ЗАПРОСОВ
-- ============================================================================

-- Кластерный индекс по дате, точке и категории (основной сценарий выборки)
CREATE CLUSTERED INDEX IX_SNS_ML_Predictions_DatePointCat 
ON dbo.SNS_ML_Predictions (VisitDate, PointID, CategoryID);
GO

-- Индекс для поиска по точкам продаж
CREATE NONCLUSTERED INDEX IX_SNS_ML_Predictions_PointID 
ON dbo.SNS_ML_Predictions (PointID, VisitDate);
GO

-- Индекс для поиска по категориям
CREATE NONCLUSTERED INDEX IX_SNS_ML_Predictions_CategoryID 
ON dbo.SNS_ML_Predictions (CategoryID, VisitDate);
GO

-- Индекс для анализа по филиалам
CREATE NONCLUSTERED INDEX IX_SNS_ML_Predictions_BranchID 
ON dbo.SNS_ML_Predictions (BranchID, VisitDate);
GO

-- Индекс для поиска по дате создания
CREATE NONCLUSTERED INDEX IX_SNS_ML_Predictions_CreatedAt 
ON dbo.SNS_ML_Predictions (CreatedAt);
GO

-- ============================================================================
-- ДОБАВЛЕНИЕ ОПИСАНИЯ ТАБЛИЦЫ И СТОЛБЦОВ
-- ============================================================================

EXEC sys.sp_addextendedproperty 
    @name = N'Description', 
    @value = N'Таблица для хранения результатов ML прогнозирования продаж по категориям. Содержит все признаки из тестового набора данных + поле с предсказанной суммой категории.', 
    @level0type = N'SCHEMA', @level0name = 'dbo', 
    @level1type = N'TABLE', @level1name = 'SNS_ML_Predictions';
GO

EXEC sys.sp_addextendedproperty 
    @name = N'Description', 
    @value = N'Дата прогноза', 
    @level0type = N'SCHEMA', @level0name = 'dbo', 
    @level1type = N'TABLE', @level1name = 'SNS_ML_Predictions', 
    @level2type = N'COLUMN', @level2name = 'VisitDate';
GO

EXEC sys.sp_addextendedproperty 
    @name = N'Description', 
    @value = N'Предсказанная моделью сумма продаж по категории', 
    @level0type = N'SCHEMA', @level0name = 'dbo', 
    @level1type = N'TABLE', @level1name = 'SNS_ML_Predictions', 
    @level2type = N'COLUMN', @level2name = 'Predicted_Category_Sum';
GO

PRINT 'Таблица SNS_ML_Predictions успешно создана!';
PRINT 'Созданы индексы для оптимизации запросов.';
GO
