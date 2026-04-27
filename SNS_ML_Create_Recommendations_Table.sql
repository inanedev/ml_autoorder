/****** Object:  Table [dbo].[SNS_ML_Brand_Recommendations]    Script Date: 2026-04-27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[SNS_ML_Brand_Recommendations](
    -- Первичный ключ (автоинкремент)
    [RecommendationId] BIGINT IDENTITY(1,1) NOT NULL,
    
    -- Входные параметры модели
    [PointId] INT NOT NULL,                    -- ID торговой точки
    [CategoryId] INT NOT NULL,                 -- ID категории товара
    [ForecastAmount] DECIMAL(18,2) NOT NULL,   -- Прогноз суммы продаж категории
    [DaysUntilVisit] INT NOT NULL,             -- Дней до следующего визита мерчандайзера
    [ReferenceDate] DATETIME NOT NULL,         -- Дата расчета рекомендации
    [TargetDayOfWeek] INT NOT NULL,            -- День недели целевого визита (0-6)
    
    -- Параметры бренда из правил (SNS_ML_Get_Brand_Rules)
    [BrandId] INT NOT NULL,                    -- ID бренда (GroupID)
    [BrandName] NVARCHAR(255) NOT NULL,        -- Название бренда
    [BrandQuantum] INT NOT NULL,               -- Квант поставки бренда
    [ImportanceLabel] NVARCHAR(100) NULL,      -- Метка важности (Must List, Drive List, и т.д.)
    [PriorityWeight] INT NOT NULL,             -- Базовый вес приоритета (0-4)
    [IsTurboBrand] BIT NOT NULL,               -- Флаг Turbo бренда
    [AvgPrice] DECIMAL(18,2) NOT NULL,         -- Средняя цена бренда
    
    -- Локальные метрики продаж
    [IsTopLocal] BIT NOT NULL,                 -- Флаг топ-бренда в микрорегионе
    [AvgDailySales] DECIMAL(18,4) NOT NULL,    -- Средние ежедневные продажи в штуках
    [RawNeed] DECIMAL(18,4) NOT NULL,          -- Сырая потребность (avg_daily_sales * days_until_visit)
    
    -- Результат расчета
    [Priority] INT NOT NULL,                   -- Итоговый приоритет бренда
    [RecommendedQty] INT NOT NULL,             -- Рекомендованное количество (округлено до кванта)
    [EstimatedCost] DECIMAL(18,2) NOT NULL,    -- Расчетная стоимость (recommended_qty * avg_price)
    [Included] BIT NOT NULL,                   -- Флаг включения в финальный заказ
    
    -- Метаданные записи
    [CreatedAt] DATETIME NOT NULL CONSTRAINT [DF_SNS_ML_Brand_Recommendations_CreatedAt] DEFAULT GETDATE(),
    [ModelVersion] NVARCHAR(50) NULL,          -- Версия ML модели (опционально)
    
    CONSTRAINT [PK_SNS_ML_Brand_Recommendations] PRIMARY KEY CLUSTERED 
    (
        [RecommendationId] ASC
    ) WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]
GO

-- Индексы для оптимизации выборок

-- Индекс для поиска рекомендаций по точке и дате
CREATE NONCLUSTERED INDEX [IX_SNS_ML_Brand_Recommendations_PointId_ReferenceDate] 
ON [dbo].[SNS_ML_Brand_Recommendations]([PointId] ASC, [ReferenceDate] DESC)
INCLUDE ([CategoryId], [BrandId], [RecommendedQty], [EstimatedCost], [Included])
WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO

-- Индекс для анализа по категориям
CREATE NONCLUSTERED INDEX [IX_SNS_ML_Brand_Recommendations_CategoryId_ReferenceDate] 
ON [dbo].[SNS_ML_Brand_Recommendations]([CategoryId] ASC, [ReferenceDate] DESC)
INCLUDE ([PointId], [BrandId], [RecommendedQty], [EstimatedCost])
WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO

-- Индекс для анализа по брендам
CREATE NONCLUSTERED INDEX [IX_SNS_ML_Brand_Recommendations_BrandId] 
ON [dbo].[SNS_ML_Brand_Recommendations]([BrandId] ASC)
INCLUDE ([PointId], [CategoryId], [ReferenceDate], [RecommendedQty], [Included])
WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO

-- Индекс для отбора включенных позиций заказа
CREATE NONCLUSTERED INDEX [IX_SNS_ML_Brand_Recommendations_Included] 
ON [dbo].[SNS_ML_Brand_Recommendations]([Included] ASC)
WHERE [Included] = 1
WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, SORT_IN_TEMPDB = OFF, DROP_EXISTING = OFF, ONLINE = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
GO

-- Добавляем описание таблицы и колонок
EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Таблица хранения рекомендаций по заказу брендов на основе ML-прогноза и правил мерчандайзинга' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Уникальный идентификатор записи рекомендации' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'RecommendationId'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'ID торговой точки' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'PointId'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'ID категории товара' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'CategoryId'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Прогноз суммы продаж категории от ML-модели' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'ForecastAmount'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Количество дней до следующего визита мерчандайзера' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'DaysUntilVisit'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Дата расчета рекомендации' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'ReferenceDate'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'День недели целевого визита (0=Понедельник, 6=Воскресенье)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'TargetDayOfWeek'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'ID бренда' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'BrandId'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Название бренда' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'BrandName'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Квант поставки (минимальное кратное для заказа)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'BrandQuantum'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Метка важности бренда (Must List, Drive List, NPI, SPEED KPI, Regular)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'ImportanceLabel'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Базовый вес приоритета из правил (0-4)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'PriorityWeight'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Флаг принадлежности к Turbo брендам' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'IsTurboBrand'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Средняя цена бренда (реализация или прайс-лист)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'AvgPrice'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Флаг топ-бренда в микрорегионе для данного дня недели' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'IsTopLocal'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Средние ежедневные продажи в штуках для данного дня недели' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'AvgDailySales'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Сырая потребность в штуках (без округления до кванта)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'RawNeed'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Итоговый приоритет бренда (расчетный)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'Priority'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Рекомендованное количество для заказа (округлено до кванта)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'RecommendedQty'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Расчетная стоимость позиции (recommended_qty * avg_price)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'EstimatedCost'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Флаг включения бренда в финальный заказ (после распределения бюджета)' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'Included'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Дата и время создания записи' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'CreatedAt'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_Description', @value=N'Версия ML модели, сгенерировавшей рекомендацию' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', @level2type=N'COLUMN',@level2name=N'ModelVersion'
GO
