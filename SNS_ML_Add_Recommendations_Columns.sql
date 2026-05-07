/****** Object:  Script to add new columns to SNS_ML_Brand_Recommendations    Script Date: 2026-04-27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- =============================================
-- Скрипт добавления новых полей в таблицу рекомендаций
-- 
-- Новые поля:
--   PredictedSum - Исходный прогноз суммы категории из ML-модели
--   ExtendedSum - Скорректированный бюджет после учёта разницы между 
--                 общим классом точки и классом категории
--   Comment - Пояснение для сотрудника, почему рассчитано такое значение
-- =============================================

DECLARE @TableName NVARCHAR(255) = 'dbo.SNS_ML_Brand_Recommendations'

-- =============================================
-- Добавление поля PredictedSum
-- =============================================
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(@TableName) 
    AND name = 'PredictedSum'
)
BEGIN
    ALTER TABLE [dbo].[SNS_ML_Brand_Recommendations]
    ADD [PredictedSum] DECIMAL(18,2) NULL
    
    PRINT 'Добавлено поле PredictedSum'
END
ELSE
BEGIN
    PRINT 'Поле PredictedSum уже существует'
END
GO

-- =============================================
-- Добавление поля ExtendedSum
-- =============================================
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(@TableName) 
    AND name = 'ExtendedSum'
)
BEGIN
    ALTER TABLE [dbo].[SNS_ML_Brand_Recommendations]
    ADD [ExtendedSum] DECIMAL(18,2) NULL
    
    PRINT 'Добавлено поле ExtendedSum'
END
ELSE
BEGIN
    PRINT 'Поле ExtendedSum уже существует'
END
GO

-- =============================================
-- Добавление поля Comment
-- =============================================
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID(@TableName) 
    AND name = 'Comment'
)
BEGIN
    ALTER TABLE [dbo].[SNS_ML_Brand_Recommendations]
    ADD [Comment] NVARCHAR(500) NULL
    
    PRINT 'Добавлено поле Comment'
END
ELSE
BEGIN
    PRINT 'Поле Comment уже существует'
END
GO

-- =============================================
-- Добавление описаний для новых полей
-- =============================================

-- Описание для PredictedSum
IF NOT EXISTS (
    SELECT 1 
    FROM sys.extended_properties 
    WHERE major_id = OBJECT_ID('dbo.SNS_ML_Brand_Recommendations')
    AND minor_id = (SELECT column_id FROM sys.columns WHERE object_id = OBJECT_ID('dbo.SNS_ML_Brand_Recommendations') AND name = 'PredictedSum')
    AND name = 'MS_Description'
)
BEGIN
    EXEC sys.sp_addextendedproperty 
        @name=N'MS_Description', 
        @value=N'Исходный прогноз суммы категории из ML-модели (до корректировки по классу точки)', 
        @level0type=N'SCHEMA',@level0name=N'dbo', 
        @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', 
        @level2type=N'COLUMN',@level2name=N'PredictedSum'
    PRINT 'Добавлено описание для PredictedSum'
END
GO

-- Описание для ExtendedSum
IF NOT EXISTS (
    SELECT 1 
    FROM sys.extended_properties 
    WHERE major_id = OBJECT_ID('dbo.SNS_ML_Brand_Recommendations')
    AND minor_id = (SELECT column_id FROM sys.columns WHERE object_id = OBJECT_ID('dbo.SNS_ML_Brand_Recommendations') AND name = 'ExtendedSum')
    AND name = 'MS_Description'
)
BEGIN
    EXEC sys.sp_addextendedproperty 
        @name=N'MS_Description', 
        @value=N'Скорректированный бюджет категории после учёта разницы между общим классом точки и классом категории (увеличивается на 10% за каждую единицу разрыва)', 
        @level0type=N'SCHEMA',@level0name=N'dbo', 
        @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', 
        @level2type=N'COLUMN',@level2name=N'ExtendedSum'
    PRINT 'Добавлено описание для ExtendedSum'
END
GO

-- Описание для Comment
IF NOT EXISTS (
    SELECT 1 
    FROM sys.extended_properties 
    WHERE major_id = OBJECT_ID('dbo.SNS_ML_Brand_Recommendations')
    AND minor_id = (SELECT column_id FROM sys.columns WHERE object_id = OBJECT_ID('dbo.SNS_ML_Brand_Recommendations') AND name = 'Comment')
    AND name = 'MS_Description'
)
BEGIN
    EXEC sys.sp_addextendedproperty 
        @name=N'MS_Description', 
        @value=N'Пояснение для сотрудника, почему рассчитано такое значение рекомендации (например: "Регулярное пополнение на 7 дн + Популярно на районе + Турбо-бренд")', 
        @level0type=N'SCHEMA',@level0name=N'dbo', 
        @level1type=N'TABLE',@level1name=N'SNS_ML_Brand_Recommendations', 
        @level2type=N'COLUMN',@level2name=N'Comment'
    PRINT 'Добавлено описание для Comment'
END
GO

PRINT 'Скрипт выполнения добавления полей завершен'
GO
