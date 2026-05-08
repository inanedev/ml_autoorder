IF OBJECT_ID('dbo.SNS_ML_Get_Raw_Data', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SNS_ML_Get_Raw_Data;
GO

/**
 * Хранимая процедура для выгрузки сырых данных о продажах
 * 
 * Назначение:
 *   Агрегация данных о продажах на уровень дня с обогащением
 *   атрибутами точек продаж и расчетом микрорегионов (сетка 3x3 км)
 * 
 * Параметры:
 *   @StartDate DATE - Начальная дата периода выгрузки (включительно)
 *   @EndDate DATE   - Конечная дата периода выгрузки (не включительно)
 * 
 * Возвращаемые колонки:
 *   VisitDate       - Дата визита/продажи
 *   PointID         - Идентификатор точки продаж
 *   CategoryID      - Идентификатор категории товара
 *   BranchID        - Идентификатор филиала/дистрибьютора
 *   PointClass      - Класс точки продаж (из атрибута 602)
 *   PointType       - Тип точки продаж (из атрибута 555)
 *   Lat             - Широта точки продаж
 *   Lon             - Долгота точки продаж
 *   MicroRegionID   - Идентификатор микрорегиона (сетка 3x3 км)
 *   SumRoubles      - Сумма продаж в рублях за день
 * 
 * Логика работы:
 *   1. Формируется справочник активных SKU с категориями
 *   2. Рассчитываются признаки точек продаж с координатами
 *   3. Вычисляется микрорегион на основе сетки 3x3 км
 *   4. Агрегируются продажи по дням с соединением всех справочников
 * 
 * Пример использования:
 *   EXEC dbo.SNS_ML_Get_Raw_Data @StartDate = '2024-01-01', @EndDate = '2024-02-01';
 * 
 * Примечания:
 *   - Размер сетки микрорегиона: 0.027 градуса (~3 км)
 *   - Фильтруются только активные товары и точки
 *   - Координаты преобразуются из строкового формата с заменой запятой на точку
 */
CREATE PROCEDURE [dbo].[SNS_ML_Get_Raw_Data]
    @StartDate DATE,
    @EndDate DATE
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Блок обработки ошибок
    BEGIN TRY
        -- Валидация входных параметров
        IF @StartDate IS NULL OR @EndDate IS NULL
        BEGIN
            RAISERROR('Параметры @StartDate и @EndDate должны быть заполнены', 16, 1);
            RETURN -1;
        END
        
        IF @StartDate > @EndDate
        BEGIN
            RAISERROR('Начальная дата не может быть больше конечной', 16, 1);
            RETURN -1;
        END

        -- Шаг сетки для 3 км: 3 / 111.0 ≈ 0.027 градуса
        DECLARE @GridStep FLOAT = 0.027; 
        
        -- Получаем имя текущей базы данных
        DECLARE @CurrentDBName NVARCHAR(128) = DB_NAME();
        
        -- Находим минимальную дату в текущей базе данных
        DECLARE @MinDateInCurrentDB DATE;
        SELECT @MinDateInCurrentDB = MIN(CAST(orDate AS DATE)) FROM DS_Orders WHERE orType = 1;
        
        -- Таблица для хранения имен баз данных для запроса
        IF OBJECT_ID('tempdb..#TargetDatabases') IS NOT NULL DROP TABLE #TargetDatabases;
        CREATE TABLE #TargetDatabases (DatabaseName NVARCHAR(128));
        
        -- Добавляем текущую базу данных
        INSERT INTO #TargetDatabases (DatabaseName) VALUES (@CurrentDBName);
        
        -- Если начальная дата меньше минимальной в текущей БД, ищем архивные базы
        IF @StartDate < @MinDateInCurrentDB OR @MinDateInCurrentDB IS NULL
        BEGIN
            -- Определяем диапазон годов для проверки
            DECLARE @StartYear INT = YEAR(@StartDate);
            DECLARE @EndYear INT = YEAR(@EndDate);
            DECLARE @CurrentYear INT = YEAR(GETDATE());
            
            -- Создаем таблицу для перебора годов
            IF OBJECT_ID('tempdb..#YearsToCheck') IS NOT NULL DROP TABLE #YearsToCheck;
            CREATE TABLE #YearsToCheck (YearToCheck INT);
            
            -- Заполняем года от StartYear до EndYear (но не больше текущего)
            DECLARE @YearCounter INT = @StartYear;
            WHILE @YearCounter <= @EndYear AND @YearCounter < @CurrentYear
            BEGIN
                INSERT INTO #YearsToCheck (YearToCheck) VALUES (@YearCounter);
                SET @YearCounter = @YearCounter + 1;
            END
            
            -- Для каждого года проверяем существование архивной БД
            DECLARE @ArchiveDBName NVARCHAR(128);
            DECLARE @YearToCheck INT;
            
            DECLARE year_cursor CURSOR LOCAL FAST_FORWARD FOR 
            SELECT YearToCheck FROM #YearsToCheck;
            
            OPEN year_cursor;
            FETCH NEXT FROM year_cursor INTO @YearToCheck;
            
            WHILE @@FETCH_STATUS = 0
            BEGIN
                -- Формируем имя архивной БД: оригинальное_название_ГГГГ
                SET @ArchiveDBName = @CurrentDBName + '_' + CAST(@YearToCheck AS NVARCHAR(4));
                
                -- Проверяем существование БД на сервере
                IF EXISTS (SELECT 1 FROM sys.databases WHERE name = @ArchiveDBName AND state = 0)
                BEGIN
                    -- Проверяем, есть ли данные в нужном диапазоне дат в этой БД
                    DECLARE @SQL NVARCHAR(MAX);
                    DECLARE @HasData INT;
                    
                    SET @SQL = N'SELECT @Result = COUNT(*) FROM [' + @ArchiveDBName + N'].dbo.DS_Orders 
                                 WHERE orType = 1 AND CAST(orDate AS DATE) >= @StartParam AND CAST(orDate AS DATE) < @EndParam';
                    
                    EXEC sp_executesql @SQL, 
                        N'@Result INT OUTPUT, @StartParam DATE, @EndParam DATE',
                        @HasData OUTPUT, @StartDate, @EndDate;
                    
                    IF @HasData > 0
                    BEGIN
                        INSERT INTO #TargetDatabases (DatabaseName) VALUES (@ArchiveDBName);
                    END
                END
                
                FETCH NEXT FROM year_cursor INTO @YearToCheck;
            END
            
            CLOSE year_cursor;
            DEALLOCATE year_cursor;
            
            DROP TABLE #YearsToCheck;
        END
        
        -- Создаем итоговую временную таблицу для результатов
        IF OBJECT_ID('tempdb..#RawDataResult') IS NOT NULL DROP TABLE #RawDataResult;
        CREATE TABLE #RawDataResult (
            VisitDate DATE,
            PointID INT,
            CategoryID INT,
            BranchID INT,
            PointClass NVARCHAR(255),
            PointType NVARCHAR(255),
            Lat FLOAT,
            Lon FLOAT,
            MicroRegionID NVARCHAR(50),
            SumRoubles DECIMAL(18,2)
        );
        
        -- Переменная для динамического SQL и имени БД
        DECLARE @TargetDB NVARCHAR(128);
        DECLARE @DynamicSQL NVARCHAR(MAX);
        
        -- Курсор для перебора всех целевых баз данных
        DECLARE db_cursor CURSOR LOCAL FAST_FORWARD FOR 
        SELECT DatabaseName FROM #TargetDatabases;
        
        OPEN db_cursor;
        FETCH NEXT FROM db_cursor INTO @TargetDB;
        
        WHILE @@FETCH_STATUS = 0
        BEGIN
            -- Формируем динамический SQL для каждой БД
            -- Определяем префикс для таблиц (если это не текущая БД, добавляем имя БД)
            DECLARE @TablePrefix NVARCHAR(256);
            IF @TargetDB = @CurrentDBName
                SET @TablePrefix = N'';
            ELSE
                SET @TablePrefix = N'[' + @TargetDB + N'].dbo.';
            
            SET @DynamicSQL = N'
            -- Временные таблицы для каждой БД
            IF OBJECT_ID(''tempdb..#ItemMap_' + @TargetDB + N''') IS NOT NULL DROP TABLE #ItemMap_' + @TargetDB + N';
            SELECT 
                CAST(i.iid AS INT) as iid, 
                CAST(i.itID AS INT) AS CategoryID
            INTO #ItemMap_' + @TargetDB + N' 
            FROM ' + @TablePrefix + N'DS_ITEMS i 
            WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;
            
            CREATE CLUSTERED INDEX IX_ItemMap_' + @TargetDB + N'_iid ON #ItemMap_' + @TargetDB + N' (iid);

            IF OBJECT_ID(''tempdb..#PointFeatures_' + @TargetDB + N''') IS NOT NULL DROP TABLE #PointFeatures_' + @TargetDB + N';
            SELECT 
                f.fid AS PointID, 
                f.distid AS BranchID, 
                
                -- Координаты (приводим к float)
                ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) AS Lat,
                ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) AS Lon,
                
                -- Атрибуты
                ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), ''Unknown'') AS PointClass,
                ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), ''Unknown'') AS PointType,
                
                -- Расчет MicroRegionID (Сетка 3x3 км)
                CAST(
                    FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) / ' + CAST(@GridStep AS NVARCHAR(20)) + N') * ' + CAST(@GridStep AS NVARCHAR(20)) + N' 
                    AS VARCHAR(20)
                ) + ''_'' + 
                CAST(
                    FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) / ' + CAST(@GridStep AS NVARCHAR(20)) + N') * ' + CAST(@GridStep AS NVARCHAR(20)) + N' 
                    AS VARCHAR(20)
                ) AS MicroRegionID

            INTO #PointFeatures_' + @TargetDB + N' 
            FROM ' + @TablePrefix + N'ds_faces f 
            LEFT JOIN ' + @TablePrefix + N'ds_facesattributes fa ON f.fid = fa.fid AND fa.activeflag = 1 
            WHERE f.ftype = 1 AND f.factiveflag = 1 
            GROUP BY f.fid, f.distid;

            CREATE CLUSTERED INDEX IX_PF_' + @TargetDB + N'_PointID ON #PointFeatures_' + @TargetDB + N' (PointID);

            -- Вставка агрегированных данных в итоговую таблицу
            INSERT INTO #RawDataResult (VisitDate, PointID, CategoryID, BranchID, PointClass, PointType, Lat, Lon, MicroRegionID, SumRoubles)
            SELECT 
                CAST(o.orDate AS DATE) AS VisitDate, 
                o.mfID AS PointID, 
                m.CategoryID,
                pf.BranchID,
                pf.PointClass,
                pf.PointType,
                pf.Lat,
                pf.Lon,
                pf.MicroRegionID,
                SUM(ISNULL(oi.SumRoubles, 0)) AS SumRoubles
            FROM ' + @TablePrefix + N'DS_Orders o 
            INNER JOIN ' + @TablePrefix + N'DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
            INNER JOIN #ItemMap_' + @TargetDB + N' m ON CAST(oi.iID AS INT) = m.iid 
            INNER JOIN #PointFeatures_' + @TargetDB + N' pf ON o.mfID = pf.PointID
            WHERE o.orType = 1 
              AND o.orDate >= @StartDateParam
              AND o.orDate < @EndDateParam
            GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID, pf.BranchID, pf.PointClass, pf.PointType, pf.Lat, pf.Lon, pf.MicroRegionID;

            -- Очистка временных таблиц для этой БД
            DROP TABLE #ItemMap_' + @TargetDB + N';
            DROP TABLE #PointFeatures_' + @TargetDB + N';
            ';
            
            EXEC sp_executesql @DynamicSQL, 
                N'@StartDateParam DATE, @EndDateParam DATE',
                @StartDate, @EndDate;
            
            FETCH NEXT FROM db_cursor INTO @TargetDB;
        END
        
        CLOSE db_cursor;
        DEALLOCATE db_cursor;
        
        DROP TABLE #TargetDatabases;
        
        -- Возвращаем итоговый результат
        SELECT 
            VisitDate,
            PointID,
            CategoryID,
            BranchID,
            PointClass,
            PointType,
            Lat,
            Lon,
            MicroRegionID,
            SumRoubles
        FROM #RawDataResult
        ORDER BY VisitDate, PointID, CategoryID;
        
        DROP TABLE #RawDataResult;
        
    END TRY
    BEGIN CATCH
        -- Обработка ошибок
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        DECLARE @ErrorSeverity INT = ERROR_SEVERITY();
        DECLARE @ErrorState INT = ERROR_STATE();
        DECLARE @ErrorLine INT = ERROR_LINE();
        
        -- Логирование ошибки
        PRINT 'Ошибка в процедуре SNS_ML_Get_Raw_Data:';
        PRINT 'Сообщение: ' + @ErrorMessage;
        PRINT 'Строка: ' + CAST(@ErrorLine AS NVARCHAR(10));
        
        -- Очистка временных таблиц если они существуют
        IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
        IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
        IF OBJECT_ID('tempdb..#TargetDatabases') IS NOT NULL DROP TABLE #TargetDatabases;
        IF OBJECT_ID('tempdb..#YearsToCheck') IS NOT NULL DROP TABLE #YearsToCheck;
        IF OBJECT_ID('tempdb..#RawDataResult') IS NOT NULL DROP TABLE #RawDataResult;
        
        -- Закрываем курсоры если они открыты
        IF CURSOR_STATUS('local', 'year_cursor') >= 0
        BEGIN
            CLOSE year_cursor;
            DEALLOCATE year_cursor;
        END
        
        IF CURSOR_STATUS('local', 'db_cursor') >= 0
        BEGIN
            CLOSE db_cursor;
            DEALLOCATE db_cursor;
        END
        
        -- Проброс ошибки дальше
        RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
        RETURN -1;
    END CATCH
END;
GO