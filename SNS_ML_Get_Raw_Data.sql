IF OBJECT_ID('dbo.SNS_ML_Get_Raw_Data', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SNS_ML_Get_Raw_Data;
GO

/**
 * Хранимая процедура для выгрузки сырых данных о продажах с учетом визитов
 * 
 * Назначение:
 *   Агрегация данных о продажах на уровень дня с обогащением
 *   атрибутами точек продаж и расчетом микрорегионов (сетка 3x3 км).
 *   Включает визиты без продаж с полной матрицей категорий.
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
 *   SumRoubles      - Сумма продаж в рублях за день (0 если визит без продаж данной категории)
 * 
 * Логика работы:
 *   1. Формируется матрица всех категорий, которые продавались за последние 3 месяца от @StartDate
 *   2. Из таблицы DS_merPointsVisits извлекаются все визиты (включая визиты без продаж)
 *   3. Для каждого визита строится полная матрица категорий
 *   4. Для проданных категорий указывается сумма, для непроданных - 0
 *   5. Рассчитываются признаки точек продаж с координатами и микрорегионами
 * 
 * Пример использования:
 *   EXEC dbo.SNS_ML_Get_Raw_Data @StartDate = '2024-01-01', @EndDate = '2024-02-01';
 * 
 * Примечания:
 *   - Размер сетки микрорегиона: 0.027 градуса (~3 км)
 *   - Фильтруются только активные товары и точки
 *   - Координаты преобразуются из строкового формата с заменой запятой на точку
 *   - Матрица категорий формируется за период: @StartDate минус 3 месяца до @EndDate
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
        
        -- Расчет периода для матрицы категорий (последние 3 месяца до @StartDate)
        DECLARE @CategoryMatrixStart DATE = DATEADD(MONTH, -3, @StartDate);
        
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
        IF @CategoryMatrixStart < @MinDateInCurrentDB OR @MinDateInCurrentDB IS NULL
        BEGIN
            -- Определяем диапазон годов для проверки
            DECLARE @StartYear INT = YEAR(@CategoryMatrixStart);
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
            
            DECLARE year_cursor CURSOR FAST_FORWARD FOR 
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
                        @HasData OUTPUT, @CategoryMatrixStart, @EndDate;
                    
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
        DECLARE db_cursor CURSOR FAST_FORWARD FOR 
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
            
            -- 1. Формируем матрицу всех категорий, которые продавались за последние 3 месяца
            IF OBJECT_ID(''tempdb..#CategoryMatrix_' + @TargetDB + N''') IS NOT NULL DROP TABLE #CategoryMatrix_' + @TargetDB + N';
            SELECT DISTINCT 
                CAST(i.itID AS INT) AS CategoryID
            INTO #CategoryMatrix_' + @TargetDB + N' 
            FROM ' + @TablePrefix + N'DS_ITEMS i 
            INNER JOIN ' + @TablePrefix + N'DS_Orders_Items oi ON CAST(i.iid AS INT) = CAST(oi.iID AS INT)
            INNER JOIN ' + @TablePrefix + N'DS_Orders o ON oi.MasterFID = o.MasterFID AND oi.orID = o.orID
            WHERE i.activeFlag = 1 
              AND i.itID IS NOT NULL
              AND o.orType = 1
              AND CAST(o.orDate AS DATE) >= ''' + CONVERT(NVARCHAR(10), @CategoryMatrixStart, 120) + N'''
              AND CAST(o.orDate AS DATE) < ''' + CONVERT(NVARCHAR(10), @EndDate, 120) + N''';
            
            -- Создаем индекс с фиксированным именем для CategoryMatrix
            DECLARE @idx_name_catmatrix NVARCHAR(256) = 'IX_CategoryMatrix_' + REPLACE(@TargetDB, '[', '') + REPLACE(']', '') + '_CategoryID';
            DECLARE @create_idx_catmatrix NVARCHAR(MAX) = 
                'CREATE CLUSTERED INDEX ' + QUOTENAME(@idx_name_catmatrix) + 
                ' ON #CategoryMatrix_' + @TargetDB + N' (CategoryID);';
            EXEC sp_executesql @create_idx_catmatrix;
            
            -- 2. Справочник активных SKU с категориями
            IF OBJECT_ID(''tempdb..#ItemMap_' + @TargetDB + N''') IS NOT NULL DROP TABLE #ItemMap_' + @TargetDB + N';
            SELECT 
                CAST(i.iid AS INT) as iid, 
                CAST(i.itID AS INT) AS CategoryID
            INTO #ItemMap_' + @TargetDB + N' 
            FROM ' + @TablePrefix + N'DS_ITEMS i 
            WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;
            
            -- Создаем индекс с фиксированным именем для ItemMap
            DECLARE @idx_name_itemmap NVARCHAR(256) = 'IX_ItemMap_' + REPLACE(@TargetDB, '[', '') + REPLACE(']', '') + '_iid';
            DECLARE @create_idx_itemmap NVARCHAR(MAX) = 
                'CREATE CLUSTERED INDEX ' + QUOTENAME(@idx_name_itemmap) + 
                ' ON #ItemMap_' + @TargetDB + N' (iid);';
            EXEC sp_executesql @create_idx_itemmap;

            -- 3. Признаки точек продаж с координатами
            IF OBJECT_ID(''tempdb..#PointFeatures_' + @TargetDB + N''') IS NOT NULL DROP TABLE #PointFeatures_' + @TargetDB + N';
            SELECT 
                f.fid AS PointID, 
                f.distid AS BranchID, 
                
                -- Координаты (приводим к float)
                ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) AS Lat,
                ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) AS Lon,
                
                -- Атрибуты
                ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), ''Unknown'') AS PointClass,
                ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), ''Unknown'') AS PointType,
                
                -- Расчет MicroRegionID (Сетка 3x3 км)
                CAST(
                    FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) / ' + CAST(@GridStep AS NVARCHAR(20)) + N') * ' + CAST(@GridStep AS NVARCHAR(20)) + N' 
                    AS VARCHAR(20)
                ) + ''_'' + 
                CAST(
                    FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN CAST(REPLACE(REPLACE(fa.attrtext, '','', ''.''), '' '', '''') AS FLOAT) END), 0) / ' + CAST(@GridStep AS NVARCHAR(20)) + N') * ' + CAST(@GridStep AS NVARCHAR(20)) + N' 
                    AS VARCHAR(20)
                ) AS MicroRegionID

            INTO #PointFeatures_' + @TargetDB + N' 
            FROM ' + @TablePrefix + N'ds_faces f 
            LEFT JOIN ' + @TablePrefix + N'ds_facesattributes fa ON f.fid = fa.fid AND fa.activeflag = 1 
            WHERE f.ftype = 1 AND f.factiveflag = 1 
            GROUP BY f.fid, f.distid;

            -- Создаем индекс с фиксированным именем для PointFeatures
            DECLARE @idx_name_pf NVARCHAR(256) = 'IX_PF_' + REPLACE(@TargetDB, '[', '') + REPLACE(']', '') + '_PointID';
            DECLARE @create_idx_pf NVARCHAR(MAX) = 
                'CREATE CLUSTERED INDEX ' + QUOTENAME(@idx_name_pf) + 
                ' ON #PointFeatures_' + @TargetDB + N' (PointID);';
            EXEC sp_executesql @create_idx_pf;

            -- 4. Извлекаем все визиты из DS_merPointsVisits
            IF OBJECT_ID(''tempdb..#AllVisits_' + @TargetDB + N''') IS NOT NULL DROP TABLE #AllVisits_' + @TargetDB + N';
            SELECT 
                CAST(mpv.vdate AS DATE) AS VisitDate,
                CAST(mpv.fid AS INT) AS PointID
            INTO #AllVisits_' + @TargetDB + N'
            FROM ' + @TablePrefix + N'DS_merPointsVisits mpv
            WHERE CAST(mpv.vdate AS DATE) >= @StartDateParam
              AND CAST(mpv.vdate AS DATE) < @EndDateParam
            GROUP BY CAST(mpv.vdate AS DATE), CAST(mpv.fid AS INT);
            
            -- Создаем индекс с фиксированным именем (временные таблицы уникальны в сессии)
            DECLARE @idx_name_allvisits NVARCHAR(256) = 'IX_AllVisits_' + REPLACE(@TargetDB, '[', '') + REPLACE(']', '') + '_VisitPoint';
            DECLARE @create_idx_allvisits NVARCHAR(MAX) = 
                'CREATE CLUSTERED INDEX ' + QUOTENAME(@idx_name_allvisits) + 
                ' ON #AllVisits_' + @TargetDB + N' (VisitDate, PointID);';
            EXEC sp_executesql @create_idx_allvisits;

            -- 5. Агрегируем продажи по дням, точкам и категориям
            IF OBJECT_ID(''tempdb..#SalesAgg_' + @TargetDB + N''') IS NOT NULL DROP TABLE #SalesAgg_' + @TargetDB + N';
            SELECT 
                CAST(o.orDate AS DATE) AS VisitDate, 
                o.mfID AS PointID, 
                m.CategoryID,
                SUM(ISNULL(oi.SumRoubles, 0)) AS SumRoubles
            INTO #SalesAgg_' + @TargetDB + N'
            FROM ' + @TablePrefix + N'DS_Orders o 
            INNER JOIN ' + @TablePrefix + N'DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
            INNER JOIN #ItemMap_' + @TargetDB + N' m ON CAST(oi.iID AS INT) = m.iid 
            WHERE o.orType = 1 
              AND o.orDate >= @StartDateParam
              AND o.orDate < @EndDateParam
            GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID;
            
            -- Создаем индекс с фиксированным именем для SalesAgg
            DECLARE @idx_name_salesagg NVARCHAR(256) = 'IX_SalesAgg_' + REPLACE(@TargetDB, '[', '') + REPLACE(']', '') + '_VisitPointCat';
            DECLARE @create_idx_salesagg NVARCHAR(MAX) = 
                'CREATE CLUSTERED INDEX ' + QUOTENAME(@idx_name_salesagg) + 
                ' ON #SalesAgg_' + @TargetDB + N' (VisitDate, PointID, CategoryID);';
            EXEC sp_executesql @create_idx_salesagg;

            -- 6. Строим полную матрицу: все визиты × все категории
            -- И соединяем с продажами (LEFT JOIN) и атрибутами точек
            INSERT INTO #RawDataResult (VisitDate, PointID, CategoryID, BranchID, PointClass, PointType, Lat, Lon, MicroRegionID, SumRoubles)
            SELECT 
                av.VisitDate,
                av.PointID,
                cm.CategoryID,
                pf.BranchID,
                pf.PointClass,
                pf.PointType,
                pf.Lat,
                pf.Lon,
                pf.MicroRegionID,
                ISNULL(sa.SumRoubles, 0) AS SumRoubles
            FROM #AllVisits_' + @TargetDB + N' av
            CROSS JOIN #CategoryMatrix_' + @TargetDB + N' cm
            INNER JOIN #PointFeatures_' + @TargetDB + N' pf ON av.PointID = pf.PointID
            LEFT JOIN #SalesAgg_' + @TargetDB + N' sa 
                ON av.VisitDate = sa.VisitDate 
                AND av.PointID = sa.PointID 
                AND cm.CategoryID = sa.CategoryID;

            -- Очистка временных таблиц для этой БД
            DROP TABLE #CategoryMatrix_' + @TargetDB + N';
            DROP TABLE #ItemMap_' + @TargetDB + N';
            DROP TABLE #PointFeatures_' + @TargetDB + N';
            DROP TABLE #AllVisits_' + @TargetDB + N';
            DROP TABLE #SalesAgg_' + @TargetDB + N';
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
        IF OBJECT_ID('tempdb..#CategoryMatrix') IS NOT NULL DROP TABLE #CategoryMatrix;
        IF OBJECT_ID('tempdb..#AllVisits') IS NOT NULL DROP TABLE #AllVisits;
        IF OBJECT_ID('tempdb..#SalesAgg') IS NOT NULL DROP TABLE #SalesAgg;
        
        -- Закрываем курсоры если они существуют и открыты
        BEGIN TRY
            IF CURSOR_STATUS('global', 'year_cursor') > 0
            BEGIN
                CLOSE year_cursor;
                DEALLOCATE year_cursor;
            END
        END TRY
        BEGIN CATCH
            -- Игнорируем ошибки при закрытии курсора
        END CATCH
        
        BEGIN TRY
            IF CURSOR_STATUS('global', 'db_cursor') > 0
            BEGIN
                CLOSE db_cursor;
                DEALLOCATE db_cursor;
            END
        END TRY
        BEGIN CATCH
            -- Игнорируем ошибки при закрытии курсора
        END CATCH
        
        -- Проброс ошибки дальше
        RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
        RETURN -1;
    END CATCH
END;
GO
