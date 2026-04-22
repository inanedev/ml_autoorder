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

        -- 1. Справочник SKU -> Категория
        -- Фильтруем только активные товары
        IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
        SELECT 
            CAST(i.iid AS INT) as iid, 
            CAST(i.itID AS INT) AS CategoryID
        INTO #ItemMap 
        FROM DS_ITEMS i 
        WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;
        
        CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);

        -- 2. Признаки точек + Расчет Микрорегиона (3x3 км)
        IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
        SELECT 
            f.fid AS PointID, 
            f.distid AS BranchID, 
            
            -- Координаты (приводим к float)
            ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) AS Lat,
            ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) AS Lon,
            
            -- Атрибуты
            ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), 'Unknown') AS PointClass, -- Класс точки
            ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), 'Unknown') AS PointType,  -- Тип точки (Атрибут 555)
            
            -- Расчет MicroRegionID (Сетка 3x3 км)
            -- Формула: FLOOR(Coord / Step) * Step
            -- Приводим к строке формата "Lat_Lon" для удобного использования как ID
            CAST(
                FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) / @GridStep) * @GridStep 
                AS VARCHAR(20)
            ) + '_' + 
            CAST(
                FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) / @GridStep) * @GridStep 
                AS VARCHAR(20)
            ) AS MicroRegionID

        INTO #PointFeatures 
        FROM ds_faces f 
        LEFT JOIN ds_facesattributes fa ON f.fid = fa.fid AND fa.activeflag = 1 
        WHERE f.ftype = 1 AND f.factiveflag = 1 
        GROUP BY f.fid, f.distid;

        CREATE CLUSTERED INDEX IX_PF_PointID ON #PointFeatures (PointID);

        -- 3. Выгрузка сырых данных (Факт продаж)
        -- Агрегация только на уровень ДНЯ (если в день несколько накладных)
        SELECT 
            CAST(o.orDate AS DATE) AS VisitDate, 
            o.mfID AS PointID, 
            m.CategoryID,
            pf.BranchID,
            pf.PointClass,
            pf.PointType,      -- Добавлен атрибут 555
            pf.Lat,
            pf.Lon,
            pf.MicroRegionID,  -- Добавлен микрорегион 3x3 км
            SUM(ISNULL(oi.SumRoubles, 0)) AS SumRoubles -- Сумма за день
        FROM DS_Orders o 
        INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
        INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid 
        INNER JOIN #PointFeatures pf ON o.mfID = pf.PointID
        WHERE o.orType = 1 
          AND o.orDate >= @StartDate
          AND o.orDate < @EndDate
        GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID, pf.BranchID, pf.PointClass, pf.PointType, pf.Lat, pf.Lon, pf.MicroRegionID;

        -- Очистка временных таблиц
        DROP TABLE #ItemMap;
        DROP TABLE #PointFeatures;
        
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
        
        -- Вызов временных таблиц если они существуют
        IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
        IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
        
        -- Проброс ошибки дальше
        RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
        RETURN -1;
    END CATCH
END;
GO
