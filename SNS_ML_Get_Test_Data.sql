IF OBJECT_ID('dbo.SNS_ML_Get_Test_Data', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SNS_ML_Get_Test_Data;
GO

/**
 * Хранимая процедура для выгрузки тестовых данных для ML модели
 * 
 * Назначение:
 *   Формирование полного набора признаков для прогнозирования продаж
 *   на уровне точка x категория x дата с использованием исторических данных
 * 
 * Параметры:
 *   @TargetDate DATE - Дата, на которую собираются фичи (прогноз будет для этой даты)
 * 
 * Возвращаемые колонки:
 *   === Базовые данные ===
 *   VisitDate                     - Дата прогноза (TargetDate)
 *   PointID                       - Идентификатор точки продаж
 *   CategoryID                    - Идентификатор категории товара
 *   BranchID                      - Идентификатор филиала/дистрибьютора
 *   PointClass                    - Класс точки продаж (из атрибута 602)
 *   PointType                     - Тип точки продаж (из атрибута 555)
 *   Lat                           - Широта точек продаж
 *   Lon                           - Долгота точек продаж
 *   MicroRegionID                 - Идентификатор микрорегиона (сетка 3x3 км)
 *   
 *   === Календарные фичи ===
 *   day_of_week                   - День недели (0-понедельник, 6-воскресенье)
 *   is_weekend                    - Флаг выходного дня
 *   is_monday                     - Флаг понедельника
 *   is_friday                     - Флаг пятницы
 *   is_saturday                   - Флаг субботы
 *   is_sunday                     - Флаг воскресенья
 *   is_holiday                    - Флаг праздника России
 *   is_pre_holiday                - Флаг предпраздничного дня
 *   is_post_holiday               - Флаг дня после праздника
 *   month                         - Месяц года
 *   quarter                       - Квартал года
 *   week_of_year                  - Неделя года
 *   is_month_start                - Флаг начала месяца
 *   is_month_end                  - Флаг конца месяца
 *   day_of_month                  - День месяца
 *   day_of_year                   - День года
 *   days_to_holiday               - Дней до ближайшего праздника
 *   days_from_holiday             - Дней от последнего праздника
 *   
 *   === Фичи истории заказов ===
 *   Days_Since_Last_Order_Category - Дней назад точка брала эту категорию
 *   Days_Since_Last_Order_Total    - Дней назад был любой заказ от точки
 *   Average_Interval_Category      - Средний интервал между закупками категории
 *   
 *   === Фичи продаж ===
 *   Prev_Order_Amount_Category    - Сумма предыдущего заказа по категории
 *   SMA_3_Category                - Скользящее среднее за 3 дня по категории
 *   SMA_7_Category                - Скользящее среднее за 7 дней по категории
 *   SMA_30_Category               - Скользящее среднее за 30 дней по категории
 *   Momentum_Category             - Отношение среднего чека за неделю к месяцу
 *   StdDev_Category               - Скользящее стандартное отклонение
 * 
 * Логика работы:
 *   1. Выбираются все активные точки продаж
 *   2. Для каждой точки формируются признаки из SNS_ML_Get_Raw_Data
 *   3. Добавляются календарные фичи для TargetDate
 *   4. Рассчитываются фичи истории заказов по данным ДО TargetDate
 *   5. Рассчитываются фичи продаж по данным ДО TargetDate
 *   6. SumRoubles НЕ включается в результат (это целевая переменная для предсказания)
 * 
 * Пример использования:
 *   EXEC dbo.SNS_ML_Get_Test_Data @TargetDate = '2024-01-15';
 * 
 * Примечания:
 *   - Размер сетки микрорегиона: 0.027 градуса (~3 км)
 *   - Фильтруются только активные товары и точки
 *   - Координаты преобразуются из строкового формата с заменой запятой на точку
 *   - Все лаговые фичи считаются строго по данным до @TargetDate (не включая саму дату)
 */
CREATE PROCEDURE [dbo].[SNS_ML_Get_Test_Data]
    @TargetDate DATE
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Блок обработки ошибок
    BEGIN TRY
        -- Валидация входных параметров
        IF @TargetDate IS NULL
        BEGIN
            RAISERROR('Параметр @TargetDate должен быть заполнен', 16, 1);
            RETURN -1;
        END

        -- Установка первого дня недели = понедельник (как в Python .dt.dayofweek)
        SET DATEFIRST 1;

        -- Шаг сетки для 3 км: 3 / 111.0 ≈ 0.027 градуса
        DECLARE @GridStep FLOAT = 0.027; 

        -- 1. Справочник SKU -> Категория (активные товары)
        IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
        SELECT 
            CAST(i.iid AS INT) as iid, 
            CAST(i.itID AS INT) AS CategoryID
        INTO #ItemMap 
        FROM DS_ITEMS i 
        WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;
        
        CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);

        -- 2. Признаки активных точек + Расчет Микрорегиона (3x3 км)
        IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
        SELECT 
            f.fid AS PointID, 
            f.distid AS BranchID, 
            
            -- Координаты (приводим к float)
            ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) AS Lat,
            ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) AS Lon,
            
            -- Атрибуты
            ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), 'Unknown') AS PointClass,
            ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), 'Unknown') AS PointType,
            
            -- Расчет MicroRegionID (Сетка 3x3 км)
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

        -- 3. Список всех активных категорий
        IF OBJECT_ID('tempdb..#Categories') IS NOT NULL DROP TABLE #Categories;
        SELECT DISTINCT CategoryID INTO #Categories FROM #ItemMap;
        
        CREATE CLUSTERED INDEX IX_Cat_CategoryID ON #Categories (CategoryID);

        -- 4. Полный грид: все активные точки x все категории для одной даты
        IF OBJECT_ID('tempdb..#FullGrid') IS NOT NULL DROP TABLE #FullGrid;
        SELECT 
            @TargetDate AS VisitDate,
            pf.PointID,
            c.CategoryID,
            pf.BranchID,
            pf.PointClass,
            pf.PointType,
            pf.Lat,
            pf.Lon,
            pf.MicroRegionID
        INTO #FullGrid
        FROM #PointFeatures pf
        CROSS JOIN #Categories c;
        
        CREATE CLUSTERED INDEX IX_FG_DatePointCat ON #FullGrid (VisitDate, PointID, CategoryID);

        -- 5. История продаж для расчета лаговых фичей (строго до @TargetDate)
        IF OBJECT_ID('tempdb..#SalesHistory') IS NOT NULL DROP TABLE #SalesHistory;
        SELECT 
            CAST(o.orDate AS DATE) AS VisitDate, 
            o.mfID AS PointID, 
            m.CategoryID,
            SUM(ISNULL(oi.SumRoubles, 0)) AS SumRoubles
        INTO #SalesHistory
        FROM DS_Orders o 
        INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
        INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid 
        WHERE o.orType = 1 
          AND o.orDate < @TargetDate
        GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID;
        
        CREATE CLUSTERED INDEX IX_SH_DatePointCat ON #SalesHistory (VisitDate, PointID, CategoryID);
        CREATE NONCLUSTERED INDEX IX_SH_PointCat ON #SalesHistory (PointID, CategoryID, VisitDate);

        -- 6. История заказов по точкам (для Days_Since_Last_Order_Total)
        IF OBJECT_ID('tempdb..#OrderHistory') IS NOT NULL DROP TABLE #OrderHistory;
        SELECT 
            CAST(o.orDate AS DATE) AS VisitDate, 
            o.mfID AS PointID
        INTO #OrderHistory
        FROM DS_Orders o 
        WHERE o.orType = 1 
          AND o.orDate < @TargetDate
        GROUP BY CAST(o.orDate AS DATE), o.mfID;
        
        CREATE CLUSTERED INDEX IX_OH_DatePoint ON #OrderHistory (VisitDate, PointID);
        CREATE NONCLUSTERED INDEX IX_OH_Point ON #OrderHistory (PointID, VisitDate);

        -- 7. Основной результат с базовыми данными и календарными фичами (без SumRoubles!)
        IF OBJECT_ID('tempdb..#ResultBase') IS NOT NULL DROP TABLE #ResultBase;
        SELECT 
            fg.VisitDate,
            fg.PointID,
            fg.CategoryID,
            fg.BranchID,
            fg.PointClass,
            fg.PointType,
            fg.Lat,
            fg.Lon,
            fg.MicroRegionID,
            
            -- Календарные фичи
            (DATEPART(WEEKDAY, fg.VisitDate) - 1) % 7 AS day_of_week,  -- 0=Monday, 6=Sunday (при DATEFIRST 1)
            CASE WHEN (DATEPART(WEEKDAY, fg.VisitDate) - 1) % 7 IN (5, 6) THEN 1 ELSE 0 END AS is_weekend,
            CASE WHEN (DATEPART(WEEKDAY, fg.VisitDate) - 1) % 7 = 0 THEN 1 ELSE 0 END AS is_monday,
            CASE WHEN (DATEPART(WEEKDAY, fg.VisitDate) - 1) % 7 = 4 THEN 1 ELSE 0 END AS is_friday,
            CASE WHEN (DATEPART(WEEKDAY, fg.VisitDate) - 1) % 7 = 5 THEN 1 ELSE 0 END AS is_saturday,
            CASE WHEN (DATEPART(WEEKDAY, fg.VisitDate) - 1) % 7 = 6 THEN 1 ELSE 0 END AS is_sunday,
            
            -- Праздники России (фиксированные даты)
            CASE WHEN (MONTH(fg.VisitDate) = 1 AND DAY(fg.VisitDate) IN (1,2,3,4,5,6,7,8))
                      OR (MONTH(fg.VisitDate) = 2 AND DAY(fg.VisitDate) = 23)
                      OR (MONTH(fg.VisitDate) = 3 AND DAY(fg.VisitDate) = 8)
                      OR (MONTH(fg.VisitDate) = 5 AND DAY(fg.VisitDate) IN (1,9))
                      OR (MONTH(fg.VisitDate) = 6 AND DAY(fg.VisitDate) = 12)
                      OR (MONTH(fg.VisitDate) = 11 AND DAY(fg.VisitDate) = 4)
                 THEN 1 ELSE 0 END AS is_holiday,
            
            -- Предпраздничный день
            CASE WHEN (MONTH(DATEADD(DAY, 1, fg.VisitDate)) = 1 AND DAY(DATEADD(DAY, 1, fg.VisitDate)) IN (1,2,3,4,5,6,7,8))
                      OR (MONTH(DATEADD(DAY, 1, fg.VisitDate)) = 2 AND DAY(DATEADD(DAY, 1, fg.VisitDate)) = 23)
                      OR (MONTH(DATEADD(DAY, 1, fg.VisitDate)) = 3 AND DAY(DATEADD(DAY, 1, fg.VisitDate)) = 8)
                      OR (MONTH(DATEADD(DAY, 1, fg.VisitDate)) = 5 AND DAY(DATEADD(DAY, 1, fg.VisitDate)) IN (1,9))
                      OR (MONTH(DATEADD(DAY, 1, fg.VisitDate)) = 6 AND DAY(DATEADD(DAY, 1, fg.VisitDate)) = 12)
                      OR (MONTH(DATEADD(DAY, 1, fg.VisitDate)) = 11 AND DAY(DATEADD(DAY, 1, fg.VisitDate)) = 4)
                      OR (DATEPART(WEEKDAY, DATEADD(DAY, 1, fg.VisitDate)) - 1) % 7 IN (5, 6)
                 THEN 1 ELSE 0 END AS is_pre_holiday,
            
            -- Постпраздничный день
            CASE WHEN (MONTH(DATEADD(DAY, -1, fg.VisitDate)) = 1 AND DAY(DATEADD(DAY, -1, fg.VisitDate)) IN (1,2,3,4,5,6,7,8))
                      OR (MONTH(DATEADD(DAY, -1, fg.VisitDate)) = 2 AND DAY(DATEADD(DAY, -1, fg.VisitDate)) = 23)
                      OR (MONTH(DATEADD(DAY, -1, fg.VisitDate)) = 3 AND DAY(DATEADD(DAY, -1, fg.VisitDate)) = 8)
                      OR (MONTH(DATEADD(DAY, -1, fg.VisitDate)) = 5 AND DAY(DATEADD(DAY, -1, fg.VisitDate)) IN (1,9))
                      OR (MONTH(DATEADD(DAY, -1, fg.VisitDate)) = 6 AND DAY(DATEADD(DAY, -1, fg.VisitDate)) = 12)
                      OR (MONTH(DATEADD(DAY, -1, fg.VisitDate)) = 11 AND DAY(DATEADD(DAY, -1, fg.VisitDate)) = 4)
                      OR (DATEPART(WEEKDAY, DATEADD(DAY, -1, fg.VisitDate)) - 1) % 7 IN (5, 6)
                 THEN 1 ELSE 0 END AS is_post_holiday,
            
            -- Дополнительные календарные фичи
            MONTH(fg.VisitDate) AS month,
            DATEPART(QUARTER, fg.VisitDate) AS quarter,
            DATEPART(WEEK, fg.VisitDate) AS week_of_year,
            CASE WHEN DAY(fg.VisitDate) = 1 THEN 1 ELSE 0 END AS is_month_start,
            CASE WHEN DAY(EOMONTH(fg.VisitDate)) = DAY(fg.VisitDate) THEN 1 ELSE 0 END AS is_month_end,
            DAY(fg.VisitDate) AS day_of_month,
            DATEPART(DAYOFYEAR, fg.VisitDate) AS day_of_year,
            
            -- Дни до ближайшего праздника (расчет через подзапрос)
            ISNULL((
                SELECT TOP 1 DATEDIFF(DAY, fg.VisitDate, h.HolidayDate)
                FROM (
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 1) AS HolidayDate UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 2) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 3) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 4) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 5) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 6) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 7) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 2, 23) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 3, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 5, 1) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 5, 9) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 6, 12) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 11, 4) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 1) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 2) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 3) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 4) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 5) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 6) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 7) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 1, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 2, 23) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 3, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 5, 1) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 5, 9) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 6, 12) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)+1, 11, 4)
                ) h
                WHERE h.HolidayDate > fg.VisitDate
                ORDER BY h.HolidayDate ASC
            ), 365) AS days_to_holiday,
            
            -- Дни от последнего праздника (расчет через подзапрос)
            ISNULL((
                SELECT TOP 1 DATEDIFF(DAY, h.HolidayDate, fg.VisitDate)
                FROM (
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 1) AS HolidayDate UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 2) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 3) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 4) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 5) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 6) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 7) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 1, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 2, 23) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 3, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 5, 1) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 5, 9) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 6, 12) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate), 11, 4) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 1) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 2) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 3) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 4) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 5) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 6) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 7) UNION ALL SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 1, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 2, 23) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 3, 8) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 5, 1) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 5, 9) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 6, 12) UNION ALL
                    SELECT DATEFROMPARTS(YEAR(fg.VisitDate)-1, 11, 4)
                ) h
                WHERE h.HolidayDate < fg.VisitDate
                ORDER BY h.HolidayDate DESC
            ), 365) AS days_from_holiday

        INTO #ResultBase
        FROM #FullGrid fg;
        
        CREATE CLUSTERED INDEX IX_RB_DatePointCat ON #ResultBase (VisitDate, PointID, CategoryID);

        -- 8. Расчет Days_Since_Last_Order_Category
        IF OBJECT_ID('tempdb..#DaysSinceCategory') IS NOT NULL DROP TABLE #DaysSinceCategory;
        SELECT 
            rb.VisitDate,
            rb.PointID,
            rb.CategoryID,
            DATEDIFF(DAY, MAX(sh.VisitDate), rb.VisitDate) AS Days_Since_Last_Order_Category
        INTO #DaysSinceCategory
        FROM #ResultBase rb
        OUTER APPLY (
            SELECT TOP 1 sh2.VisitDate
            FROM #SalesHistory sh2
            WHERE sh2.PointID = rb.PointID 
              AND sh2.CategoryID = rb.CategoryID
              AND sh2.VisitDate < rb.VisitDate
            ORDER BY sh2.VisitDate DESC
        ) sh
        GROUP BY rb.VisitDate, rb.PointID, rb.CategoryID, sh.VisitDate;
        
        CREATE CLUSTERED INDEX IX_DSC_DatePointCat ON #DaysSinceCategory (VisitDate, PointID, CategoryID);

        -- 9. Расчет Days_Since_Last_Order_Total
        IF OBJECT_ID('tempdb..#DaysSinceTotal') IS NOT NULL DROP TABLE #DaysSinceTotal;
        SELECT 
            rb.VisitDate,
            rb.PointID,
            rb.CategoryID,
            DATEDIFF(DAY, MAX(oh.VisitDate), rb.VisitDate) AS Days_Since_Last_Order_Total
        INTO #DaysSinceTotal
        FROM #ResultBase rb
        OUTER APPLY (
            SELECT TOP 1 oh2.VisitDate
            FROM #OrderHistory oh2
            WHERE oh2.PointID = rb.PointID
              AND oh2.VisitDate < rb.VisitDate
            ORDER BY oh2.VisitDate DESC
        ) oh
        GROUP BY rb.VisitDate, rb.PointID, rb.CategoryID, oh.VisitDate;
        
        CREATE CLUSTERED INDEX IX_DST_DatePointCat ON #DaysSinceTotal (VisitDate, PointID, CategoryID);

        -- 10. Расчет Average_Interval_Category (средний интервал между заказами категории)
        IF OBJECT_ID('tempdb..#AvgInterval') IS NOT NULL DROP TABLE #AvgInterval;
        WITH CategoryIntervals AS (
            SELECT 
                PointID,
                CategoryID,
                VisitDate,
                LAG(VisitDate) OVER (PARTITION BY PointID, CategoryID ORDER BY VisitDate) AS PrevVisitDate
            FROM #SalesHistory
        )
        SELECT 
            PointID,
            CategoryID,
            AVG(CAST(DATEDIFF(DAY, PrevVisitDate, VisitDate) AS FLOAT)) AS Average_Interval_Category
        INTO #AvgInterval
        FROM CategoryIntervals
        WHERE PrevVisitDate IS NOT NULL
        GROUP BY PointID, CategoryID;
        
        CREATE CLUSTERED INDEX IX_AI_PointCat ON #AvgInterval (PointID, CategoryID);

        -- 11. Расчет фичей продаж: SMA и другие
        IF OBJECT_ID('tempdb..#SalesFeatures') IS NOT NULL DROP TABLE #SalesFeatures;
        SELECT 
            rb.VisitDate,
            rb.PointID,
            rb.CategoryID,
            
            -- Prev_Order_Amount_Category (сумма предыдущего заказа)
            ISNULL((
                SELECT TOP 1 sh2.SumRoubles
                FROM #SalesHistory sh2
                WHERE sh2.PointID = rb.PointID 
                  AND sh2.CategoryID = rb.CategoryID
                  AND sh2.VisitDate < rb.VisitDate
                ORDER BY sh2.VisitDate DESC
            ), 0) AS Prev_Order_Amount_Category,
            
            -- SMA_3_Category
            ISNULL((
                SELECT AVG(CAST(sh2.SumRoubles AS FLOAT))
                FROM #SalesHistory sh2
                WHERE sh2.PointID = rb.PointID 
                  AND sh2.CategoryID = rb.CategoryID
                  AND sh2.VisitDate >= DATEADD(DAY, -3, rb.VisitDate)
                  AND sh2.VisitDate < rb.VisitDate
            ), 0) AS SMA_3_Category,
            
            -- SMA_7_Category
            ISNULL((
                SELECT AVG(CAST(sh2.SumRoubles AS FLOAT))
                FROM #SalesHistory sh2
                WHERE sh2.PointID = rb.PointID 
                  AND sh2.CategoryID = rb.CategoryID
                  AND sh2.VisitDate >= DATEADD(DAY, -7, rb.VisitDate)
                  AND sh2.VisitDate < rb.VisitDate
            ), 0) AS SMA_7_Category,
            
            -- SMA_30_Category
            ISNULL((
                SELECT AVG(CAST(sh2.SumRoubles AS FLOAT))
                FROM #SalesHistory sh2
                WHERE sh2.PointID = rb.PointID 
                  AND sh2.CategoryID = rb.CategoryID
                  AND sh2.VisitDate >= DATEADD(DAY, -30, rb.VisitDate)
                  AND sh2.VisitDate < rb.VisitDate
            ), 0) AS SMA_30_Category,
            
            -- Momentum_Category (отношение SMA_7 к SMA_30)
            CASE 
                WHEN (
                    SELECT AVG(CAST(sh2.SumRoubles AS FLOAT))
                    FROM #SalesHistory sh2
                    WHERE sh2.PointID = rb.PointID 
                      AND sh2.CategoryID = rb.CategoryID
                      AND sh2.VisitDate >= DATEADD(DAY, -30, rb.VisitDate)
                      AND sh2.VisitDate < rb.VisitDate
                ) > 0 THEN
                    ISNULL((
                        SELECT AVG(CAST(sh2.SumRoubles AS FLOAT))
                        FROM #SalesHistory sh2
                        WHERE sh2.PointID = rb.PointID 
                          AND sh2.CategoryID = rb.CategoryID
                          AND sh2.VisitDate >= DATEADD(DAY, -7, rb.VisitDate)
                          AND sh2.VisitDate < rb.VisitDate
                    ), 0) /
                    (
                        SELECT AVG(CAST(sh2.SumRoubles AS FLOAT))
                        FROM #SalesHistory sh2
                        WHERE sh2.PointID = rb.PointID 
                          AND sh2.CategoryID = rb.CategoryID
                          AND sh2.VisitDate >= DATEADD(DAY, -30, rb.VisitDate)
                          AND sh2.VisitDate < rb.VisitDate
                    )
                ELSE NULL
            END AS Momentum_Category,
            
            -- StdDev_Category
            ISNULL((
                SELECT STDEV(CAST(sh2.SumRoubles AS FLOAT))
                FROM #SalesHistory sh2
                WHERE sh2.PointID = rb.PointID 
                  AND sh2.CategoryID = rb.CategoryID
                  AND sh2.VisitDate >= DATEADD(DAY, -30, rb.VisitDate)
                  AND sh2.VisitDate < rb.VisitDate
            ), 0) AS StdDev_Category
            
        INTO #SalesFeatures
        FROM #ResultBase rb;
        
        CREATE CLUSTERED INDEX IX_SF_DatePointCat ON #SalesFeatures (VisitDate, PointID, CategoryID);

        -- 12. Финальный результат: объединение всех фичей (без SumRoubles - это целевая переменная!)
        SELECT 
            rb.VisitDate,
            rb.PointID,
            rb.CategoryID,
            rb.BranchID,
            rb.PointClass,
            rb.PointType,
            rb.Lat,
            rb.Lon,
            rb.MicroRegionID,
            
            -- Календарные фичи
            rb.day_of_week,
            rb.is_weekend,
            rb.is_monday,
            rb.is_friday,
            rb.is_saturday,
            rb.is_sunday,
            rb.is_holiday,
            rb.is_pre_holiday,
            rb.is_post_holiday,
            rb.month,
            rb.quarter,
            rb.week_of_year,
            rb.is_month_start,
            rb.is_month_end,
            rb.day_of_month,
            rb.day_of_year,
            rb.days_to_holiday,
            rb.days_from_holiday,
            -- Фичи истории заказов
            dsc.Days_Since_Last_Order_Category,
            dst.Days_Since_Last_Order_Total,
            ai.Average_Interval_Category,
            
            -- Фичи продаж
            sf.Prev_Order_Amount_Category,
            sf.SMA_3_Category,
            sf.SMA_7_Category,
            sf.SMA_30_Category,
            sf.Momentum_Category,
            sf.StdDev_Category
            
        FROM #ResultBase rb
        LEFT JOIN #DaysSinceCategory dsc ON rb.VisitDate = dsc.VisitDate 
                                         AND rb.PointID = dsc.PointID 
                                         AND rb.CategoryID = dsc.CategoryID
        LEFT JOIN #DaysSinceTotal dst ON rb.VisitDate = dst.VisitDate 
                                      AND rb.PointID = dst.PointID 
                                      AND rb.CategoryID = dst.CategoryID
        LEFT JOIN #AvgInterval ai ON rb.PointID = ai.PointID 
                                  AND rb.CategoryID = ai.CategoryID
        LEFT JOIN #SalesFeatures sf ON rb.VisitDate = sf.VisitDate 
                                    AND rb.PointID = sf.PointID 
                                    AND rb.CategoryID = sf.CategoryID
        ORDER BY rb.VisitDate, rb.PointID, rb.CategoryID;

        -- Очистка временных таблиц
        DROP TABLE #ItemMap;
        DROP TABLE #PointFeatures;
        DROP TABLE #Categories;
        DROP TABLE #FullGrid;
        DROP TABLE #SalesHistory;
        DROP TABLE #OrderHistory;
        DROP TABLE #ResultBase;
        DROP TABLE #DaysSinceCategory;
        DROP TABLE #DaysSinceTotal;
        DROP TABLE #AvgInterval;
        DROP TABLE #SalesFeatures;
        
    END TRY
    BEGIN CATCH
        -- Обработка ошибок
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        DECLARE @ErrorSeverity INT = ERROR_SEVERITY();
        DECLARE @ErrorState INT = ERROR_STATE();
        DECLARE @ErrorLine INT = ERROR_LINE();
        
        -- Логирование ошибки
        PRINT 'Ошибка в процедуре SNS_ML_Get_Test_Data:';
        PRINT 'Сообщение: ' + @ErrorMessage;
        PRINT 'Строка: ' + CAST(@ErrorLine AS NVARCHAR(10));
        
        -- Вызов временных таблиц если они существуют
        IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
        IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
        IF OBJECT_ID('tempdb..#Categories') IS NOT NULL DROP TABLE #Categories;
        IF OBJECT_ID('tempdb..#FullGrid') IS NOT NULL DROP TABLE #FullGrid;
        IF OBJECT_ID('tempdb..#SalesHistory') IS NOT NULL DROP TABLE #SalesHistory;
        IF OBJECT_ID('tempdb..#OrderHistory') IS NOT NULL DROP TABLE #OrderHistory;
        IF OBJECT_ID('tempdb..#ResultBase') IS NOT NULL DROP TABLE #ResultBase;
        IF OBJECT_ID('tempdb..#DaysSinceCategory') IS NOT NULL DROP TABLE #DaysSinceCategory;
        IF OBJECT_ID('tempdb..#DaysSinceTotal') IS NOT NULL DROP TABLE #DaysSinceTotal;
        IF OBJECT_ID('tempdb..#AvgInterval') IS NOT NULL DROP TABLE #AvgInterval;
        IF OBJECT_ID('tempdb..#SalesFeatures') IS NOT NULL DROP TABLE #SalesFeatures;
        
        -- Проброс ошибки дальше
        RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
        RETURN -1;
    END CATCH
END;
GO
