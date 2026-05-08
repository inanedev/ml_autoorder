IF OBJECT_ID('dbo.SNS_ML_Get_Test_Data', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SNS_ML_Get_Test_Data;
GO

CREATE PROCEDURE [dbo].[SNS_ML_Get_Test_Data]
    @TargetDate DATE
AS
BEGIN
    SET NOCOUNT ON;
    
    BEGIN TRY
        -- Валидация входных параметров
        IF @TargetDate IS NULL
        BEGIN
            RAISERROR('Параметр @TargetDate должен быть заполнен', 16, 1);
            RETURN -1;
        END

        -- Установка первого дня недели = понедельник (как в Python .weekday() + 1)
        SET DATEFIRST 1;

        DECLARE @GridStep FLOAT = 0.027; 

        -- 1. Справочник SKU -> Категория (оригинальные таблицы DS_ITEMS)
        IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
        SELECT 
            CAST(i.iid AS INT) as iid, 
            CAST(i.itID AS INT) AS CategoryID
        INTO #ItemMap 
        FROM DS_ITEMS i 
        WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;
        
        CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);

        -- 2. Признаки точек (оригинальные таблицы ds_faces, ds_facesattributes)
        IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
        SELECT 
            f.fid AS PointID, 
            f.distid AS BranchID, 
            ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) AS Lat,
            ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) AS Lon,
            ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), 'Unknown') AS PointClass,
            ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), 'Unknown') AS PointType,
            ISNULL(MAX(CASE WHEN fa.attrid = 644 THEN fa.attrtext END), '') AS VisitDays,
            CAST(FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) / @GridStep) * @GridStep AS VARCHAR(20)) + '_' + 
            CAST(FLOOR(ISNULL(MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END), 0) / @GridStep) * @GridStep AS VARCHAR(20)) AS MicroRegionID
        INTO #PointFeatures 
        FROM ds_faces f 
        LEFT JOIN ds_facesattributes fa ON f.fid = fa.fid AND fa.activeflag = 1 
        WHERE f.ftype = 1 AND f.factiveflag = 1 
        GROUP BY f.fid, f.distid;

        CREATE CLUSTERED INDEX IX_PF_PointID ON #PointFeatures (PointID);

        -- 3. Категории (оригинальные DS_Orders, DS_Orders_Items)
        IF OBJECT_ID('tempdb..#AllCategories') IS NOT NULL DROP TABLE #AllCategories;
        SELECT DISTINCT m.CategoryID INTO #AllCategories FROM DS_Orders o INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid WHERE o.orType = 1 AND o.orDate >= DATEADD(MONTH, -1, @TargetDate) AND o.orDate < @TargetDate;

        IF OBJECT_ID('tempdb..#ClientCategories') IS NOT NULL DROP TABLE #ClientCategories;
        SELECT DISTINCT o.mfID AS PointID, m.CategoryID INTO #ClientCategories FROM DS_Orders o INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid WHERE o.orType = 1 AND o.orDate >= DATEADD(MONTH, -3, @TargetDate) AND o.orDate < @TargetDate;

        -- 4. Грид
        IF OBJECT_ID('tempdb..#FullGrid') IS NOT NULL DROP TABLE #FullGrid;
        SELECT CAST(@TargetDate AS DATE) AS VisitDate, cc.PointID, cc.CategoryID, pf.BranchID, pf.PointClass, pf.PointType, pf.Lat, pf.Lon, pf.MicroRegionID, pf.VisitDays INTO #FullGrid FROM #ClientCategories cc INNER JOIN #PointFeatures pf ON cc.PointID = pf.PointID
        UNION ALL
        SELECT CAST(@TargetDate AS DATE) AS VisitDate, pf.PointID, ac.CategoryID, pf.BranchID, pf.PointClass, pf.PointType, pf.Lat, pf.Lon, pf.MicroRegionID, pf.VisitDays FROM #PointFeatures pf CROSS JOIN #AllCategories ac WHERE pf.PointID NOT IN (SELECT DISTINCT PointID FROM #ClientCategories);

        -- 5. История визитов и продаж
        IF OBJECT_ID('tempdb..#VisitHistory') IS NOT NULL DROP TABLE #VisitHistory;
        SELECT CAST(o.orDate AS DATE) AS VisitDate, o.mfID AS PointID INTO #VisitHistory FROM DS_Orders o WHERE o.orType = 1 AND o.orDate >= DATEADD(MONTH, -1, @TargetDate) AND o.orDate < @TargetDate GROUP BY CAST(o.orDate AS DATE), o.mfID;

        IF OBJECT_ID('tempdb..#SalesHistory') IS NOT NULL DROP TABLE #SalesHistory;
        SELECT CAST(o.orDate AS DATE) AS VisitDate, o.mfID AS PointID, m.CategoryID, SUM(ISNULL(oi.SumRoubles, 0)) AS SumRoubles INTO #SalesHistory FROM DS_Orders o INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid WHERE o.orType = 1 AND o.orDate >= DATEADD(MONTH, -1, @TargetDate) AND o.orDate < @TargetDate GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID;

        -- 7. Календарные фичи с использованием таблицы SNS_ML_Holidays
        IF OBJECT_ID('tempdb..#ResultBase') IS NOT NULL DROP TABLE #ResultBase;
        SELECT 
            fg.*,
            DATEPART(WEEKDAY, fg.VisitDate) AS DayOfWeek,
            CASE WHEN DATEPART(WEEKDAY, fg.VisitDate) = 5 THEN 1 ELSE 0 END AS IsFriday,
            CASE WHEN DATEPART(WEEKDAY, fg.VisitDate) = 1 THEN 1 ELSE 0 END AS IsMonday,
            DATEPART(QUARTER, fg.VisitDate) AS Quarter,
            MONTH(fg.VisitDate) AS Month,
            DATEPART(WEEK, fg.VisitDate) AS WeekOfYear,
            DAY(fg.VisitDate) AS DayOfMonth,
            DATEPART(DAYOFYEAR, fg.VisitDate) AS DayOfYear,
            CASE 
                WHEN DAY(fg.VisitDate) >= DAY(EOMONTH(fg.VisitDate)) - 2 THEN 1
                WHEN DAY(fg.VisitDate) <= 2 AND DATEPART(WEEKDAY, fg.VisitDate) >= 5 THEN 1
                WHEN DAY(fg.VisitDate) + 7 > DAY(EOMONTH(fg.VisitDate)) AND DATEPART(WEEKDAY, fg.VisitDate) = 6 THEN 1
                ELSE 0 
            END AS isEndOfMonth,
            
            -- Праздник из таблицы (вместо жесткого кода)
            CASE WHEN EXISTS (SELECT 1 FROM dbo.SNS_ML_Holidays h WHERE h.dayoff = fg.VisitDate) 
                 THEN 1 ELSE 0 END AS IsHoliday,
            
            -- Расчет дней до/после праздника через таблицу SNS_ML_Holidays
            ISNULL((SELECT TOP 1 DATEDIFF(DAY, fg.VisitDate, h.dayoff) FROM dbo.SNS_ML_Holidays h WHERE h.dayoff > fg.VisitDate ORDER BY h.dayoff), 365) AS DaysToNextHoliday,
            ISNULL((SELECT TOP 1 DATEDIFF(DAY, h.dayoff, fg.VisitDate) FROM dbo.SNS_ML_Holidays h WHERE h.dayoff < fg.VisitDate ORDER BY h.dayoff DESC), 365) AS DaysSinceLastHoliday,
            
            CASE WHEN EXISTS (SELECT 1 FROM dbo.SNS_ML_Holidays h WHERE h.dayoff > fg.VisitDate AND h.dayoff <= DATEADD(DAY, 3, fg.VisitDate)) THEN 1 ELSE 0 END AS IsPreHoliday,
            CASE WHEN EXISTS (SELECT 1 FROM dbo.SNS_ML_Holidays h WHERE h.dayoff < fg.VisitDate AND h.dayoff >= DATEADD(DAY, -3, fg.VisitDate)) THEN 1 ELSE 0 END AS IsPostHoliday

        INTO #ResultBase FROM #FullGrid fg;

        -- 8. Расчет DaysLastVisit и DaysNextVisit (оригинальная логика)
        IF OBJECT_ID('tempdb..#VisitWithShifts') IS NOT NULL DROP TABLE #VisitWithShifts;
        WITH LastVisitBefore AS (SELECT PointID, MAX(VisitDate) AS LastVisitDate FROM #VisitHistory WHERE VisitDate < @TargetDate GROUP BY PointID),
        NextVisitAfter AS (SELECT PointID, MIN(VisitDate) AS NextVisitDate FROM #VisitHistory WHERE VisitDate > @TargetDate GROUP BY PointID),
        VisitDaysCalc AS (
            SELECT pf.PointID, pf.VisitDays, DATEPART(WEEKDAY, @TargetDate) AS CurrentDayOfWeek,
            (SELECT MIN(CAST(x.value AS INT)) FROM (SELECT t.c.value('.', 'INT') AS value FROM (SELECT CAST('<n>' + REPLACE(pf.VisitDays, ',', '</n><n>') + '</n>' AS XML) AS xmldata) AS a CROSS APPLY xmldata.nodes('/n') AS t(c)) x WHERE x.value > DATEPART(WEEKDAY, @TargetDate)) AS NextVisitDayInWeek,
            (SELECT MIN(CAST(x.value AS INT)) FROM (SELECT t.c.value('.', 'INT') AS value FROM (SELECT CAST('<n>' + REPLACE(pf.VisitDays, ',', '</n><n>') + '</n>' AS XML) AS xmldata) AS a CROSS APPLY xmldata.nodes('/n') AS t(c)) x) AS MinVisitDayInWeek
            FROM #PointFeatures pf
        )
        SELECT lv.PointID, @TargetDate AS VisitDate, ISNULL(DATEDIFF(DAY, lv.LastVisitDate, @TargetDate), 7) AS DaysLastVisit,
            CASE WHEN nv.NextVisitDate IS NOT NULL THEN DATEDIFF(DAY, @TargetDate, nv.NextVisitDate)
                 WHEN vdc.VisitDays <> '' THEN CASE WHEN vdc.NextVisitDayInWeek IS NOT NULL THEN vdc.NextVisitDayInWeek - vdc.CurrentDayOfWeek ELSE (7 - vdc.CurrentDayOfWeek) + vdc.MinVisitDayInWeek END
                 ELSE 7 END AS DaysNextVisit
        INTO #VisitWithShifts FROM LastVisitBefore lv LEFT JOIN NextVisitAfter nv ON lv.PointID = nv.PointID LEFT JOIN VisitDaysCalc vdc ON lv.PointID = vdc.PointID;

        -- 9-10. Продажи категорий (оригинальная логика)
        IF OBJECT_ID('tempdb..#CategorySalesWithShifts') IS NOT NULL DROP TABLE #CategorySalesWithShifts;
        SELECT PointID, CategoryID, @TargetDate AS VisitDate, ISNULL(DATEDIFF(DAY, MAX(VisitDate), @TargetDate), 7) AS DaysLastSalesCategory INTO #CategorySalesWithShifts FROM #SalesHistory WHERE VisitDate < @TargetDate GROUP BY PointID, CategoryID;

        IF OBJECT_ID('tempdb..#DailyCategorySales') IS NOT NULL DROP TABLE #DailyCategorySales;
        SELECT PointID, CategoryID, @TargetDate AS VisitDate, SumRoubles AS LastSalesCategory INTO #DailyCategorySales FROM (SELECT PointID, CategoryID, SumRoubles, ROW_NUMBER() OVER (PARTITION BY PointID, CategoryID ORDER BY VisitDate DESC) AS rn FROM #SalesHistory WHERE VisitDate < @TargetDate) tmp WHERE rn = 1;

        -- 11. Финальный результат с добавлением isHolidayNextVisit
        SELECT 
            rb.VisitDate, rb.PointID, rb.CategoryID, rb.BranchID, rb.PointClass, rb.PointType, rb.Lat, rb.Lon, rb.MicroRegionID,
            rb.DayOfWeek, rb.IsFriday, rb.IsMonday, rb.DaysToNextHoliday, rb.DaysSinceLastHoliday, rb.IsPreHoliday, rb.IsPostHoliday, rb.Quarter, rb.Month, rb.WeekOfYear, rb.DayOfMonth, rb.DayOfYear, rb.isEndOfMonth,
            SIN(2.0 * PI() * CAST(rb.DayOfYear AS FLOAT) / 366.0) AS DayOfYear_sin, COS(2.0 * PI() * CAST(rb.DayOfYear AS FLOAT) / 366.0) AS DayOfYear_cos,
            SIN(2.0 * PI() * CAST(rb.Month AS FLOAT) / 12.0) AS Month_sin, COS(2.0 * PI() * CAST(rb.Month AS FLOAT) / 12.0) AS Month_cos,
            SIN(2.0 * PI() * CAST(rb.DayOfWeek AS FLOAT) / 7.0) AS DayOfWeek_sin, COS(2.0 * PI() * CAST(rb.DayOfWeek AS FLOAT) / 7.0) AS DayOfWeek_cos,
            SIN(2.0 * PI() * CAST(rb.WeekOfYear AS FLOAT) / 53.0) AS WeekOfYear_sin, COS(2.0 * PI() * CAST(rb.WeekOfYear AS FLOAT) / 53.0) AS WeekOfYear_cos,
            SIN(2.0 * PI() * CAST(rb.Quarter AS FLOAT) / 4.0) AS Quarter_sin, COS(2.0 * PI() * CAST(rb.Quarter AS FLOAT) / 4.0) AS Quarter_cos,
            ISNULL(vws.DaysLastVisit, 7) AS DaysLastVisit, ISNULL(vws.DaysNextVisit, 7) AS DaysNextVisit,
            ISNULL(csw.DaysLastSalesCategory, 7) AS DaysLastSalesCategory, ISNULL(dcs.LastSalesCategory, 0) AS LastSalesCategory,
            
            -- НОВАЯ ФИЧА: Если следующий визит попадает на выходной или праздник из таблицы
            CASE 
                WHEN EXISTS (
                    SELECT 1 FROM dbo.SNS_ML_Holidays h 
                    WHERE h.dayoff = DATEADD(DAY, ISNULL(vws.DaysNextVisit, 7), rb.VisitDate)
                ) 
                OR DATEPART(WEEKDAY, DATEADD(DAY, ISNULL(vws.DaysNextVisit, 7), rb.VisitDate)) IN (6, 7)
                THEN 1 
                ELSE 0 
            END AS isHolidayNextVisit
            
        FROM #ResultBase rb
        LEFT JOIN #VisitWithShifts vws ON rb.PointID = vws.PointID AND rb.VisitDate = vws.VisitDate
        LEFT JOIN #CategorySalesWithShifts csw ON rb.PointID = csw.PointID AND rb.CategoryID = csw.CategoryID AND rb.VisitDate = csw.VisitDate
        LEFT JOIN #DailyCategorySales dcs ON rb.PointID = dcs.PointID AND rb.CategoryID = dcs.CategoryID AND rb.VisitDate = dcs.VisitDate
        ORDER BY rb.VisitDate, rb.PointID, rb.CategoryID;

    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;
GO