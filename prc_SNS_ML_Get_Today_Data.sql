IF OBJECT_ID('SNS_ML_Get_Today_Data', 'P') IS NOT NULL
    DROP PROCEDURE SNS_ML_Get_Today_Data;
GO

CREATE PROCEDURE SNS_ML_Get_Today_Data
    @TargetDate DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    IF @TargetDate IS NULL SET @TargetDate = CAST(GETDATE() AS DATE);
    
    DECLARE @HistoryStart DATETIME = DATEADD(year, -1, CAST(@TargetDate AS DATETIME));
    DECLARE @TodayDW INT = (DATEPART(dw, @TargetDate) + @@DATEFIRST - 2) % 7 + 1;

    -- 1. Справочник SKU -> Группа
    IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
    SELECT CAST(i.iid AS INT) as iid, i.itID, CAST(br.AttrValueId AS INT) AS brand_id 
    INTO #ItemMap FROM DS_ITEMS i 
    INNER JOIN DS_ObjectsAttributes br ON i.iid = br.id AND br.Activeflag = 1 AND br.AttrId = 635
    WHERE i.activeFlag = 1;
    CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);

    -- 2. Маршрут и график (DaysUntilNextVisit)
    CREATE TABLE #tmp_route (DistID int, TPfid int, TTfid int, planFlag int);
    INSERT INTO #tmp_route EXEC DMT_Get_RoutesEx 1, @TargetDate, @TargetDate, null, '#38;2;4;34';

    IF OBJECT_ID('tempdb..#ActiveRoute') IS NOT NULL DROP TABLE #ActiveRoute;
    SELECT DISTINCT 
        r.TTfid as PointID,
        CASE 
            WHEN fa.attrtext LIKE '%' + CAST(((@TodayDW + 0) % 7 + 1) AS CHAR(1)) + '%' THEN 1
            WHEN fa.attrtext LIKE '%' + CAST(((@TodayDW + 1) % 7 + 1) AS CHAR(1)) + '%' THEN 2
            WHEN fa.attrtext LIKE '%' + CAST(((@TodayDW + 2) % 7 + 1) AS CHAR(1)) + '%' THEN 3
            WHEN fa.attrtext LIKE '%' + CAST(((@TodayDW + 3) % 7 + 1) AS CHAR(1)) + '%' THEN 4
            WHEN fa.attrtext LIKE '%' + CAST(((@TodayDW + 4) % 7 + 1) AS CHAR(1)) + '%' THEN 5
            WHEN fa.attrtext LIKE '%' + CAST(((@TodayDW + 5) % 7 + 1) AS CHAR(1)) + '%' THEN 6
            ELSE 7 
        END as DaysUntilNextVisit
    INTO #ActiveRoute
    FROM #tmp_route r
    LEFT JOIN DS_FacesAttributes fa ON r.TTfid = fa.fid AND fa.attrid = 2097 AND fa.activeflag = 1
    WHERE r.planFlag = 1;

    -- 3. Признаки точек и Микрорегионы
    IF OBJECT_ID('tempdb..#PF_Raw') IS NOT NULL DROP TABLE #PF_Raw;
    SELECT 
        f.fid as PointID, f.distid as BranchID,
        MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END) as Lat,
        MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END) as Lon,
        ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), 'Unknown') as PointClass,
        ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), 'Unknown') as PointType
    INTO #PF_Raw
    FROM ds_faces f
    INNER JOIN #ActiveRoute ar ON f.fid = ar.PointID
    LEFT JOIN ds_facesattributes fa ON f.fid = fa.fid AND fa.activeflag = 1
    WHERE f.ftype = 1 AND f.factiveflag = 1
    GROUP BY f.fid, f.distid;

    -- Вспомогательная колонка микрорегиона для джойнов
    IF OBJECT_ID('tempdb..#PF_Final') IS NOT NULL DROP TABLE #PF_Final;
    SELECT *, 
        CASE WHEN Lat IS NOT NULL AND Lon IS NOT NULL 
        THEN CAST(CAST(Lat AS DECIMAL(10,2)) AS VARCHAR(20)) + '_' + CAST(CAST(Lon AS DECIMAL(10,2)) AS VARCHAR(20))
        ELSE 'Br_' + CAST(BranchID AS VARCHAR(10)) END as MicroRegionID
    INTO #PF_Final FROM #PF_Raw;

    -- 4. Продажи в микрорайонах (Кластерные хиты) за 3 мес
    IF OBJECT_ID('tempdb..#ClusterTop') IS NOT NULL DROP TABLE #ClusterTop;
    SELECT 
        p.MicroRegionID, p.PointType, m.brand_id as GroupID,
        SUM(oi.SumRoubles) as ClusterSales
    INTO #ClusterTop
    FROM DS_Orders o
    INNER JOIN DS_Orders_Items oi ON o.orID = oi.orID AND o.MasterFID = oi.MasterFID
    INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid
    INNER JOIN #PF_Final p ON o.mfID = p.PointID
    WHERE o.orDate >= DATEADD(month, -3, @TargetDate)
    GROUP BY p.MicroRegionID, p.PointType, m.brand_id;

    -- 5. Личная история точек
    IF OBJECT_ID('tempdb..#SalesRaw') IS NOT NULL DROP TABLE #SalesRaw;
    SELECT o.mfID as PointID, o.orID, o.orDate, m.itID as CategoryID, m.brand_id as GroupID, oi.SumRoubles as sSum, oi.Amount as sQty
    INTO #SalesRaw FROM DS_Orders o INNER JOIN #ActiveRoute ap ON o.mfID = ap.PointID
    INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID
    INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid 
    WHERE o.orType = 1 AND o.orDate >= @HistoryStart AND o.orDate < CAST(@TargetDate AS DATETIME);

    IF OBJECT_ID('tempdb..#H_Final') IS NOT NULL DROP TABLE #H_Final;
    SELECT sr.PointID, sr.CategoryID, sr.GroupID, AVG(cat.CatSum) as AvgCatSum, AVG(sr.sSum) as AvgBrandSum, AVG(sr.sQty) as AvgBrandQty, DATEDIFF(day, MAX(sr.orDate), @TargetDate) as DaysSinceLastSale
    INTO #H_Final FROM #SalesRaw sr
    INNER JOIN (SELECT orID, CategoryID, SUM(sSum) as CatSum FROM #SalesRaw GROUP BY orID, CategoryID) cat ON sr.orID = cat.orID AND sr.CategoryID = cat.CategoryID
    GROUP BY sr.PointID, sr.CategoryID, sr.GroupID;

    -- ФИНАЛЬНЫЙ ВЫВОД
    SELECT 
        @TargetDate as VisitDate, p.PointID, p.BranchID, p.PointClass, p.PointType, p.MicroRegionID,
        ig.itID as CategoryID, ig.brand_id as GroupID,
        ISNULL(h.AvgCatSum, 0) as LastAvgCategorySum,
        ISNULL(h.AvgBrandSum, 0) as LastAvgBrandSum,
        ISNULL(h.AvgBrandQty, 0) as LastAvgBrandQty,
        ISNULL(h.DaysSinceLastSale, 999) as DaysSinceLastSale,
        ISNULL(ar.DaysUntilNextVisit, 7) as DaysUntilNextVisit,
        ISNULL(ct.ClusterSales, 0) as ClusterSales
    FROM #PF_Final p
    INNER JOIN #ActiveRoute ar ON p.PointID = ar.PointID
    CROSS JOIN (SELECT DISTINCT itID, brand_id FROM #ItemMap) ig
    LEFT JOIN #H_Final h ON p.PointID = h.PointID AND ig.brand_id = h.GroupID AND ig.itID = h.CategoryID
    LEFT JOIN #ClusterTop ct ON p.MicroRegionID = ct.MicroRegionID AND p.PointType = ct.PointType AND ig.brand_id = ct.GroupID
    ORDER BY p.PointID, ig.brand_id;
END
GO
