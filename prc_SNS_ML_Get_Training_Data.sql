IF OBJECT_ID('SNS_ML_Get_Training_Data', 'P') IS NOT NULL
    DROP PROCEDURE SNS_ML_Get_Training_Data;
GO

CREATE PROCEDURE SNS_ML_Get_Training_Data
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @StartDate DATE = DATEADD(YEAR, -1, GETDATE());

    -- 1. Справочник SKU -> Группа
    IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
    SELECT CAST(i.iid AS INT) as iid, i.itID, CAST(br.AttrValueId AS INT) AS brand_id 
    INTO #ItemMap FROM DS_ITEMS i 
    INNER JOIN DS_ObjectsAttributes br ON i.iid = br.id AND br.Activeflag = 1 AND br.AttrId = 635
    WHERE i.activeFlag = 1;
    CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);

    -- 2. Признаки точек
    IF OBJECT_ID('tempdb..#PointFeatures') IS NOT NULL DROP TABLE #PointFeatures;
    SELECT f.fid AS PointID, f.distid AS BranchID, 
           MAX(CASE WHEN fa.attrid = 360 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END) AS Lat,
           MAX(CASE WHEN fa.attrid = 361 THEN TRY_CAST(REPLACE(REPLACE(fa.attrtext, ',', '.'), ' ', '') AS FLOAT) END) AS Lon,
           ISNULL(MAX(CASE WHEN fa.attrid = 602 THEN fa.attrtext END), 'Unknown') AS PointClass,
           ISNULL(MAX(CASE WHEN fa.attrid = 555 THEN fa.attrtext END), 'Unknown') AS PointType
    INTO #PointFeatures FROM ds_faces f LEFT JOIN ds_facesattributes fa ON f.fid = fa.fid AND fa.activeflag = 1 
    WHERE f.ftype = 1 AND f.factiveflag = 1 GROUP BY f.fid, f.distid;
    CREATE CLUSTERED INDEX IX_PointFeatures_ID ON #PointFeatures (PointID);

    -- 3. Агрегация РЕАЛЬНЫХ продаж
    IF OBJECT_ID('tempdb..#SalesRaw') IS NOT NULL DROP TABLE #SalesRaw;
    SELECT 
        CAST(o.orDate AS DATE) AS VisitDate, o.mfID AS PointID, m.itID AS CategoryID, m.brand_id AS GroupID, 
        SUM(SUM(oi.SumRoubles)) OVER(PARTITION BY CAST(o.orDate AS DATE), o.mfID, m.itID) AS TargetCategorySum,
        SUM(oi.SumRoubles) AS TargetBrandSum
    INTO #SalesRaw
    FROM DS_Orders o
    INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
    INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid
    WHERE o.orType = 1 AND o.orDate >= @StartDate
    GROUP BY CAST(o.orDate AS DATE), o.mfID, m.itID, m.brand_id;

    -- 4. Сборка финальной сетки (Факты + Нули) во временную таблицу для быстрого расчета LAG
    IF OBJECT_ID('tempdb..#FinalGrid') IS NOT NULL DROP TABLE #FinalGrid;
    
    SELECT s.VisitDate, s.PointID, s.CategoryID, s.GroupID, s.TargetCategorySum, s.TargetBrandSum 
    INTO #FinalGrid 
    FROM #SalesRaw s
    
    UNION ALL
    
    -- Генерация нулей по историческому профилю
    SELECT v.VisitDate, v.PointID, h.CategoryID, h.GroupID, v.TargetCategorySum, 0 
    FROM (SELECT DISTINCT VisitDate, PointID, CategoryID, TargetCategorySum FROM #SalesRaw) v
    INNER JOIN (SELECT DISTINCT PointID, CategoryID, GroupID FROM #SalesRaw) h 
        ON v.PointID = h.PointID AND v.CategoryID = h.CategoryID
    WHERE NOT EXISTS (
        SELECT 1 FROM #SalesRaw s4 
        WHERE s4.VisitDate = v.VisitDate AND s4.PointID = v.PointID AND s4.GroupID = h.GroupID
    );

    CREATE CLUSTERED INDEX IX_Grid ON #FinalGrid (PointID, GroupID, VisitDate);

    -- 5. ФИНАЛЬНЫЙ ВЫВОД с использованием LAG (В разы быстрее подзапроса)
    SELECT 
        t.VisitDate, t.PointID, p.BranchID, p.PointClass, p.PointType,
        CAST(ROUND(p.Lat, 2) AS VARCHAR(10)) + '_' + CAST(ROUND(p.Lon, 2) AS VARCHAR(10)) AS MicroRegionID,
        t.CategoryID, t.GroupID, t.TargetCategorySum, t.TargetBrandSum,
        ISNULL(DATEDIFF(DAY, PrevDate, t.VisitDate), 999) AS DaysSinceLastSale
    FROM (
        SELECT *,
            -- Находим дату предыдущего визита в рамках этой точки и бренда
            LAG(VisitDate) OVER (PARTITION BY PointID, GroupID ORDER BY VisitDate) AS PrevDate
        FROM #FinalGrid
    ) t
    INNER JOIN #PointFeatures p ON t.PointID = p.PointID
    ORDER BY t.VisitDate, t.PointID;

    DROP TABLE #ItemMap; DROP TABLE #PointFeatures; DROP TABLE #SalesRaw; DROP TABLE #FinalGrid;
END
GO
