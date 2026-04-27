CREATE PROCEDURE SNS_ML_Get_Brand_Sales
    @StartDate DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;

    -- Если дата не указана, используем текущую дату
    IF @StartDate IS NULL
        SET @StartDate = CAST(GETDATE() AS DATE);

    -- Рассчитываем дату начала периода (3 месяца назад)
    DECLARE @EndDate DATE = @StartDate;
    DECLARE @PeriodStart DATE = DATEADD(MONTH, -3, @StartDate);

    -- Временная таблица для маппинга товаров и брендов
    IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
    SELECT 
        CAST(i.iid AS INT) as iid, 
        CAST(brand.attrvalueid AS INT) as brand,
        br.attrvaluename as brand_name,
        CAST(i.itID AS INT) AS CategoryID
    INTO #ItemMap 
    FROM DS_ITEMS i 
    INNER JOIN DS_ObjectsAttributes brand ON i.iID = brand.id AND brand.Activeflag = 1 AND brand.AttrId = 635
    INNER JOIN DS_AttributesValues br ON brand.AttrValueId = br.AttrValueID
    WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;

    CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);

    -- Шаг сетки для 3 км: 3 / 111.0 ≈ 0.027 градуса
    DECLARE @GridStep FLOAT = 0.027; 

    -- Признаки точек + Расчет Микрорегиона (3x3 км)
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

    -- Основной запрос: выборка данных о продажах по категориям и брендам за 3 месяца
    SELECT 
        o.mfid as PointId, 
        CAST(o.orDate AS DATE) AS VisitDate, 
        pf.BranchID,
        pf.PointClass,
        pf.PointType,
        pf.Lat,
        pf.Lon,
        pf.MicroRegionID,
        map.CategoryID, 
        map.brand, 
        map.brand_name, 
        SUM(oi.Amount) as brand_amount
    FROM DS_Orders o
    INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
    INNER JOIN #ItemMap map ON oi.iID = map.iid
    INNER JOIN #PointFeatures pf ON o.mfID = pf.PointID
    WHERE o.orDate >= @PeriodStart 
      AND o.orDate < @EndDate 
      AND o.orType = 1
    GROUP BY 
        o.mfid, 
        CAST(o.orDate AS DATE), 
        map.CategoryID, 
        map.brand, 
        map.brand_name,  
        pf.BranchID,
        pf.PointClass,
        pf.PointType,
        pf.Lat,
        pf.Lon,
        pf.MicroRegionID
    ORDER BY 
        VisitDate DESC, 
        CategoryID, 
        brand;
END;
