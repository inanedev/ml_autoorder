declare @cdate date
set @cdate = dbo.DateOnly(GETDATE())

    IF OBJECT_ID('tempdb..#ItemMap') IS NOT NULL DROP TABLE #ItemMap;
    SELECT 
        CAST(i.iid AS INT) as iid, 
		cast(brand.attrvalueid as int) as brand,
		br.attrvaluename as brand_name,
        CAST(i.itID AS INT) AS CategoryID
    INTO #ItemMap 
    FROM DS_ITEMS i 
	inner join DS_ObjectsAttributes brand on i.iID = brand.id and brand.Activeflag =1 and brand.AttrId = 635
	inner join DS_AttributesValues br on brand.AttrValueId = br.AttrValueID
    WHERE i.activeFlag = 1 AND i.itID IS NOT NULL;
    

    CREATE CLUSTERED INDEX IX_ItemMap_iid ON #ItemMap (iid);
	--select * from #ItemMap

	    -- Шаг сетки для 3 км: 3 / 111.0 ≈ 0.027 градуса
    DECLARE @GridStep FLOAT = 0.027; 


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



select o.mfid, CAST(o.orDate AS DATE) AS VisitDate, 
        pf.BranchID,
        pf.PointClass,
        pf.PointType,      -- Добавлен атрибут 555
        pf.Lat,
        pf.Lon,
        pf.MicroRegionID,  -- Добавлен микрорегион 3x3 км
map.CategoryID, map.brand, map.brand_name, sum(oi.Amount) as brand_amount from DS_Orders o
inner join DS_Orders_Items oi on o.MasterFID = oi.MasterFID and o.orID = oi.orID 
inner join #ItemMap map on oi.iID = map.iid
INNER JOIN #PointFeatures pf ON o.mfID = pf.PointID
where o.orDate<@cdate and o.orType = 1
group by o.mfid, CAST(o.orDate AS DATE), map.CategoryID, map.brand, map.brand_name,  pf.BranchID,
        pf.PointClass,
        pf.PointType,      -- Добавлен атрибут 555
        pf.Lat,
        pf.Lon,
        pf.MicroRegionID


