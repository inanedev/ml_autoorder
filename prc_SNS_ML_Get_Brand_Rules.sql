IF OBJECT_ID('SNS_ML_Get_Brand_Rules', 'P') IS NOT NULL
    DROP PROCEDURE SNS_ML_Get_Brand_Rules;
GO

CREATE PROCEDURE SNS_ML_Get_Brand_Rules
AS
BEGIN
    SET NOCOUNT ON;

    -- 1. Сбор базовых цен из прайс-листов типа %DSS
    IF OBJECT_ID('tempdb..#ListPrices') IS NOT NULL DROP TABLE #ListPrices;
    SELECT 
        br.AttrValueId AS brand_id,
        AVG(ip.costroubles) AS AvgListPrice
    INTO #ListPrices
    FROM DS_ITEMS_PRICES ip
    INNER JOIN DS_PRICELISTS p ON ip.plid = p.plid
    INNER JOIN DS_ObjectsAttributes br ON ip.iid = br.id AND br.attrid = 635 AND br.Activeflag = 1
    WHERE p.activeFlag = 1 AND p.NotActive = 0 AND p.plName LIKE '%DSS' AND ip.activeflag = 1
    GROUP BY br.AttrValueId;

    -- 2. Сбор реальных цен реализации за последние 3 месяца
    IF OBJECT_ID('tempdb..#RealPrices') IS NOT NULL DROP TABLE #RealPrices;
    SELECT 
        br.AttrValueId AS brand_id,
        SUM(oi.SumRoubles) / NULLIF(SUM(oi.Amount), 0) AS AvgRealPrice
    INTO #RealPrices
    FROM DS_Orders o
    INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID
    INNER JOIN DS_ObjectsAttributes br ON oi.iID = br.id AND br.attrid = 635 AND br.Activeflag = 1
    WHERE o.orType = 1 AND o.orDate >= DATEADD(MONTH, -3, GETDATE())
    GROUP BY br.AttrValueId;

    -- 3. Определение РЕЗЕРВНОГО кванта (через Max IID -> UnitLevel 2)
    -- Берем SKU с максимальным ID для каждого бренда и его rate упаковки
    IF OBJECT_ID('tempdb..#BackupQuantums') IS NOT NULL DROP TABLE #BackupQuantums;
    SELECT 
        t.brand_id,
        ui.rate AS BackupQuantum
    INTO #BackupQuantums
    FROM (
        SELECT 
            CAST(br.AttrValueId AS INT) as brand_id, 
            MAX(i.iid) as max_iid
        FROM DS_ITEMS i
        INNER JOIN DS_ObjectsAttributes br ON i.iid = br.id AND br.attrid = 635 AND br.Activeflag = 1
        WHERE i.activeFlag = 1
        GROUP BY br.AttrValueId
    ) t
    INNER JOIN DS_UnitItems ui ON t.max_iid = ui.iid
    WHERE ui.activeflag = 1 AND ui.UnitLevel = 2;

    -- 4. Итоговая сборка правил с многоуровневой логикой
    SELECT * FROM (
        SELECT 
            ig.itID AS CategoryID,
            ig.brand_id AS GroupID,
            -- ЛОГИКА КВАНТА: Атрибут 937 -> Если NULL, то Резервный квант -> Если NULL, то 1
            ISNULL(
                TRY_CAST(REPLACE(REPLACE(LTRIM(RTRIM(q.attrtext)), ',', '.'), ' ', '') AS INT), 
                ISNULL(bq.BackupQuantum, 1)
            ) AS BrandQuantum,
            
            ISNULL(LTRIM(RTRIM(v.attrtext)), 'Regular') AS ImportanceLabel,
            CASE 
                WHEN v.attrtext LIKE '%Must List%'   THEN 4
                WHEN v.attrtext LIKE '%Drive List%'  THEN 3
                WHEN v.attrtext LIKE '%NPI%'        THEN 2
                WHEN v.attrtext LIKE '%SPEED KPI%'  THEN 1
                ELSE 0 
            END AS PriorityWeight,
            
            ISNULL(rp.AvgRealPrice, lp.AvgListPrice) AS AvgPrice
        FROM (
            SELECT DISTINCT i.itID, CAST(br.AttrValueId AS INT) AS brand_id
            FROM DS_ITEMS i 
            INNER JOIN DS_ObjectsAttributes br ON i.iid = br.id AND br.Activeflag = 1 AND br.AttrId = 635
            WHERE i.activeFlag = 1
        ) ig
        LEFT JOIN #RealPrices rp ON ig.brand_id = rp.brand_id
        LEFT JOIN #ListPrices lp ON ig.brand_id = lp.brand_id
        LEFT JOIN #BackupQuantums bq ON ig.brand_id = bq.brand_id
        LEFT JOIN DS_ObjectsAttributes q ON ig.brand_id = q.id AND q.attrid = 937 AND q.Activeflag = 1
        LEFT JOIN DS_ObjectsAttributes v ON ig.brand_id = v.id AND v.attrid = 2241 AND v.Activeflag = 1
    ) t
    WHERE t.AvgPrice > 0;

    DROP TABLE #ListPrices;
    DROP TABLE #RealPrices;
    DROP TABLE #BackupQuantums;
END
GO
