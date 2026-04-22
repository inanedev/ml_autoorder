-- ============================================================================
-- Generate Test Dataset for ML Model
-- Назначение: Создание тестового датасета для прогноза суммы закупки
-- Формат: Одна строка на точку на сегодня со всеми фичами (без целевой переменной)
-- ============================================================================

SET NOCOUNT ON;

-- ============================================================================
-- 1. Определяем сегодняшнюю дату
-- ============================================================================
DECLARE @Today DATE = CAST(GETDATE() AS DATE);

-- ============================================================================
-- 2. Получаем все активные точки продаж (ftype=1, factiveflag=1)
-- ============================================================================
WITH ActiveFaces AS (
    SELECT 
        fid AS FaceID
    FROM ds_faces
    WHERE ftype = 1 
      AND factiveflag = 1
),

-- ============================================================================
-- 3. Календарь дат (история за последний год для расчета фичей)
-- ============================================================================
DateRange AS (
    SELECT TOP (365)
        DATEADD(DAY, -ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) + 1, @Today) AS DateValue
    FROM sys.objects o1
    CROSS JOIN sys.objects o2
),

-- ============================================================================
-- 4. Агрегация продаж по дням и категориям (исторические данные)
-- Логика полностью идентична SNS_ML_Get_Raw_Data.sql
-- ============================================================================
SalesHistory AS (
    SELECT 
        CAST(o.orDate AS DATE) AS SaleDate,
        o.mfID AS FaceID,
        m.CategoryID,
        SUM(ISNULL(oi.forsumroubles, 0)) AS TotalAmount,
        COUNT(DISTINCT o.orID) AS OrderCount
    FROM DS_Orders o 
    INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
    INNER JOIN (
        -- Справочник SKU -> Категория (как в SNS_ML_Get_Raw_Data)
        SELECT 
            CAST(i.iid AS INT) as iid, 
            CAST(i.itID AS INT) AS CategoryID
        FROM DS_ITEMS i 
        WHERE i.activeFlag = 1 AND i.itID IS NOT NULL
    ) m ON CAST(oi.iID AS INT) = m.iid 
    INNER JOIN ds_faces f ON o.mfID = f.fid
    WHERE o.orType = 1 
      AND o.orDate >= DATEADD(DAY, -60, @Today) -- 60 дней истории для расчетов
      AND f.ftype = 1 
      AND f.factiveflag = 1
    GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID
),

-- ============================================================================
-- 5. История заказов по точкам (для расчета дней с последнего заказа)
-- ============================================================================
LastOrderCategory AS (
    SELECT 
        FaceID,
        CategoryID,
        MAX(SaleDate) AS LastOrderDate
    FROM SalesHistory
    GROUP BY FaceID, CategoryID
),

LastOrderTotal AS (
    SELECT 
        FaceID,
        MAX(SaleDate) AS LastOrderDate
    FROM SalesHistory
    GROUP BY FaceID
),

-- ============================================================================
-- 6. Средние интервалы между заказами по категории
-- ============================================================================
OrderIntervals AS (
    SELECT 
        FaceID,
        CategoryID,
        AVG(DATEDIFF(DAY, PrevDate, SaleDate)) AS AvgInterval
    FROM (
        SELECT 
            FaceID,
            CategoryID,
            SaleDate,
            LAG(SaleDate) OVER (PARTITION BY FaceID, CategoryID ORDER BY SaleDate) AS PrevDate
        FROM SalesHistory
    ) t
    WHERE PrevDate IS NOT NULL
    GROUP BY FaceID, CategoryID
),

-- ============================================================================
-- 7. Лаговые признаки продаж (предыдущий заказ, SMA, импульс, отклонение)
-- ============================================================================
SalesWithLags AS (
    SELECT 
        FaceID,
        CategoryID,
        SaleDate,
        TotalAmount,
        LAG(TotalAmount) OVER (PARTITION BY FaceID, CategoryID ORDER BY SaleDate) AS PrevAmount,
        AVG(TotalAmount) OVER (
            PARTITION BY FaceID, CategoryID 
            ORDER BY SaleDate 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS SMA_3,
        AVG(TotalAmount) OVER (
            PARTITION BY FaceID, CategoryID 
            ORDER BY SaleDate 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS SMA_7,
        AVG(TotalAmount) OVER (
            PARTITION BY FaceID, CategoryID 
            ORDER BY SaleDate 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS SMA_30,
        STDEV(TotalAmount) OVER (
            PARTITION BY FaceID, CategoryID 
            ORDER BY SaleDate 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS StdDev_30
    FROM SalesHistory
),

-- Последние значения лаговых признаков для каждой точки и категории
LatestSalesFeatures AS (
    SELECT 
        FaceID,
        CategoryID,
        PrevAmount,
        SMA_3,
        SMA_7,
        SMA_30,
        StdDev_30
    FROM (
        SELECT 
            FaceID,
            CategoryID,
            PrevAmount,
            SMA_3,
            SMA_7,
            SMA_30,
            StdDev_30,
            ROW_NUMBER() OVER (PARTITION BY FaceID, CategoryID ORDER BY SaleDate DESC) AS rn
        FROM SalesWithLags
    ) t
    WHERE rn = 1
),

-- ============================================================================
-- 8. Расчет импульса (Momentum)
-- ============================================================================
MomentumCalc AS (
    SELECT 
        FaceID,
        CategoryID,
        AVG(CASE WHEN SaleDate >= DATEADD(DAY, -7, @Today) THEN TotalAmount END) AS AvgWeek,
        AVG(CASE WHEN SaleDate >= DATEADD(DAY, -30, @Today) THEN TotalAmount END) AS AvgMonth
    FROM SalesHistory
    GROUP BY FaceID, CategoryID
),

MomentumFeatures AS (
    SELECT 
        FaceID,
        CategoryID,
        CASE 
            WHEN AvgMonth > 0 THEN AvgWeek / AvgMonth 
            ELSE NULL 
        END AS Momentum
    FROM MomentumCalc
),

-- ============================================================================
-- 9. Календарные фичи для сегодняшней даты
-- ============================================================================
CalendarFeatures AS (
    SELECT 
        @Today AS DateValue,
        DATEPART(WEEKDAY, @Today) AS DayOfWeek,
        CASE WHEN DATEPART(WEEKDAY, @Today) IN (1, 7) THEN 1 ELSE 0 END AS IsWeekend,
        CASE WHEN DATEPART(WEEKDAY, @Today) = 2 THEN 1 ELSE 0 END AS IsMonday,
        CASE WHEN DATEPART(WEEKDAY, @Today) = 6 THEN 1 ELSE 0 END AS IsFriday,
        CASE WHEN DATEPART(WEEKDAY, @Today) = 7 THEN 1 ELSE 0 END AS IsSaturday,
        CASE WHEN DATEPART(WEEKDAY, @Today) = 1 THEN 1 ELSE 0 END AS IsSunday,
        DATEPART(MONTH, @Today) AS Month,
        DATEPART(QUARTER, @Today) AS Quarter,
        DATEPART(WEEK, @Today) AS WeekOfYear,
        CASE WHEN DAY(@Today) = 1 THEN 1 ELSE 0 END AS IsMonthStart,
        CASE WHEN DAY(@Today) = DAY(EOMONTH(@Today)) THEN 1 ELSE 0 END AS IsMonthEnd,
        DAY(@Today) AS DayOfMonth,
        DATEPART(DAYOFYEAR, @Today) AS DayOfYear,
        -- Праздники РФ (фиксированные даты)
        CASE WHEN @Today IN (
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
            '2024-02-23', '2024-03-08', '2024-05-01', '2024-05-09', '2024-06-12', '2024-11-04',
            '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08',
            '2025-02-23', '2025-03-08', '2025-05-01', '2025-05-09', '2025-06-12', '2025-11-04'
        ) THEN 1 ELSE 0 END AS IsHoliday,
        -- Предпраздничные дни
        CASE WHEN DATEADD(DAY, 1, @Today) IN (
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
            '2024-02-23', '2024-03-08', '2024-05-01', '2024-05-09', '2024-06-12', '2024-11-04',
            '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08',
            '2025-02-23', '2025-03-08', '2025-05-01', '2025-05-09', '2025-06-12', '2025-11-04'
        ) THEN 1 ELSE 0 END AS IsPreHoliday,
        -- Постпраздничные дни (после выходных)
        CASE WHEN DATEADD(DAY, -1, @Today) IN (
            '2024-01-01', '2024-01-07', '2024-02-23', '2024-03-08', '2024-05-01', '2024-05-09', '2024-06-12', '2024-11-04',
            '2025-01-01', '2025-01-07', '2025-02-23', '2025-03-08', '2025-05-01', '2025-05-09', '2025-06-12', '2025-11-04'
        ) THEN 1 ELSE 0 END AS IsPostHoliday
),

-- ============================================================================
-- 10. Атрибуты точек продаж
-- ============================================================================
FaceAttributes AS (
    SELECT 
        f.fid AS FaceID,
        f.fcode AS FaceCode,
        f.fname AS FaceName,
        f.fclass AS FaceClass,
        f.ftype AS FaceType,
        f.faddress AS Address,
        f.latitude AS Latitude,
        f.longitude AS Longitude,
        -- Микрорегион (сетка 3×3 км)
        CONCAT(
            ROUND(f.latitude / 0.03, 0) * 0.03, '_', 
            ROUND(f.longitude / 0.03, 0) * 0.03
        ) AS MicroRegion
    FROM ds_faces f
    WHERE f.ftype = 1 AND f.factiveflag = 1
),

-- ============================================================================
-- 11. Список категорий для каждой точки (на основе истории)
-- ============================================================================
FaceCategories AS (
    SELECT DISTINCT
        FaceID,
        CategoryID
    FROM SalesHistory
),

-- ============================================================================
-- 12. Финальный датасет: объединяем все фичи
-- ============================================================================
FinalDataset AS (
    SELECT 
        -- Идентификаторы
        fc.FaceID,
        fa.FaceCode,
        fa.FaceName,
        fa.FaceClass,
        fa.MicroRegion,
        fc.CategoryID,
        
        -- Дата
        cf.DateValue,
        
        -- Атрибуты точки
        fa.Address,
        fa.Latitude,
        fa.Longitude,
        
        -- Календарные фичи
        cf.DayOfWeek,
        cf.IsWeekend,
        cf.IsMonday,
        cf.IsFriday,
        cf.IsSaturday,
        cf.IsSunday,
        cf.Month,
        cf.Quarter,
        cf.WeekOfYear,
        cf.IsMonthStart,
        cf.IsMonthEnd,
        cf.DayOfMonth,
        cf.DayOfYear,
        cf.IsHoliday,
        cf.IsPreHoliday,
        cf.IsPostHoliday,
        
        -- История заказов
        DATEDIFF(DAY, loc.LastOrderDate, @Today) AS Days_Since_Last_Order_Category,
        DATEDIFF(DAY, lot.LastOrderDate, @Today) AS Days_Since_Last_Order_Total,
        oi.AvgInterval AS Average_Interval_Category,
        
        -- Продажи и тренды
        lsf.PrevAmount AS Prev_Order_Amount_Category,
        lsf.SMA_3 AS SMA_3_Category,
        lsf.SMA_7 AS SMA_7_Category,
        lsf.SMA_30 AS SMA_30_Category,
        mf.Momentum AS Momentum_Category,
        lsf.StdDev_30 AS StdDev_Category
        
    FROM FaceCategories fc
    JOIN FaceAttributes fa ON fc.FaceID = fa.FaceID
    CROSS JOIN CalendarFeatures cf
    LEFT JOIN LastOrderCategory loc ON fc.FaceID = loc.FaceID AND fc.CategoryID = loc.CategoryID
    LEFT JOIN LastOrderTotal lot ON fc.FaceID = lot.FaceID
    LEFT JOIN OrderIntervals oi ON fc.FaceID = oi.FaceID AND fc.CategoryID = oi.CategoryID
    LEFT JOIN LatestSalesFeatures lsf ON fc.FaceID = lsf.FaceID AND fc.CategoryID = lsf.CategoryID
    LEFT JOIN MomentumFeatures mf ON fc.FaceID = mf.FaceID AND fc.CategoryID = mf.CategoryID
)

-- ============================================================================
-- ВЫВОД: Тестовый датасет на сегодня
-- ============================================================================
SELECT * 
FROM FinalDataset
ORDER BY FaceID, CategoryID;
