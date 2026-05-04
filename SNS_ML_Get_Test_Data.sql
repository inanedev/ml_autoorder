IF OBJECT_ID('dbo.SNS_ML_Get_Test_Data', 'P') IS NOT NULL
    DROP PROCEDURE dbo.SNS_ML_Get_Test_Data;
GO

/**
 * Хранимая процедура для выгрузки тестовых данных для ML модели
 * 
 * Назначение:
 *   Формирование полного набора признаков для прогнозирования продаж
 *   на уровне точка x категория x дата с использованием исторических данных
 *   Возвращает ровно те же столбцы и по той же логике, что и скрипт sns_ml_add_features.py
 * 
 * Параметры:
 *   @TargetDate DATE - Дата, на которую собираются фичи
 * 
 * Возвращаемые колонки:
 *   === Базовые данные (из SNS_ML_Get_Raw_Data) ===
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
 *   === Календарные фичи (из add_calendar_features) ===
 *   DayOfWeek                     - День недели (1-понедельник, 2-вторник, ..., 7-воскресенье)
 *   IsFriday                      - Признак пятницы (1/0)
 *   IsMonday                      - Признак понедельника (1/0)
 *   DaysToNextHoliday             - Количество дней до начала ближайшего национального праздника
 *   DaysSinceLastHoliday          - Количество дней с окончания ближайшего национального праздника
 *   IsPreHoliday                  - Является ли день предпраздничным (до праздника 3 и менее дней)
 *   IsPostHoliday                 - Является ли день постпраздничным (после праздника прошло 3 и менее дней)
 *   Quarter                       - Номер квартала (1-4)
 *   Month                         - Номер месяца (1-12)
 *   WeekOfYear                    - Номер недели в году (1-53)
 *   DayOfMonth                    - День месяца (1-31)
 *   DayOfYear                     - День года (1-366)
 *   isEndOfMonth                  - Бинарная фича: последние 3 дня месяца или первые 2 дня месяца, 
 *                                   если до конца недели <= 2 дней, либо последняя суббота месяца
 *   DayOfYear_sin                 - Синус дня года (циклический признак для сезонности)
 *   DayOfYear_cos                 - Косинус дня года (циклический признак для сезонности)
 *   Month_sin                     - Синус месяца (циклический признак для месячной сезонности)
 *   Month_cos                     - Косинус месяца (циклический признак для месячной сезонности)
 *   DayOfWeek_sin                 - Синус дня недели (циклический признак для недельной сезонности)
 *   DayOfWeek_cos                 - Косинус дня недели (циклический признак для недельной сезонности)
 *   WeekOfYear_sin                - Синус недели года (циклический признак для годовой сезонности)
 *   WeekOfYear_cos                - Косинус недели года (циклический признак для годовой сезонности)
 *   Quarter_sin                   - Синус квартала (циклический признак для квартальной сезонности)
 *   Quarter_cos                   - Косинус квартала (циклический признак для квартальной сезонности)
 *   
 *   Циклические признаки (синус/косинус) кодируют циклическую природу времени,
 *   чтобы модель понимала близость значений на границах циклов (например,
 *   воскресенье близко к понедельнику, декабрь близок к январю).
 *   
 *   === Фичи посещений (из add_visit_features) ===
 *   DaysLastVisit                 - Количество дней с предыдущего VisitDate для PointID (если первое посещение = 7)
 *   DaysNextVisit                 - Количество дней до следующего VisitDate для PointID. Если следующего визита нет в истории,
 *                                   рассчитывается по атрибуту 644 (дни недели посещений: 1-пн, 2-вт, ..., 7-вс). 
 *                                   Находит ближайший день недели вперед от текущего. Если дней нет = 7.
 *   
 *   === Фичи продаж категорий (из add_category_sales_features) ===
 *   DaysLastSalesCategory         - Количество дней с момента последней продажи категории в точку (если первая продажа = 7)
 *   
 *   === Фичи последней продажи категории (из add_last_sales_category_feature) ===
 *   LastSalesCategory             - Сумма последней продажи категории в точку (если первая продажа = 0)
 * 
 * Логика работы:
 *   1. Выбираются все активные точки продаж
 *   2. Отбираются только категории, которые продавались хоть одному клиенту за последний месяц (до @TargetDate)
 *   3. Для каждой точки формируются признаки из SNS_ML_Get_Raw_Data для отобранных категорий
 *   4. Добавляются календарные фичи для TargetDate (аналогично add_calendar_features)
 *   5. Рассчитываются фичи истории посещений (аналогично add_visit_features)
 *   6. Рассчитываются фичи продаж категорий (аналогично add_category_sales_features и add_last_sales_category_feature)
 * 
 * Пример использования:
 *   EXEC dbo.SNS_ML_Get_Test_Data @TargetDate = '2024-01-15';
 * 
 * Примечания:
 *   - Размер сетки микрорегиона: 0.027 градуса (~3 км)
 *   - Фильтруются только активные товары и точки
 *   - Категории ограничены теми, которые продавались за последний месяц (до @TargetDate)
 *   - Координаты преобразуются из строкового формата с заменой запятой на точку
 *   - Все лаговые фичи считаются строго по данным до @TargetDate (не включая саму дату)
 *   - DaysNextVisit рассчитывается по истории визитов; если следующего визита нет, используется атрибут 644
 *     (дни недели посещений в формате '1,3,5' где 1=понедельник, ..., 7=воскресенье)
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

        -- Установка первого дня недели = понедельник (как в Python .weekday() + 1)
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
            ISNULL(MAX(CASE WHEN fa.attrid = 644 THEN fa.attrtext END), '') AS VisitDays,
            
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

        -- 3. Список категорий, которые продавались хоть одному клиенту за последний месяц
        IF OBJECT_ID('tempdb..#Categories') IS NOT NULL DROP TABLE #Categories;
        SELECT DISTINCT m.CategoryID INTO #Categories 
        FROM DS_Orders o 
        INNER JOIN DS_Orders_Items oi ON o.MasterFID = oi.MasterFID AND o.orID = oi.orID 
        INNER JOIN #ItemMap m ON CAST(oi.iID AS INT) = m.iid 
        WHERE o.orType = 1 
          AND o.orDate >= DATEADD(MONTH, -1, @TargetDate)
          AND o.orDate < @TargetDate;
        
        CREATE CLUSTERED INDEX IX_Cat_CategoryID ON #Categories (CategoryID);

        -- 4. Полный грид: все активные точки x все категории для одной даты
        IF OBJECT_ID('tempdb..#FullGrid') IS NOT NULL DROP TABLE #FullGrid;
        SELECT 
            CAST(@TargetDate AS DATE) AS VisitDate,
            pf.PointID,
            c.CategoryID,
            pf.BranchID,
            pf.PointClass,
            pf.PointType,
            pf.Lat,
            pf.Lon,
            pf.MicroRegionID,
            pf.VisitDays
        INTO #FullGrid
        FROM #PointFeatures pf
        CROSS JOIN #Categories c;
        
        CREATE CLUSTERED INDEX IX_FG_DatePointCat ON #FullGrid (VisitDate, PointID, CategoryID);

        -- 5. История посещений для расчета DaysLastVisit и DaysNextVisit (аналог add_visit_features)
        -- Берем данные за последний месяц до @TargetDate
        IF OBJECT_ID('tempdb..#VisitHistory') IS NOT NULL DROP TABLE #VisitHistory;
        SELECT 
            CAST(o.orDate AS DATE) AS VisitDate, 
            o.mfID AS PointID
        INTO #VisitHistory
        FROM DS_Orders o 
        WHERE o.orType = 1 
          AND o.orDate >= DATEADD(MONTH, -1, @TargetDate)
          AND o.orDate < @TargetDate
        GROUP BY CAST(o.orDate AS DATE), o.mfID;
        
        CREATE CLUSTERED INDEX IX_VH_DatePoint ON #VisitHistory (VisitDate, PointID);
        CREATE NONCLUSTERED INDEX IX_VH_Point ON #VisitHistory (PointID, VisitDate);

        -- 6. История продаж по категориям для DaysLastSalesCategory и LastSalesCategory (аналог add_category_sales_features и add_last_sales_category_feature)
        -- Берем данные за последний месяц до @TargetDate
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
          AND o.orDate >= DATEADD(MONTH, -1, @TargetDate)
          AND o.orDate < @TargetDate
        GROUP BY CAST(o.orDate AS DATE), o.mfID, m.CategoryID;
        
        CREATE CLUSTERED INDEX IX_SH_DatePointCat ON #SalesHistory (VisitDate, PointID, CategoryID);
        CREATE NONCLUSTERED INDEX IX_SH_PointCat ON #SalesHistory (PointID, CategoryID, VisitDate);

        -- 7. Основной результат с базовыми данными и календарными фичами
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
            
            -- Календарные фичи (аналог add_calendar_features)
            -- DayOfWeek: 1-понедельник, 7-воскресенье (Python: weekday() + 1)
            DATEPART(WEEKDAY, fg.VisitDate) AS DayOfWeek,
            
            -- IsFriday: пятница = 5
            CASE WHEN DATEPART(WEEKDAY, fg.VisitDate) = 5 THEN 1 ELSE 0 END AS IsFriday,
            
            -- IsMonday: понедельник = 1
            CASE WHEN DATEPART(WEEKDAY, fg.VisitDate) = 1 THEN 1 ELSE 0 END AS IsMonday,
            
            -- Quarter
            DATEPART(QUARTER, fg.VisitDate) AS Quarter,
            
            -- Month
            MONTH(fg.VisitDate) AS Month,
            
            -- WeekOfYear (ISO week)
            DATEPART(WEEK, fg.VisitDate) AS WeekOfYear,
            
            -- DayOfMonth
            DAY(fg.VisitDate) AS DayOfMonth,
            
            -- DayOfYear
            DATEPART(DAYOFYEAR, fg.VisitDate) AS DayOfYear,
            
            -- isEndOfMonth: последние 3 дня месяца ИЛИ (первые 2 дня месяца И до конца недели <= 2 дней) ИЛИ последняя суббота месяца
            CASE 
                WHEN DAY(fg.VisitDate) >= DAY(EOMONTH(fg.VisitDate)) - 2 THEN 1
                WHEN DAY(fg.VisitDate) <= 2 AND DATEPART(WEEKDAY, fg.VisitDate) >= 5 THEN 1
                WHEN DAY(fg.VisitDate) + 7 > DAY(EOMONTH(fg.VisitDate)) AND DATEPART(WEEKDAY, fg.VisitDate) = 6 THEN 1
                ELSE 0 
            END AS isEndOfMonth,
            
            -- Праздники России (фиксированные даты, как в get_russia_holidays)
            CASE WHEN (MONTH(fg.VisitDate) = 1 AND DAY(fg.VisitDate) IN (1,2,3,4,5,6,7,8))
                      OR (MONTH(fg.VisitDate) = 2 AND DAY(fg.VisitDate) = 23)
                      OR (MONTH(fg.VisitDate) = 3 AND DAY(fg.VisitDate) = 8)
                      OR (MONTH(fg.VisitDate) = 5 AND DAY(fg.VisitDate) IN (1,9))
                      OR (MONTH(fg.VisitDate) = 6 AND DAY(fg.VisitDate) = 12)
                      OR (MONTH(fg.VisitDate) = 11 AND DAY(fg.VisitDate) = 4)
                 THEN 1 ELSE 0 END AS IsHoliday,
            
            -- Дни до ближайшего праздника (DaysToNextHoliday)
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
            ), 365) AS DaysToNextHoliday,
            
            -- Дни от последнего праздника (DaysSinceLastHoliday)
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
            ), 365) AS DaysSinceLastHoliday,
            
            -- IsPreHoliday: до праздника 3 и менее дней
            CASE WHEN ISNULL((
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
            ), 365) <= 3 THEN 1 ELSE 0 END AS IsPreHoliday,
            
            -- IsPostHoliday: после праздника прошло 3 и менее дней
            CASE WHEN ISNULL((
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
            ), 365) <= 3 THEN 1 ELSE 0 END AS IsPostHoliday

        INTO #ResultBase
        FROM #FullGrid fg;
        
        CREATE CLUSTERED INDEX IX_RB_DatePointCat ON #ResultBase (VisitDate, PointID, CategoryID);

        -- 8. Расчет DaysLastVisit и DaysNextVisit (аналог add_visit_features)
        -- Для каждой точки находим последний визит до @TargetDate и следующий визит после @TargetDate
        -- в пределах данных за последний месяц
        -- Если следующего визита нет, используем атрибут 644 (дни недели посещений) для расчета DaysNextVisit
        IF OBJECT_ID('tempdb..#VisitWithShifts') IS NOT NULL DROP TABLE #VisitWithShifts;
        WITH LastVisitBefore AS (
            -- Последний визит до @TargetDate для каждой точки
            SELECT 
                PointID,
                MAX(VisitDate) AS LastVisitDate
            FROM #VisitHistory
            WHERE VisitDate < @TargetDate
            GROUP BY PointID
        ),
        NextVisitAfter AS (
            -- Следующий визит после @TargetDate для каждой точки (минимальная дата > @TargetDate)
            SELECT 
                PointID,
                MIN(VisitDate) AS NextVisitDate
            FROM #VisitHistory
            WHERE VisitDate > @TargetDate
            GROUP BY PointID
        ),
        -- Расчет DaysNextVisit на основе атрибута 644 (дни недели посещений: 1-пн, 2-вт, ..., 7-вс)
        -- Находим ближайший день недели вперед от текущего дня недели
        VisitDaysCalc AS (
            SELECT 
                pf.PointID,
                pf.VisitDays,
                DATEPART(WEEKDAY, @TargetDate) AS CurrentDayOfWeek,
                -- Разбиваем строку дней (например '1,3,5') на отдельные дни недели
                -- И находим минимальный день недели, который больше текущего
                -- Используем XML метод вместо STRING_SPLIT для совместимости с SQL Server 2012
                (
                    SELECT MIN(CAST(x.value AS INT))
                    FROM (
                        SELECT t.c.value('.', 'INT') AS value
                        FROM (
                            SELECT CAST('<n>' + REPLACE(pf.VisitDays, ',', '</n><n>') + '</n>' AS XML) AS xmldata
                        ) AS a
                        CROSS APPLY xmldata.nodes('/n') AS t(c)
                    ) x
                    WHERE x.value IS NOT NULL
                      AND x.value > DATEPART(WEEKDAY, @TargetDate)
                ) AS NextVisitDayInWeek,
                -- Если нет дня в текущей неделе, ищем в следующей (минимальный день из списка)
                -- Используем XML метод вместо STRING_SPLIT для совместимости с SQL Server 2012
                (
                    SELECT MIN(CAST(x.value AS INT))
                    FROM (
                        SELECT t.c.value('.', 'INT') AS value
                        FROM (
                            SELECT CAST('<n>' + REPLACE(pf.VisitDays, ',', '</n><n>') + '</n>' AS XML) AS xmldata
                        ) AS a
                        CROSS APPLY xmldata.nodes('/n') AS t(c)
                    ) x
                    WHERE x.value IS NOT NULL
                ) AS MinVisitDayInWeek
            FROM #PointFeatures pf
        )
        SELECT 
            lv.PointID,
            @TargetDate AS VisitDate,
            ISNULL(DATEDIFF(DAY, lv.LastVisitDate, @TargetDate), 7) AS DaysLastVisit,
            CASE 
                -- Если есть следующий визит в истории, используем его
                WHEN nv.NextVisitDate IS NOT NULL THEN DATEDIFF(DAY, @TargetDate, nv.NextVisitDate)
                -- Если визита нет, но есть атрибут 644, рассчитываем по дням недели
                WHEN vdc.VisitDays IS NOT NULL AND vdc.VisitDays <> '' THEN
                    CASE 
                        -- Если есть день недели в текущей неделе после текущего дня
                        WHEN vdc.NextVisitDayInWeek IS NOT NULL 
                            THEN vdc.NextVisitDayInWeek - vdc.CurrentDayOfWeek
                        -- Иначе берем первый день из следующей недели
                        ELSE (7 - vdc.CurrentDayOfWeek) + vdc.MinVisitDayInWeek
                    END
                -- По умолчанию 7
                ELSE 7
            END AS DaysNextVisit
        INTO #VisitWithShifts
        FROM LastVisitBefore lv
        LEFT JOIN NextVisitAfter nv ON lv.PointID = nv.PointID
        LEFT JOIN VisitDaysCalc vdc ON lv.PointID = vdc.PointID;
        
        CREATE CLUSTERED INDEX IX_VWS_PointDate ON #VisitWithShifts (PointID, VisitDate);

        -- 9. Расчет DaysLastSalesCategory (аналог add_category_sales_features)
        -- Для каждой комбинации PointID-CategoryID находим последнюю продажу до @TargetDate
        IF OBJECT_ID('tempdb..#CategorySalesWithShifts') IS NOT NULL DROP TABLE #CategorySalesWithShifts;
        SELECT 
            sh.PointID,
            sh.CategoryID,
            @TargetDate AS VisitDate,
            ISNULL(DATEDIFF(DAY, sh.LastSaleDate, @TargetDate), 7) AS DaysLastSalesCategory
        INTO #CategorySalesWithShifts
        FROM (
            SELECT 
                PointID,
                CategoryID,
                MAX(VisitDate) AS LastSaleDate
            FROM #SalesHistory
            WHERE VisitDate < @TargetDate
            GROUP BY PointID, CategoryID
        ) sh;
        
        CREATE CLUSTERED INDEX IX_CSW_PointCatDate ON #CategorySalesWithShifts (PointID, CategoryID, VisitDate);

        -- 10. Расчет LastSalesCategory (аналог add_last_sales_category_feature)
        -- Для каждой комбинации PointID-CategoryID находим сумму последней продажи до @TargetDate
        IF OBJECT_ID('tempdb..#DailyCategorySales') IS NOT NULL DROP TABLE #DailyCategorySales;
        WITH LastSale AS (
            SELECT 
                PointID,
                CategoryID,
                VisitDate,
                SumRoubles,
                ROW_NUMBER() OVER (PARTITION BY PointID, CategoryID ORDER BY VisitDate DESC) AS rn
            FROM #SalesHistory
            WHERE VisitDate < @TargetDate
        )
        SELECT 
            PointID,
            CategoryID,
            @TargetDate AS VisitDate,
            ISNULL(SumRoubles, 0) AS LastSalesCategory
        INTO #DailyCategorySales
        FROM LastSale
        WHERE rn = 1;
        
        CREATE CLUSTERED INDEX IX_DCS_PointCatDate ON #DailyCategorySales (PointID, CategoryID, VisitDate);

        -- 11. Финальный результат: объединение всех фичей
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
            rb.DayOfWeek,
            rb.IsFriday,
            rb.IsMonday,
            rb.DaysToNextHoliday,
            rb.DaysSinceLastHoliday,
            rb.IsPreHoliday,
            rb.IsPostHoliday,
            rb.Quarter,
            rb.Month,
            rb.WeekOfYear,
            rb.DayOfMonth,
            rb.DayOfYear,
            rb.isEndOfMonth,
            
            -- Циклические временные признаки (синус/косинус)
            SIN(2.0 * PI() * CAST(rb.DayOfYear AS FLOAT) / 366.0) AS DayOfYear_sin,
            COS(2.0 * PI() * CAST(rb.DayOfYear AS FLOAT) / 366.0) AS DayOfYear_cos,
            SIN(2.0 * PI() * CAST(rb.Month AS FLOAT) / 12.0) AS Month_sin,
            COS(2.0 * PI() * CAST(rb.Month AS FLOAT) / 12.0) AS Month_cos,
            SIN(2.0 * PI() * CAST(rb.DayOfWeek AS FLOAT) / 7.0) AS DayOfWeek_sin,
            COS(2.0 * PI() * CAST(rb.DayOfWeek AS FLOAT) / 7.0) AS DayOfWeek_cos,
            SIN(2.0 * PI() * CAST(rb.WeekOfYear AS FLOAT) / 53.0) AS WeekOfYear_sin,
            COS(2.0 * PI() * CAST(rb.WeekOfYear AS FLOAT) / 53.0) AS WeekOfYear_cos,
            SIN(2.0 * PI() * CAST(rb.Quarter AS FLOAT) / 4.0) AS Quarter_sin,
            COS(2.0 * PI() * CAST(rb.Quarter AS FLOAT) / 4.0) AS Quarter_cos,
            
            -- Фичи посещений (DaysLastVisit, DaysNextVisit)
            ISNULL(vws.DaysLastVisit, 7) AS DaysLastVisit,
            ISNULL(vws.DaysNextVisit, 7) AS DaysNextVisit,
            
            -- Фичи продаж категорий (DaysLastSalesCategory)
            ISNULL(csw.DaysLastSalesCategory, 7) AS DaysLastSalesCategory,
            
            -- Фичи последней продажи категории (LastSalesCategory)
            ISNULL(dcs.LastSalesCategory, 0) AS LastSalesCategory
            
        FROM #ResultBase rb
        LEFT JOIN #VisitWithShifts vws ON rb.PointID = vws.PointID AND rb.VisitDate = vws.VisitDate
        LEFT JOIN #CategorySalesWithShifts csw ON rb.PointID = csw.PointID 
                                              AND rb.CategoryID = csw.CategoryID 
                                              AND rb.VisitDate = csw.VisitDate
        LEFT JOIN #DailyCategorySales dcs ON rb.PointID = dcs.PointID 
                                          AND rb.CategoryID = dcs.CategoryID 
                                          AND rb.VisitDate = dcs.VisitDate
        ORDER BY rb.VisitDate, rb.PointID, rb.CategoryID;

        -- Очистка временных таблиц
        DROP TABLE #ItemMap;
        DROP TABLE #PointFeatures;
        DROP TABLE #Categories;
        DROP TABLE #FullGrid;
        DROP TABLE #VisitHistory;
        DROP TABLE #SalesHistory;
        DROP TABLE #ResultBase;
        DROP TABLE #VisitWithShifts;
        DROP TABLE #CategorySalesWithShifts;
        DROP TABLE #DailyCategorySales;
        
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
        IF OBJECT_ID('tempdb..#VisitHistory') IS NOT NULL DROP TABLE #VisitHistory;
        IF OBJECT_ID('tempdb..#SalesHistory') IS NOT NULL DROP TABLE #SalesHistory;
        IF OBJECT_ID('tempdb..#ResultBase') IS NOT NULL DROP TABLE #ResultBase;
        IF OBJECT_ID('tempdb..#VisitWithShifts') IS NOT NULL DROP TABLE #VisitWithShifts;
        IF OBJECT_ID('tempdb..#CategorySalesWithShifts') IS NOT NULL DROP TABLE #CategorySalesWithShifts;
        IF OBJECT_ID('tempdb..#DailyCategorySales') IS NOT NULL DROP TABLE #DailyCategorySales;
        
        -- Проброс ошибки дальше
        RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
        RETURN -1;
    END CATCH
END;
GO
