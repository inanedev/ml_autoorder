IF OBJECT_ID('SNS_ML_Recommendations', 'U') IS NOT NULL
    DROP TABLE SNS_ML_Recommendations;

CREATE TABLE SNS_ML_Recommendations (
    CalculationDate DATE,
    PointID INT,
    CategoryID INT,       -- Новое поле
    GroupID INT,
    RecQty FLOAT,
    IsPriority BIT,
    Reason NVARCHAR(50),
    CatBudget MONEY,      -- Новое поле (лимит категории +20%)
    GroupBudget MONEY,   -- Новое поле (прогноз CatBoost для бренда)
    PRIMARY KEY (CalculationDate, PointID, GroupID)
);