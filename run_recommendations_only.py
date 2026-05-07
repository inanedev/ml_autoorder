"""
Скрипт для запуска пайплайна рекомендаций без обучения модели.

Этот скрипт начинает работу сразу с этапа формирования рекомендаций,
используя уже существующие предсказания из таблицы SNS_ML_Predictions.

Использование:
    python run_recommendations_only.py --point-id 123 --category-id 456
    python run_recommendations_only.py --days-until-visit 7
    
Аргументы:
    --point-id: ID точки для рекомендации (опционально, по умолчанию все точки)
    --category-id: ID категории для рекомендации (опционально, по умолчанию все категории)
    --days-until-visit: Дней до следующего визита (по умолчанию 7)
    --verbose: Включить подробное логгирование
    --skip-save: Не сохранять рекомендации в БД (только вывод на экран)
"""

import argparse
import logging
from datetime import datetime
from typing import Optional, List, Tuple
import pandas as pd

# Импорт модулей
from fetch_raw_data import get_connection
from order_recommendation import OrderRecommender
from save_recommendation import RecommendationStorage

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Запуск пайплайна рекомендаций без обучения модели (использует готовые предсказания)'
    )
    
    # Аргументы для рекомендаций
    parser.add_argument(
        '--point-id',
        type=int,
        help='ID торговой точки для рекомендации (по умолчанию: все точки из прогнозов)'
    )
    parser.add_argument(
        '--category-id',
        type=int,
        help='ID категории для рекомендации (по умолчанию: все категории из прогнозов)'
    )
    parser.add_argument(
        '--days-until-visit',
        type=int,
        default=7,
        help='Количество дней до следующего визита (по умолчанию: 7)'
    )
    
    # Дополнительные опции
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Включить подробный режим логгирования'
    )
    parser.add_argument(
        '--skip-save',
        action='store_true',
        help='Не сохранять рекомендации в БД (только вывод на экран)'
    )
    
    return parser.parse_args()


def load_predictions_from_db(
    point_id: Optional[int] = None,
    category_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Загружает предсказания сумм из таблицы SNS_ML_Predictions.
    
    Args:
        point_id: Фильтр по ID точки (опционально)
        category_id: Фильтр по ID категории (опционально)
        
    Returns:
        DataFrame с предсказаниями
    """
    conn = None
    try:
        conn = get_connection()
        
        query = """
            SELECT 
                PointID,
                CategoryID,
                predict_new,
                DaysNextVisit,
                VisitDate
            FROM dbo.SNS_ML_Predictions 
            WHERE 1=1
        """
        params = []
        
        if point_id is not None:
            query += " AND PointId = ?"
            params.append(point_id)
            
        if category_id is not None:
            query += " AND CategoryId = ?"
            params.append(category_id)
        
        logger.info(f"Загрузка предсказаний из БД...")
        df = pd.read_sql(query, conn, params=params)
        
        if not df.empty:
            # Нормализация имен колонок
            df.columns = [col.lower() for col in df.columns]
            logger.info(f"Загружено {len(df)} записей предсказаний")
        else:
            logger.warning("Предсказания не найдены в таблице SNS_ML_Predictions")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке предсказаний: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


def generate_brand_recommendations(
    predictions_df: pd.DataFrame,
    days_until_visit: int = 7,
    reference_date: Optional[datetime] = None
) -> List[Tuple[int, int, float, pd.DataFrame]]:
    """
    Генерирует рекомендации по брендам для всех точек и категорий из предсказаний.
    
    Использует оптимизированный пакетный метод generate_recommendations_batch().
    
    Args:
        predictions_df: DataFrame с предсказаниями сумм
        days_until_visit: Дней до следующего визита
        reference_date: Дата расчета (по умолчанию сегодня)
        
    Returns:
        Список кортежей (point_id, category_id, forecast_amount, recommendation_df)
    """
    if reference_date is None:
        reference_date = datetime.now().date()
    
    recommender = OrderRecommender()
    
    logger.info(f"Генерация рекомендаций для {len(predictions_df)} записей прогнозов...")
    
    # Используем новый оптимизированный пакетный метод
    results = recommender.generate_recommendations_batch(
        predictions_df=predictions_df,
        days_until_visit=days_until_visit,
        reference_date=reference_date,
        force_refresh=True
    )
    
    logger.info(f"Сгенерировано рекомендаций для {len(results)} пар точка-категория")
    return results


def save_all_recommendations(
    recommendations: List[Tuple[int, int, float, pd.DataFrame]],
    reference_date: datetime,
    model_version: str = "v1.0.0"
) -> int:
    """
    Сохраняет все рекомендации в базу данных.
    
    Args:
        recommendations: Список кортежей с рекомендациями
        reference_date: Дата расчета
        model_version: Версия модели
        
    Returns:
        Количество сохраненных записей
    """
    storage = RecommendationStorage()
    total_saved = 0
    
    logger.info(f"Сохранение {len(recommendations)} наборов рекомендаций в БД...")
    
    for idx, (point_id, category_id, forecast_amount, rec_df) in enumerate(recommendations):
        try:
            # Получаем days_until_visit из первой строки recommendation DataFrame
            # Если колонка существует, используем её, иначе fallback на 7
            if 'days_until_visit' in rec_df.columns and len(rec_df) > 0:
                save_days_until_visit = int(rec_df['days_until_visit'].iloc[0])
            else:
                save_days_until_visit = 7
            
            saved_count = storage.save_recommendation(
                recommendation_df=rec_df,
                point_id=point_id,
                category_id=category_id,
                forecast_amount=forecast_amount,
                days_until_visit=save_days_until_visit,
                reference_date=reference_date,
                model_version=model_version
            )
            total_saved += saved_count
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Сохранено {idx + 1}/{len(recommendations)} наборов рекомендаций ({total_saved} записей)")
                
        except Exception as e:
            logger.error(f"Ошибка при сохранении рекомендации для точки {point_id}, категории {category_id}: {e}")
            continue
    
    logger.info(f"Всего сохранено {total_saved} записей рекомендаций")
    return total_saved


def print_recommendations_summary(recommendations: List[Tuple[int, int, float, pd.DataFrame]]):
    """Выводит краткую сводку по рекомендациям."""
    if not recommendations:
        print("\n=== Рекомендации не сгенерированы ===")
        return
    
    print("\n" + "="*80)
    print("СВОДКА ПО РЕКОМЕНДАЦИЯМ")
    print("="*80)
    
    total_brands = 0
    total_estimated_cost = 0.0
    
    for point_id, category_id, forecast_amount, rec_df in recommendations:
        n_brands = len(rec_df[rec_df['included']])
        estimated_cost = rec_df[rec_df['included']]['estimated_cost'].sum() if 'estimated_cost' in rec_df.columns else 0.0
        
        print(f"\nТочка {point_id}, Категория {category_id}:")
        print(f"  Прогноз суммы: {forecast_amount:,.2f} руб.")
        print(f"  Рекомендовано брендов: {n_brands}")
        print(f"  Расчетная стоимость: {estimated_cost:,.2f} руб.")
        
        if n_brands > 0:
            top_brands = rec_df[rec_df['included']].head(5)
            print(f"  Топ-5 брендов:")
            for _, row in top_brands.iterrows():
                brand_name = row.get('brand_name', 'N/A')
                qty = row.get('recommended_qty', 0)
                cost = row.get('estimated_cost', 0.0)
                print(f"    - {brand_name}: {qty} шт. ({cost:,.2f} руб.)")
        
        total_brands += n_brands
        total_estimated_cost += estimated_cost
    
    print("\n" + "-"*80)
    print(f"ВСЕГО:")
    print(f"  Пар точка-категория: {len(recommendations)}")
    print(f"  Всего рекомендованных позиций: {total_brands}")
    print(f"  Общая расчетная стоимость: {total_estimated_cost:,.2f} руб.")
    print("="*80)


def main():
    """Основная функция пайплайна рекомендаций."""
    args = parse_arguments()
    
    # Настройка уровня логгирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "="*80)
    print("ЗАПУСК ПАЙПЛАЙНА ГЕНЕРАЦИИ РЕКОМЕНДАЦИЙ (БЕЗ ОБУЧЕНИЯ МОДЕЛИ)")
    print("="*80)
    
    print(f"\nДней до следующего визита: {args.days_until_visit}")
    
    # ==================== ЭТАП 1: Загрузка предсказаний ====================
    print("\n" + "-"*80)
    print("ЭТАП 1: Загрузка предсказаний сумм из БД")
    print("-"*80)
    
    predictions_df = load_predictions_from_db(
        point_id=args.point_id,
        category_id=args.category_id
    )
    
    if predictions_df.empty:
        logger.error("Не удалось загрузить предсказания. Проверьте таблицу SNS_ML_Predictions")
        print("\n✗ Ошибка: предсказания не найдены. Пайплайн остановлен.")
        print("\nПодсказка: Сначала запустите полный пайплайн с обучением модели:")
        print("  python run_full_pipeline.py --start-date 2024-01-01 --end-date 2024-12-31")
        return
    
    print(f"✓ Загружено {len(predictions_df)} записей предсказаний")
    
    # ==================== ЭТАП 2: Генерация рекомендаций по брендам ====================
    print("\n" + "-"*80)
    print("ЭТАП 2: Генерация рекомендаций по брендам")
    print("-"*80)
    
    reference_date = datetime.now().date()
    
    recommendations = generate_brand_recommendations(
        predictions_df=predictions_df,
        days_until_visit=args.days_until_visit,
        reference_date=reference_date
    )
    
    if not recommendations:
        logger.warning("Рекомендации не были сгенерированы")
        print("\n⚠ Предупреждение: рекомендации не сгенерированы")
    else:
        print(f"\n✓ Сгенерировано рекомендаций для {len(recommendations)} пар точка-категория")
    
    # ==================== ЭТАП 3: Сохранение рекомендаций ====================
    if not args.skip_save and recommendations:
        print("\n" + "-"*80)
        print("ЭТАП 3: Сохранение рекомендаций в БД")
        print("-"*80)
        
        total_saved = save_all_recommendations(
            recommendations=recommendations,
            reference_date=reference_date,
            model_version="v1.0.0"
        )
        
        print(f"\n✓ Сохранено {total_saved} записей рекомендаций в SNS_ML_Brand_Recommendations")
    elif args.skip_save:
        print("\n" + "-"*80)
        print("ЭТАП 3: Пропущен (флаг --skip-save)")
        print("-"*80)
    
    # ==================== ФИНАЛЬНАЯ СВОДКА ====================
    print_recommendations_summary(recommendations)
    
    print("\n" + "="*80)
    print("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
