import json
import pandas as pd
from datetime import datetime, timedelta

# Загружаем данные
with open('user_logs_paid_241024_250909.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df['date'] = df['datetime'].dt.date

print('=== ОСНОВНАЯ СТАТИСТИКА ===')
print(f'Общее количество записей: {len(df)}')
print(f'Уникальных пользователей: {df["telegram_id"].nunique()}')
print(f'Период данных: {df["date"].min()} - {df["date"].max()}')
print(f'Среднее количество покупок на пользователя: {len(df) / df["telegram_id"].nunique():.2f}')

print('\n=== АНАЛИЗ ПОВТОРНЫХ ПОКУПОК ===')
user_purchases = df.groupby('telegram_id').size()
print(f'Пользователей с 1 покупкой: {(user_purchases == 1).sum()} ({(user_purchases == 1).sum() / len(user_purchases) * 100:.1f}%)')
print(f'Пользователей с 2+ покупками: {(user_purchases > 1).sum()} ({(user_purchases > 1).sum() / len(user_purchases) * 100:.1f}%)')
print(f'Пользователей с 5+ покупками: {(user_purchases >= 5).sum()} ({(user_purchases >= 5).sum() / len(user_purchases) * 100:.1f}%)')

print('\n=== АНАЛИЗ ПОДПИСОК ===')
sub_stats = df['sub_name'].value_counts()
for sub, count in sub_stats.items():
    print(f'{sub}: {count} ({count/len(df)*100:.1f}%)')

print('\n=== АНАЛИЗ ПО ДНЯМ НЕДЕЛИ ===')
df['day_of_week'] = df['datetime'].dt.day_name()
dow_stats = df['day_of_week'].value_counts()
for day, count in dow_stats.items():
    print(f'{day}: {count} ({count/len(df)*100:.1f}%)')

print('\n=== АНАЛИЗ ПО ЧАСАМ ===')
df['hour'] = df['datetime'].dt.hour
hourly_stats = df['hour'].value_counts().sort_index()
print('Топ-5 часов активности:')
for hour, count in hourly_stats.head().items():
    print(f'{hour:02d}:00 - {count} покупок ({count/len(df)*100:.1f}%)')

print('\n=== РАСЧЕТ RETENTION ===')
# Rolling retention calculation
first_purchase = df.groupby('telegram_id')['date'].min().reset_index()
first_purchase.columns = ['telegram_id', 'first_purchase_date']

df_with_cohort = df.merge(first_purchase, on='telegram_id')
df_with_cohort['days_since_first'] = (pd.to_datetime(df_with_cohort['date']) - pd.to_datetime(df_with_cohort['first_purchase_date'])).dt.days

# Calculate retention for different periods
retention_1d = 0
retention_7d = 0
retention_30d = 0

all_dates = sorted(df['date'].unique())
for current_date in all_dates:
    users_today = set(df[df['date'] == current_date]['telegram_id'])
    
    if len(users_today) == 0:
        continue
    
    # Day 1 retention
    check_date_1d = current_date + timedelta(days=1)
    users_check_1d = set(df[df['date'] == check_date_1d]['telegram_id'])
    retained_1d = users_today.intersection(users_check_1d)
    retention_1d += len(retained_1d) / len(users_today) * 100 if len(users_today) > 0 else 0
    
    # Day 7 retention
    check_date_7d = current_date + timedelta(days=7)
    users_check_7d = set(df[df['date'] == check_date_7d]['telegram_id'])
    retained_7d = users_today.intersection(users_check_7d)
    retention_7d += len(retained_7d) / len(users_today) * 100 if len(users_today) > 0 else 0
    
    # Day 30 retention
    check_date_30d = current_date + timedelta(days=30)
    users_check_30d = set(df[df['date'] == check_date_30d]['telegram_id'])
    retained_30d = users_today.intersection(users_check_30d)
    retention_30d += len(retained_30d) / len(users_today) * 100 if len(users_today) > 0 else 0

print(f'Средний Retention Day 1: {retention_1d / len(all_dates):.2f}%')
print(f'Средний Retention Day 7: {retention_7d / len(all_dates):.2f}%')
print(f'Средний Retention Day 30: {retention_30d / len(all_dates):.2f}%')

print('\n=== АНАЛИЗ КОГОРТ ===')
df_with_cohort['cohort_month'] = pd.to_datetime(df_with_cohort['first_purchase_date']).dt.to_period('M')
cohort_data = df_with_cohort.groupby(['cohort_month', 'days_since_first'])['telegram_id'].nunique().reset_index()
cohort_data.columns = ['cohort_month', 'period', 'users']

cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period', values='users')
cohort_pivot = cohort_pivot.fillna(0)

cohort_sizes = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.div(cohort_sizes, axis=0) * 100

print('Когортный анализ (последние 3 месяца):')
print(retention_matrix.iloc[-3:].round(1))
