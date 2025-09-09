import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

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

# Создаем визуализации
fig = plt.figure(figsize=(20, 15))

# 1. График повторных покупок
ax1 = plt.subplot(2, 3, 1)
purchase_categories = ['1 покупка', '2 покупки', '3-5 покупок', '6-10 покупок', '10+ покупок']
purchase_counts = [
    (user_purchases == 1).sum(),
    ((user_purchases >= 2) & (user_purchases < 3)).sum(),
    ((user_purchases >= 3) & (user_purchases < 6)).sum(),
    ((user_purchases >= 6) & (user_purchases < 11)).sum(),
    (user_purchases >= 11).sum()
]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
wedges, texts, autotexts = ax1.pie(purchase_counts, labels=purchase_categories, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax1.set_title('Распределение пользователей по количеству покупок', fontsize=14, fontweight='bold')

# 2. График подписок
ax2 = plt.subplot(2, 3, 2)
sub_names = sub_stats.index
sub_counts = sub_stats.values
bars = ax2.bar(range(len(sub_names)), sub_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
ax2.set_title('Распределение по типам подписок', fontsize=14, fontweight='bold')
ax2.set_xlabel('Тип подписки')
ax2.set_ylabel('Количество покупок')
ax2.set_xticks(range(len(sub_names)))
ax2.set_xticklabels(sub_names, rotation=45, ha='right')
for i, (bar, count) in enumerate(zip(bars, sub_counts)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{count}\n({count/len(df)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10)

# 3. График по дням недели
ax3 = plt.subplot(2, 3, 3)
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts_ordered = [dow_stats.get(day, 0) for day in dow_order]
dow_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
bars = ax3.bar(dow_labels, dow_counts_ordered, color='#4ecdc4')
ax3.set_title('Покупки по дням недели', fontsize=14, fontweight='bold')
ax3.set_xlabel('День недели')
ax3.set_ylabel('Количество покупок')
for i, (bar, count) in enumerate(zip(bars, dow_counts_ordered)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{count}\n({count/len(df)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10)

# 4. График по часам
ax4 = plt.subplot(2, 3, 4)
hours = hourly_stats.index
counts = hourly_stats.values
ax4.plot(hours, counts, marker='o', linewidth=2, markersize=4, color='#45b7d1')
ax4.set_title('Покупки по часам дня', fontsize=14, fontweight='bold')
ax4.set_xlabel('Час')
ax4.set_ylabel('Количество покупок')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, 24, 2))

# 5. Когортная матрица ретеншна
ax5 = plt.subplot(2, 3, 5)
# Берем последние 6 месяцев для лучшей визуализации
cohort_display = retention_matrix.iloc[-6:]
# Ограничиваем количество дней для читаемости
cohort_display = cohort_display.iloc[:, :30]  # Первые 30 дней
sns.heatmap(cohort_display, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Retention (%)'}, ax=ax5)
ax5.set_title('Когортный анализ ретеншна (последние 6 месяцев)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Дни с первой покупки')
ax5.set_ylabel('Когорта (месяц)')

# 6. Динамика ежедневных покупок
ax6 = plt.subplot(2, 3, 6)
daily_purchases = df.groupby('date').size()
ax6.plot(daily_purchases.index, daily_purchases.values, marker='o', markersize=2, color='#96ceb4')
ax6.set_title('Динамика ежедневных покупок', fontsize=14, fontweight='bold')
ax6.set_xlabel('Дата')
ax6.set_ylabel('Количество покупок')
ax6.tick_params(axis='x', rotation=45)
ax6.grid(True, alpha=0.3)

# Добавляем общий заголовок
fig.suptitle('Анализ ретеншна пользователей - Визуализация данных', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('cohort_analysis_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Дополнительная визуализация - Retention кривые
fig2, ax = plt.subplots(figsize=(12, 8))

# Создаем данные для retention кривых
days = [1, 7, 30]
retention_values = [retention_1d / len(all_dates), retention_7d / len(all_dates), retention_30d / len(all_dates)]

ax.plot(days, retention_values, marker='o', linewidth=3, markersize=8, color='#ff6b6b')
ax.set_title('Кривая ретеншна', fontsize=16, fontweight='bold')
ax.set_xlabel('Дни с первой покупки')
ax.set_ylabel('Retention (%)')
ax.set_xticks(days)
ax.grid(True, alpha=0.3)

# Добавляем значения на график
for i, (day, value) in enumerate(zip(days, retention_values)):
    ax.annotate(f'{value:.2f}%', (day, value), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=12, fontweight='bold')

# Добавляем зоны качества
ax.axhspan(0, 5, alpha=0.2, color='red', label='Критически низкий')
ax.axhspan(5, 15, alpha=0.2, color='orange', label='Низкий')
ax.axhspan(15, 30, alpha=0.2, color='yellow', label='Средний')
ax.axhspan(30, 100, alpha=0.2, color='green', label='Высокий')

ax.legend()
plt.tight_layout()
plt.savefig('retention_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n=== ВИЗУАЛИЗАЦИИ СОЗДАНЫ ===')
print('Файлы сохранены:')
print('- cohort_analysis_visualization.png - основные графики')
print('- retention_curve.png - кривая ретеншна')
