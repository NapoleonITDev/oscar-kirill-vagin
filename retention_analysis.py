import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

class RetentionAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
        self.df = None
        self.load_data()
    
    def load_data(self):
        print("Загрузка данных...")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.df = pd.DataFrame(self.data)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')
        self.df = self.df.dropna(subset=['datetime'])
        self.df['date'] = self.df['datetime'].dt.date
        self.df = self.df.sort_values('datetime')
        
        print(f"Загружено {len(self.df)} записей")
        print(f"Период: {self.df['date'].min()} - {self.df['date'].max()}")
        print(f"Уникальных пользователей: {self.df['telegram_id'].nunique()}")
    
    def calculate_cohort_retention(self):
        print("\n=== АНАЛИЗ КОГОРТНОГО РЕТЕНШНА ===")
        
        first_purchase = self.df.groupby('telegram_id')['date'].min().reset_index()
        first_purchase.columns = ['telegram_id', 'first_purchase_date']
        
        df_with_cohort = self.df.merge(first_purchase, on='telegram_id')
        df_with_cohort['days_since_first'] = (pd.to_datetime(df_with_cohort['date']) - pd.to_datetime(df_with_cohort['first_purchase_date'])).dt.days
        df_with_cohort['cohort_month'] = pd.to_datetime(df_with_cohort['first_purchase_date']).dt.to_period('M')
        
        cohort_data = df_with_cohort.groupby(['cohort_month', 'days_since_first'])['telegram_id'].nunique().reset_index()
        cohort_data.columns = ['cohort_month', 'period', 'users']
        
        cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period', values='users')
        cohort_pivot = cohort_pivot.fillna(0)
        
        cohort_sizes = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.div(cohort_sizes, axis=0) * 100
        
        return retention_matrix, cohort_pivot
    
    def calculate_rolling_retention(self, days=[1, 7, 30]):
        print(f"\n=== ROLLING RETENTION (дни: {days}) ===")
        
        results = {}
        
        for day in days:
            print(f"\n--- Retention Day {day} ---")
            
            daily_retention = []
            dates = []
            all_dates = sorted(self.df['date'].unique())
            
            for current_date in all_dates:
                users_today = set(self.df[self.df['date'] == current_date]['telegram_id'])
                
                if len(users_today) == 0:
                    continue
                
                check_date = current_date + timedelta(days=day)
                users_check_date = set(self.df[self.df['date'] == check_date]['telegram_id'])
                retained_users = users_today.intersection(users_check_date)
                
                retention_rate = len(retained_users) / len(users_today) * 100 if len(users_today) > 0 else 0
                
                daily_retention.append(retention_rate)
                dates.append(current_date)
            
            retention_df = pd.DataFrame({
                'date': dates,
                f'retention_day_{day}': daily_retention
            })
            
            results[f'day_{day}'] = retention_df
            
            avg_retention = np.mean(daily_retention)
            median_retention = np.median(daily_retention)
            std_retention = np.std(daily_retention)
            
            print(f"Средний retention Day {day}: {avg_retention:.2f}%")
            print(f"Медианный retention Day {day}: {median_retention:.2f}%")
            print(f"Стандартное отклонение: {std_retention:.2f}%")
            print(f"Минимум: {min(daily_retention):.2f}%")
            print(f"Максимум: {max(daily_retention):.2f}%")
        
        return results
    
    def calculate_repeat_purchase_rate(self):
        print("\n=== АНАЛИЗ ПОВТОРНЫХ ПОКУПОК ===")
        
        user_purchases = self.df.groupby('telegram_id').size().reset_index()
        user_purchases.columns = ['telegram_id', 'purchase_count']
        
        user_purchases['category'] = pd.cut(
            user_purchases['purchase_count'], 
            bins=[0, 1, 2, 5, 10, float('inf')], 
            labels=['1 покупка', '2 покупки', '3-5 покупок', '6-10 покупок', '10+ покупок']
        )
        
        category_stats = user_purchases['category'].value_counts()
        total_users = len(user_purchases)
        
        print("Распределение пользователей по количеству покупок:")
        for category, count in category_stats.items():
            percentage = count / total_users * 100
            print(f"{category}: {count} ({percentage:.1f}%)")
        
        repeat_buyers = len(user_purchases[user_purchases['purchase_count'] > 1])
        repeat_rate = repeat_buyers / total_users * 100
        
        print(f"\nДоля пользователей с повторными покупками: {repeat_rate:.1f}%")
        
        return user_purchases, category_stats
    
    def analyze_subscription_patterns(self):
        print("\n=== АНАЛИЗ ПАТТЕРНОВ ПОДПИСОК ===")
        
        sub_stats = self.df['sub_name'].value_counts()
        print("Распределение по типам подписок:")
        for sub_name, count in sub_stats.items():
            percentage = count / len(self.df) * 100
            print(f"{sub_name}: {count} ({percentage:.1f}%)")
        
        self.df['day_of_week'] = self.df['datetime'].dt.day_name()
        dow_stats = self.df['day_of_week'].value_counts()
        print(f"\nРаспределение покупок по дням недели:")
        for day, count in dow_stats.items():
            percentage = count / len(self.df) * 100
            print(f"{day}: {count} ({percentage:.1f}%)")
        
        self.df['hour'] = self.df['datetime'].dt.hour
        hourly_stats = self.df['hour'].value_counts().sort_index()
        print(f"\nРаспределение покупок по часам (топ-10):")
        for hour, count in hourly_stats.head(10).items():
            percentage = count / len(self.df) * 100
            print(f"{hour:02d}:00: {count} ({percentage:.1f}%)")
    
    def create_visualizations(self, retention_data, cohort_matrix):
        print("\n=== СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ===")
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        ax1 = plt.subplot(2, 3, 1)
        for day_key, data in retention_data.items():
            day_num = day_key.split('_')[1]
            plt.plot(data['date'], data[f'retention_day_{day_num}'], 
                    label=f'Day {day_num}', marker='o', markersize=3)
        plt.title('Динамика Retention по дням', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Retention (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(cohort_matrix.iloc[-6:], annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Retention (%)'})
        plt.title('Когортный Retention (последние 6 месяцев)', fontsize=14, fontweight='bold')
        plt.xlabel('Дни с первой покупки')
        plt.ylabel('Когорта (месяц)')
        
        ax3 = plt.subplot(2, 3, 3)
        dow_counts = self.df['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = dow_counts.reindex(dow_order)
        plt.bar(range(len(dow_counts)), dow_counts.values)
        plt.title('Покупки по дням недели', fontsize=14, fontweight='bold')
        plt.xlabel('День недели')
        plt.ylabel('Количество покупок')
        plt.xticks(range(len(dow_counts)), [d[:3] for d in dow_counts.index], rotation=45)
        
        ax4 = plt.subplot(2, 3, 4)
        hourly_counts = self.df['hour'].value_counts().sort_index()
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o')
        plt.title('Покупки по часам дня', fontsize=14, fontweight='bold')
        plt.xlabel('Час')
        plt.ylabel('Количество покупок')
        plt.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        pivot_hour_dow = self.df.pivot_table(values='id', index='day_of_week', columns='hour', 
                                           aggfunc='count', fill_value=0)
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_hour_dow = pivot_hour_dow.reindex(dow_order)
        sns.heatmap(pivot_hour_dow, cmap='YlOrRd', cbar_kws={'label': 'Количество покупок'})
        plt.title('Активность по дням недели и часам', fontsize=14, fontweight='bold')
        plt.xlabel('Час')
        plt.ylabel('День недели')
        
        ax6 = plt.subplot(2, 3, 6)
        daily_purchases = self.df.groupby('date').size()
        plt.plot(daily_purchases.index, daily_purchases.values, marker='o', markersize=2)
        plt.title('Динамика ежедневных покупок', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Количество покупок')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('retention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_recommendations(self, retention_data, user_purchases):
        print("\n" + "="*60)
        print("РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ РЕТЕНШНА")
        print("="*60)
        
        avg_retention_1d = np.mean([data[f'retention_day_1'].mean() for data in retention_data.values() if f'retention_day_1' in data.columns])
        avg_retention_7d = np.mean([data[f'retention_day_7'].mean() for data in retention_data.values() if f'retention_day_7' in data.columns])
        avg_retention_30d = np.mean([data[f'retention_day_30'].mean() for data in retention_data.values() if f'retention_day_30' in data.columns])
        
        print(f"\nТЕКУЩИЕ ПОКАЗАТЕЛИ:")
        print(f"• Retention Day 1: {avg_retention_1d:.1f}%")
        print(f"• Retention Day 7: {avg_retention_7d:.1f}%")
        print(f"• Retention Day 30: {avg_retention_30d:.1f}%")
        
        repeat_rate = len(user_purchases[user_purchases['purchase_count'] > 1]) / len(user_purchases) * 100
        print(f"• Доля повторных покупателей: {repeat_rate:.1f}%")
        
        print(f"\nПРОЦЕССЫ ДЛЯ РАЗВИТИЯ РЕТЕНШНА:")
        print("="*60)
        
        recommendations = [
            {
                "приоритет": "ВЫСОКИЙ",
                "направление": "Onboarding и первое впечатление",
                "процессы": [
                    "Создать пошаговый гайд для новых пользователей",
                    "Настроить welcome-серию писем/уведомлений",
                    "Добавить интерактивный тур по функционалу",
                    "Создать систему достижений для первых действий"
                ],
                "метрики": "Retention Day 1"
            },
            {
                "приоритет": "ВЫСОКИЙ", 
                "направление": "Персональные уведомления",
                "процессы": [
                    "Настроить персонализированные push-уведомления",
                    "Создать систему email-маркетинга с сегментацией",
                    "Внедрить A/B тестирование сообщений",
                    "Добавить напоминания о неиспользованных функциях"
                ],
                "метрики": "Retention Day 7"
            },
            {
                "приоритет": "СРЕДНИЙ",
                "направление": "Программа лояльности",
                "процессы": [
                    "Создать систему бонусов и скидок",
                    "Внедрить реферальную программу",
                    "Добавить VIP-статусы для активных пользователей",
                    "Создать календарь специальных предложений"
                ],
                "метрики": "Retention Day 30"
            },
            {
                "приоритет": "СРЕДНИЙ",
                "направление": "Контент и вовлечение",
                "процессы": [
                    "Создать регулярный контент (блог, новости)",
                    "Добавить геймификацию (очки, уровни)",
                    "Внедрить социальные функции (отзывы, рейтинги)",
                    "Создать сообщество пользователей"
                ],
                "метрики": "Общее вовлечение"
            },
            {
                "приоритет": "НИЗКИЙ",
                "направление": "Техническая оптимизация",
                "процессы": [
                    "Улучшить производительность приложения",
                    "Добавить офлайн-режим",
                    "Оптимизировать процесс покупки",
                    "Внедрить аналитику поведения пользователей"
                ],
                "метрики": "UX метрики"
            }
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['направление']} (Приоритет: {rec['приоритет']})")
            print(f"   Целевая метрика: {rec['метрики']}")
            print("   Процессы:")
            for process in rec['процессы']:
                print(f"   • {process}")
        
        print(f"\nСИСТЕМА ИЗМЕРЕНИЯ ЭФФЕКТИВНОСТИ:")
        print("="*60)
        print("1. Еженедельные отчеты по retention метрикам")
        print("2. A/B тестирование всех изменений")
        print("3. Сегментация пользователей по поведению")
        print("4. Отслеживание конверсии на каждом этапе воронки")
        print("5. Регулярные опросы пользователей")
        
        return recommendations
    
    def run_full_analysis(self):
        print("ЗАПУСК ПОЛНОГО АНАЛИЗА РЕТЕНШНА")
        print("="*60)
        
        cohort_matrix, cohort_pivot = self.calculate_cohort_retention()
        retention_data = self.calculate_rolling_retention()
        user_purchases, category_stats = self.calculate_repeat_purchase_rate()
        self.analyze_subscription_patterns()
        self.create_visualizations(retention_data, cohort_matrix)
        recommendations = self.generate_recommendations(retention_data, user_purchases)
        
        return {
            'cohort_matrix': cohort_matrix,
            'retention_data': retention_data,
            'user_purchases': user_purchases,
            'recommendations': recommendations
        }

def main():
    analyzer = RetentionAnalyzer('user_logs_paid_241024_250909.json')
    results = analyzer.run_full_analysis()
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("Результаты сохранены в файл 'retention_analysis.png'")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
