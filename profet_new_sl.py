import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import streamlit as st
import io

# Функция для прогнозирования выручки одного отделения
def forecast_department_revenue(data, department, forecast_days):
    """
    Прогнозирует суммарную выручку для отделения за указанное количество дней.
    Args:
        data: DataFrame с данными (ds, department, y).
        department: Название отделения.
        forecast_days: Количество дней для прогноза.
    Returns:
        forecast: DataFrame с прогнозом.
        model: Обученная модель Prophet.
        growth: Процент роста выручки.
        total_forecast_revenue: Суммарная выручка за указанный период.
        mae: Средняя абсолютная ошибка (или None).
    """
    # Фильтрация данных по отделению
    dept_data = data[data['department'] == department][['ds', 'y']].copy()
    dept_data['ds'] = pd.to_datetime(dept_data['ds'])
    
    # Проверка на достаточность данных (минимум 30 дней)
    if len(dept_data) < 30:
        st.warning(f"Предупреждение: недостаточно данных для отделения {department} ({len(dept_data)} записей).")
        print(f"Предупреждение: недостаточно данных для отделения {department} ({len(dept_data)} записей).")
        return None, None, None, None, None
    
    # Проверка на валидность данных
    if dept_data['y'].isna().any() or not np.isfinite(dept_data['y']).all():
        st.error(f"Ошибка: некорректные данные (NaN или бесконечные значения) для отделения {department}.")
        print(f"Ошибка: некорректные данные (NaN или бесконечные значения) для отделения {department}.")
        return None, None, None, None, None
    
    # Последняя дата в данных
    last_date = dept_data['ds'].max()
    st.write(f"Последняя дата для {department}: {last_date}")
    print(f"Последняя дата для {department}: {last_date}")
    target_end_date = last_date + pd.Timedelta(days=forecast_days)
    
    # Проверка, достаточно ли данных для прогноза до целевой даты
    min_date = last_date - pd.Timedelta(days=365)
    if last_date < min_date:
        st.warning(f"Предупреждение: данные для {department} заканчиваются слишком рано ({last_date}).")
        print(f"Предупреждение: данные для {department} заканчиваются слишком рано ({last_date}).")
        return None, None, None, None, None
    
    # Разделение на обучающую и тестовую выборки
    train_data = dept_data[:-7] if len(dept_data) >= 37 else dept_data
    test_data = dept_data[-7:] if len(dept_data) >= 37 else pd.DataFrame()
    
    # Инициализация модели Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    model.add_country_holidays(country_name='RU')
    try:
        model.fit(train_data)
    except Exception as e:
        st.error(f"Ошибка при обучении модели для {department}: {e}")
        print(f"Ошибка при обучении модели для {department}: {e}")
        return None, None, None, None, None
    
    # Оценка точности, если есть тестовые данные
    mae = None
    if not test_data.empty:
        test_forecast = model.predict(test_data[['ds']])
        mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])
    
    # Прогноз на forecast_days + 10 дней (запас для надежности)
    future = model.make_future_dataframe(periods=forecast_days + 10, freq='D')
    forecast = model.predict(future)
    
    # Фильтрация прогноза за период от last_date + 1 до last_date + forecast_days
    forecast_period = forecast[
        (forecast['ds'] > last_date) & 
        (forecast['ds'] <= target_end_date)
    ]
    
    if forecast_period.empty:
        st.error(f"Ошибка: Прогноз для {department} не содержит данных за период {last_date + pd.Timedelta(days=1)} – {target_end_date}. Последняя дата: {forecast['ds'].max()}")
        print(f"Ошибка: Прогноз для {department} не содержит данных за период {last_date + pd.Timedelta(days=1)} – {target_end_date}. Последняя дата: {forecast['ds'].max()}")
        return None, None, None, None, None
    
    # Суммарная прогнозная выручка за период
    total_forecast_revenue = forecast_period['yhat'].sum()
    
    # Средняя выручка за последние 30 дней
    last_30_days = train_data[train_data['ds'] >= (train_data['ds'].max() - pd.Timedelta(days=30))]['y'].mean()
    
    # Вычисление роста (в процентах)
    growth = ((total_forecast_revenue - last_30_days * forecast_days) / (last_30_days * forecast_days)) * 100 if last_30_days != 0 else 0
    
    return forecast, model, growth, total_forecast_revenue, mae

# Основная функция
def main():
    """
    Основная функция: позволяет загрузить Excel-файл, задать количество дней для прогноза,
    прогнозирует выручку, отображает результаты в Streamlit и сохраняет в Excel.
    """
    st.title("Прогноз выручки отделений медицинского учреждения")
    st.write("Загрузите Excel-файл с данными (столбцы: ds, department, y) и укажите количество дней для прогноза.")

    # Загрузка файла через Streamlit
    uploaded_file = st.file_uploader("Выберите Excel-файл (.xlsx или .xls)", type=["xlsx", "xls"])
    
    # Ввод количества дней для прогноза
    forecast_days = st.number_input(
        "Количество дней для прогноза",
        min_value=1,
        max_value=365,
        value=31,
        step=1
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
            print(f"Ошибка при чтении файла: {e}")
            return
        
        # Преобразование дат
        try:
            data['ds'] = pd.to_datetime(data['ds'], format="%d.%m.%Y")
        except Exception as e:
            st.error(f"Ошибка при преобразовании дат: {e}")
            print(f"Ошибка при преобразовании дат: {e}")
            return
        
        # Проверка структуры данных
        if not all(col in data.columns for col in ['ds', 'department', 'y']):
            st.error("Ошибка: Данные должны содержать столбцы 'ds', 'department', 'y'.")
            print("Ошибка: Данные должны содержать столбцы 'ds', 'department', 'y'.")
            return
        
        # Диагностика данных
        st.subheader("Диагностика данных")
        st.write("Количество записей по отделениям:")
        st.write(data.groupby('department').size())
        st.write(f"Последняя дата в данных: {data['ds'].max()}")
        st.write("Последние даты по отделениям:")
        st.write(data.groupby('department')['ds'].max())
        print("Количество записей по отделениям:")
        print(data.groupby('department').size())
        print(f"Последняя дата в данных: {data['ds'].max()}")
        print("Последние даты по отделениям:")
        print(data.groupby('department')['ds'].max())
        
        # Список отделений
        departments = data['department'].unique()
        st.write(f"Найдено {len(departments)} отделений: {list(departments)}")
        print(f"Найдено {len(departments)} отделений: {list(departments)}")
        
        # Словарь для хранения результатов
        results = {}
        
        # Прогнозирование для каждого отделения
        st.subheader("Обработка отделений")
        for dept in departments:
            st.write(f"Обработка отделения: {dept}")
            print(f"\nОбработка отделения: {dept}")
            forecast, model, growth, total_forecast_revenue, mae = forecast_department_revenue(data, dept, forecast_days)
            if forecast is not None:
                results[dept] = {
                    'forecast': forecast,
                    'model': model,
                    'growth': growth,
                    'total_forecast_revenue': total_forecast_revenue,
                    'mae': mae
                }
            else:
                st.warning(f"Пропущено отделение {dept} из-за ошибок или недостатка данных.")
                print(f"Пропущено отделение {dept} из-за ошибок или недостатка данных.")
        
        # Проверка, есть ли результаты
        if not results:
            st.error("Ошибка: Не удалось создать прогнозы ни для одного отделения.")
            print("Ошибка: Не удалось создать прогнозы ни для одного отделения.")
            return
        
        # Сбор результатов для Excel и таблицы
        output_data = []
        for dept, result in results.items():
            last_date = data[data['department'] == dept]['ds'].max()
            output_data.append({
                'Отделение': dept,
                'Начало периода': last_date + pd.Timedelta(days=1),
                'Конец периода': last_date + pd.Timedelta(days=forecast_days),
                'Суммарная выручка (руб.)': result['total_forecast_revenue'],
                'Потенциальный рост (%)': result['growth'],
                'MAE (руб.)': result['mae'] if result['mae'] is not None else 'Недоступно'
            })
        
        # Создание DataFrame и сохранение в Excel
        output_df = pd.DataFrame(output_data)
        try:
            output_df.to_excel('forecast_results.xlsx', index=False, engine='openpyxl')
            st.success("Результаты успешно сохранены в 'forecast_results.xlsx'")
            print("Результаты успешно сохранены в 'forecast_results.xlsx'")
        except ImportError:
            st.error("Ошибка: Библиотека 'openpyxl' не установлена. Установите ее с помощью 'pip install openpyxl'.")
            print("Ошибка: Библиотека 'openpyxl' не установлена. Установите ее с помощью 'pip install openpyxl'.")
            return
        except Exception as e:
            st.error(f"Ошибка при сохранении в Excel: {e}")
            print(f"Ошибка при сохранении в Excel: {e}")
            return
        
        # Отображение таблицы результатов
        st.subheader("Результаты прогноза")
        st.write(f"Таблица с суммарной выручкой, ростом и MAE за {forecast_days} дней:")
        st.dataframe(output_df)
        
        # Кнопка для скачивания Excel
        buffer = io.BytesIO()
        output_df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        st.download_button(
            label="Скачать результаты (Excel)",
            data=buffer,
            file_name="forecast_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Визуализация точек роста
        growth_data = pd.DataFrame([
            {'department': dept, 'growth': result['growth']}
            for dept, result in results.items()
        ])
        
        if growth_data.empty:
            st.error("Ошибка: Нет данных для визуализации точек роста.")
            print("Ошибка: Нет данных для визуализации точек роста.")
            return
        
        growth_data = growth_data.sort_values(by='growth', ascending=False)
        
        st.subheader("Точки роста")
        st.write(f"Столбчатая диаграмма потенциального роста выручки за {forecast_days} дней (%):")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(growth_data['department'], growth_data['growth'], color='skyblue')
        ax.set_title(f'Потенциальный рост выручки по отделениям (%) за {forecast_days} дней')
        ax.set_xlabel('Отделение')
        ax.set_ylabel('Рост (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Выбор отделения для отображения прогноза
        st.subheader("Прогноз для выбранного отделения")
        selected_dept = st.selectbox("Выберите отделение:", list(results.keys()))
        if selected_dept:
            result = results[selected_dept]
            st.write(f"Прогноз выручки для {selected_dept} за {forecast_days + 10} дней:")
            fig = result['model'].plot(result['forecast'])
            plt.title(f'Прогноз выручки для {selected_dept} за {forecast_days + 10} дней')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()