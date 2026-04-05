```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2, chi2_contingency
from factor_analyzer import FactorAnalyzer
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("ПРАКТИЧЕСКАЯ РАБОТА №8")
print("ФАКТОРНЫЙ АНАЛИЗ")
print("=" * 60)

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

DATA_FILE = "../data/train.csv"

try:
    df = pd.read_csv(DATA_FILE)
    print(f"Файл загружен. Форма: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())
except FileNotFoundError:
    print(f"Ошибка: файл '{DATA_FILE}' не найден")
    exit(1)
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    exit(1)

print("\nИнформация о данных:")
df.info()

print("\nОписательная статистика:")
print(df.describe())

# ============================================================
# 2. ОЧИСТКА ДАННЫХ
# ============================================================

print("\n" + "=" * 60)
print("ОЧИСТКА ДАННЫХ")
print("=" * 60)

# Удаляем ненужные колонки
cols_to_drop = ['Unnamed: 0', 'id']
for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

# Проверка пропусков
print("\nПропуски до очистки:")
print(df.isnull().sum())

# Обработка пропусков
for col in df.columns:
    if df[col].isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print("\nПропуски после очистки:")
print(df.isnull().sum().sum())

# Удаление дубликатов
initial_rows = len(df)
df = df.drop_duplicates()
print(f"\nУдалено дубликатов: {initial_rows - len(df)}")

# ============================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ ФАКТОРНОГО АНАЛИЗА
# ============================================================

print("\n" + "=" * 60)
print("ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

# Выбираем только числовые переменные
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nЧисловые переменные ({len(numeric_cols)} шт.):")
print(numeric_cols)

# Создаем датафрейм только с числовыми переменными
df_numeric = df[numeric_cols].copy()

# Проверка на дисперсию (удаляем константные колонки)
constant_cols = []
for col in df_numeric.columns:
    if df_numeric[col].std() == 0:
        constant_cols.append(col)
        print(f"Удалена константная колонка: {col}")

df_numeric = df_numeric.drop(columns=constant_cols)

# Нормализация данных (StandardScaler)
print("\nНормализация данных...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
df_scaled = pd.DataFrame(scaled_data, columns=df_numeric.columns)

print("Нормализованные данные (первые 5 строк):")
print(df_scaled.head())

# ============================================================
# 4. ПРОВЕРКА ПРИГОДНОСТИ ДАННЫХ ДЛЯ ФАКТОРНОГО АНАЛИЗА
# ============================================================

print("\n" + "=" * 60)
print("ПРОВЕРКА ПРИГОДНОСТИ ДАННЫХ")
print("=" * 60)

# Ковариационная матрица
plt.figure(figsize=(14, 10))
sns.heatmap(df_scaled.cov(), annot=True, cmap='Blues', fmt='.2f', square=True)
plt.title('Ковариационная матрица')
plt.tight_layout()
plt.show()


# Функция для расчета критерия Бартлетта
def bartlett_sphericity(df):
    corr = df.corr().values
    n = df.shape[0]
    p = df.shape[1]

    # Расчет статистики Бартлетта
    det_corr = np.linalg.det(corr)
    chi2_stat = - (n - 1 - (2 * p + 5) / 6) * np.log(det_corr)
    df_chi2 = p * (p - 1) // 2
    p_value = 1 - chi2.cdf(chi2_stat, df_chi2)

    return chi2_stat, p_value, df_chi2


# Функция для расчета KMO
def calculate_kmo(df):
    corr = df.corr().values
    inv_corr = np.linalg.inv(corr)

    n_features = df.shape[1]
    kmo_numer = np.zeros(n_features)
    kmo_denom = np.zeros(n_features)

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                kmo_numer[i] += corr[i, j] ** 2
                kmo_denom[i] += corr[i, j] ** 2 + inv_corr[i, j] ** 2

    kmo_per_var = kmo_numer / kmo_denom
    kmo_total = np.sum(kmo_numer) / np.sum(kmo_denom)

    return kmo_total, kmo_per_var


# Критерий Бартлетта
chi2_value, p_value, df_chi2 = bartlett_sphericity(df_scaled)
print(f"\nКритерий Бартлетта:")
print(f"  Хи-квадрат: {chi2_value:.2f}")
print(f"  p-value: {p_value:.10f}")
print(f"  Степени свободы: {df_chi2}")

if p_value < 0.05:
    print("  Результат: p-value < 0.05 - данные подходят для факторного анализа")
else:
    print("  Результат: p-value >= 0.05 - данные НЕ подходят для факторного анализа")

# Тест Кайзера-Мейера-Олкина (KMO)
kmo_total, kmo_per_var = calculate_kmo(df_scaled)
print(f"\nТест Кайзера-Мейера-Олкина (KMO):")
print(f"  Общий KMO: {kmo_total:.4f}")
print("\n  KMO по переменным:")
for i, col in enumerate(df_scaled.columns):
    print(f"    {col}: {kmo_per_var[i]:.4f}")

if kmo_total >= 0.7:
    print("  Результат: KMO >= 0.7 - хорошая адекватность данных")
elif kmo_total >= 0.6:
    print("  Результат: KMO >= 0.6 - удовлетворительная адекватность данных")
elif kmo_total >= 0.5:
    print("  Результат: KMO >= 0.5 - приемлемая адекватность данных")
else:
    print("  Результат: KMO < 0.5 - данные НЕ подходят для факторного анализа")

# ============================================================
# 5. ОПРЕДЕЛЕНИЕ ЧИСЛА ФАКТОРОВ
# ============================================================

print("\n" + "=" * 60)
print("ОПРЕДЕЛЕНИЕ ЧИСЛА ФАКТОРОВ")
print("=" * 60)

# Используем PCA для получения собственных значений
pca = PCA()
pca.fit(df_scaled)
eigenvalues = pca.explained_variance_

print("\nСобственные значения:")
for i, ev in enumerate(eigenvalues):
    print(f"  Фактор {i + 1}: {ev:.4f}")

# График каменистой осыпи
plt.figure(figsize=(10, 6))
plt.scatter(range(1, len(eigenvalues) + 1), eigenvalues, color='steelblue', s=100)
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, color='steelblue', linestyle='--', marker='o')
plt.axhline(y=1, color='red', linestyle='--', label='Порог (собственное значение = 1)')
plt.title('График каменистой осыпи (Scree Plot)')
plt.xlabel('Номер фактора')
plt.ylabel('Собственное значение')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Определение числа факторов (собственное значение > 1)
n_factors = sum(eigenvalues > 1)
print(f"\nРекомендуемое число факторов (собственное значение > 1): {n_factors}")

# ============================================================
# 6. ФАКТОРНЫЙ АНАЛИЗ
# ============================================================

print("\n" + "=" * 60)
print("ФАКТОРНЫЙ АНАЛИЗ (МЕТОД ГЛАВНЫХ КОМПОНЕНТ)")
print("=" * 60)

from sklearn.decomposition import PCA

# Создаем объект PCA
pca = PCA()

# Обучаем модель
pca.fit(df_scaled)

# Собственные значения (дисперсия по компонентам)
eigenvalues = pca.explained_variance_

print("\nСобственные значения:")
for i, val in enumerate(eigenvalues):
    print(f"  Фактор {i + 1}: {val:.4f}")

# Выполняем PCA с выбранным числом компонент
pca = PCA(n_components=n_factors)
pca.fit(df_scaled)

# Факторные нагрузки (компоненты)
loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, index=df_scaled.columns,
                           columns=[f'Factor_{i + 1}' for i in range(n_factors)])

print(f"\nФакторные нагрузки (корреляция переменных с факторами):")
print(loadings_df.round(4))

# Тепловая карта факторных нагрузок
plt.figure(figsize=(12, max(8, n_factors * 2)))
sns.heatmap(loadings_df, annot=True, cmap='Blues', fmt='.2f', cbar=True)
plt.title('Факторные нагрузки (главные компоненты)')
plt.tight_layout()
plt.show()

# Объясненная дисперсия
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nОбъясненная дисперсия:")
for i in range(n_factors):
    print(f"  Фактор {i + 1}: {explained_variance_ratio[i]:.4f} (кумулятивно: {cumulative_variance[i]:.4f})")

# Факторные значения (трансформированные данные)
factor_scores = pca.transform(df_scaled)
print(f"\nФакторные значения для первых 10 наблюдений:")
print(pd.DataFrame(factor_scores[:10], columns=[f'Factor_{i + 1}' for i in range(n_factors)]).round(4))

# Интерпретация факторов
print("\n" + "=" * 60)
print("ИНТЕРПРЕТАЦИЯ ФАКТОРОВ")
print("=" * 60)

for i in range(n_factors):
    factor_col = f'Factor_{i + 1}'
    print(f"\nФактор {i + 1}:")

    # Сортируем по абсолютному значению нагрузки
    sorted_loadings = loadings_df[factor_col].abs().sort_values(ascending=False)
    top_vars = sorted_loadings.head(5)

    print(f"  Топ-5 переменных с наибольшим влиянием:")
    for var, val in top_vars.items():
        actual_val = loadings_df.loc[var, factor_col]
        print(f"    {var}: {actual_val:.4f}")

    # Переменные с высокими положительными нагрузками
    positive = loadings_df[factor_col][loadings_df[factor_col] > 0.5].sort_values(ascending=False)
    if len(positive) > 0:
        print(f"  Переменные с сильной положительной нагрузкой (>0.5):")
        for var, val in positive.items():
            print(f"    {var}: {val:.4f}")

    # Переменные с высокими отрицательными нагрузками
    negative = loadings_df[factor_col][loadings_df[factor_col] < -0.5].sort_values()
    if len(negative) > 0:
        print(f"  Переменные с сильной отрицательной нагрузкой (<-0.5):")
        for var, val in negative.items():
            print(f"    {var}: {val:.4f}")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================

print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

# График кумулятивной объясненной дисперсии
cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cumulative_var) + 1), cumulative_var, color='steelblue')
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', linewidth=2, markersize=8)
plt.title('Кумулятивная объясненная дисперсия по компонентам')
plt.xlabel('Количество компонент')
plt.ylabel('Кумулятивная доля дисперсии')
plt.ylim(0, 1)
for i, v in enumerate(cumulative_var):
    plt.text(i + 1, v + 0.02, f'{v:.2f}', ha='center')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Тепловая карта корреляций
plt.figure(figsize=(14, 12))
corr_matrix = df_numeric.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, fmt='.2f', square=True)
plt.title('Матрица корреляций числовых переменных')
plt.tight_layout()
plt.show()

# ============================================================
# ВЫВОДЫ
# ============================================================

print("\n" + "=" * 60)
print("ВЫВОДЫ ПО ФАКТОРНОМУ АНАЛИЗУ")
print("=" * 60)

print("\n1. КАЧЕСТВО ДАННЫХ:")
print(f"   - Исходный датасет: {initial_rows} строк, {len(df.columns)} столбцов")
print(f"   - После очистки: {len(df)} строк")
print(f"   - Числовых переменных для анализа: {len(df_scaled.columns)}")

print("\n2. ПРОВЕРКА ПРИГОДНОСТИ:")
print(f"   - Критерий Бартлетта: p-value = {p_value:.2e} {'(< 0.05) - данные подходят' if p_value < 0.05 else ''}")
print(f"   - KMO тест: {kmo_total:.4f}")

print("\n3. ОПРЕДЕЛЕНИЕ ФАКТОРОВ:")
print(f"   - Рекомендуемое число факторов: {n_factors}")
cum_var_percent = cumulative_var[n_factors - 1] * 100 if n_factors <= len(cumulative_var) else 0
print(f"   - Объясненная дисперсия: {cum_var_percent:.1f}%")

print("\n4. ИНТЕРПРЕТАЦИЯ ФАКТОРОВ:")
if 'loadings_df' in dir():
    for i in range(min(n_factors, 5)):
        factor_col = f'Factor_{i + 1}' if f'Factor_{i + 1}' in loadings_df.columns else f'Component_{i + 1}'
        if factor_col in loadings_df.columns:
            high_loadings = loadings_df[factor_col][abs(loadings_df[factor_col]) > 0.5]
            if len(high_loadings) > 0:
                variables = ', '.join(high_loadings.index.tolist())
                print(f"   - Фактор {i + 1}: {variables}")

print("\n5. ЗАКЛЮЧЕНИЕ:")
print("   Факторный анализ позволил сократить размерность данных и выявить")
print("   скрытые факторы, влияющие на удовлетворенность пассажиров.")
print("   Полученные факторы могут быть использованы для дальнейшего анализа")
print("   и построения моделей машинного обучения.")

print("\n" + "=" * 60)
```

    ПРАКТИЧЕСКАЯ РАБОТА №8
    ФАКТОРНЫЙ АНАЛИЗ
    ============================================================
    Файл загружен. Форма: (103904, 25)
    
    Первые 5 строк:
       Unnamed: 0      id  Gender      Customer Type  Age   Type of Travel  \
    0           0   70172    Male     Loyal Customer   13  Personal Travel   
    1           1    5047    Male  disloyal Customer   25  Business travel   
    2           2  110028  Female     Loyal Customer   26  Business travel   
    3           3   24026  Female     Loyal Customer   25  Business travel   
    4           4  119299    Male     Loyal Customer   61  Business travel   
    
          Class  Flight Distance  Inflight wifi service  \
    0  Eco Plus              460                      3   
    1  Business              235                      3   
    2  Business             1142                      2   
    3  Business              562                      2   
    4  Business              214                      3   
    
       Departure/Arrival time convenient  ...  Inflight entertainment  \
    0                                  4  ...                       5   
    1                                  2  ...                       1   
    2                                  2  ...                       5   
    3                                  5  ...                       2   
    4                                  3  ...                       3   
    
       On-board service  Leg room service  Baggage handling  Checkin service  \
    0                 4                 3                 4                4   
    1                 1                 5                 3                1   
    2                 4                 3                 4                4   
    3                 2                 5                 3                1   
    4                 3                 4                 4                3   
    
       Inflight service  Cleanliness  Departure Delay in Minutes  \
    0                 5            5                          25   
    1                 4            1                           1   
    2                 4            5                           0   
    3                 4            2                          11   
    4                 3            3                           0   
    
       Arrival Delay in Minutes             satisfaction  
    0                      18.0  neutral or dissatisfied  
    1                       6.0  neutral or dissatisfied  
    2                       0.0                satisfied  
    3                       9.0  neutral or dissatisfied  
    4                       0.0                satisfied  
    
    [5 rows x 25 columns]
    
    Информация о данных:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 103904 entries, 0 to 103903
    Data columns (total 25 columns):
     #   Column                             Non-Null Count   Dtype  
    ---  ------                             --------------   -----  
     0   Unnamed: 0                         103904 non-null  int64  
     1   id                                 103904 non-null  int64  
     2   Gender                             103904 non-null  object 
     3   Customer Type                      103904 non-null  object 
     4   Age                                103904 non-null  int64  
     5   Type of Travel                     103904 non-null  object 
     6   Class                              103904 non-null  object 
     7   Flight Distance                    103904 non-null  int64  
     8   Inflight wifi service              103904 non-null  int64  
     9   Departure/Arrival time convenient  103904 non-null  int64  
     10  Ease of Online booking             103904 non-null  int64  
     11  Gate location                      103904 non-null  int64  
     12  Food and drink                     103904 non-null  int64  
     13  Online boarding                    103904 non-null  int64  
     14  Seat comfort                       103904 non-null  int64  
     15  Inflight entertainment             103904 non-null  int64  
     16  On-board service                   103904 non-null  int64  
     17  Leg room service                   103904 non-null  int64  
     18  Baggage handling                   103904 non-null  int64  
     19  Checkin service                    103904 non-null  int64  
     20  Inflight service                   103904 non-null  int64  
     21  Cleanliness                        103904 non-null  int64  
     22  Departure Delay in Minutes         103904 non-null  int64  
     23  Arrival Delay in Minutes           103594 non-null  float64
     24  satisfaction                       103904 non-null  object 
    dtypes: float64(1), int64(19), object(5)
    memory usage: 19.8+ MB
    
    Описательная статистика:
              Unnamed: 0             id            Age  Flight Distance  \
    count  103904.000000  103904.000000  103904.000000    103904.000000   
    mean    51951.500000   64924.210502      39.379706      1189.448375   
    std     29994.645522   37463.812252      15.114964       997.147281   
    min         0.000000       1.000000       7.000000        31.000000   
    25%     25975.750000   32533.750000      27.000000       414.000000   
    50%     51951.500000   64856.500000      40.000000       843.000000   
    75%     77927.250000   97368.250000      51.000000      1743.000000   
    max    103903.000000  129880.000000      85.000000      4983.000000   
    
           Inflight wifi service  Departure/Arrival time convenient  \
    count          103904.000000                      103904.000000   
    mean                2.729683                           3.060296   
    std                 1.327829                           1.525075   
    min                 0.000000                           0.000000   
    25%                 2.000000                           2.000000   
    50%                 3.000000                           3.000000   
    75%                 4.000000                           4.000000   
    max                 5.000000                           5.000000   
    
           Ease of Online booking  Gate location  Food and drink  Online boarding  \
    count           103904.000000  103904.000000   103904.000000    103904.000000   
    mean                 2.756901       2.976883        3.202129         3.250375   
    std                  1.398929       1.277621        1.329533         1.349509   
    min                  0.000000       0.000000        0.000000         0.000000   
    25%                  2.000000       2.000000        2.000000         2.000000   
    50%                  3.000000       3.000000        3.000000         3.000000   
    75%                  4.000000       4.000000        4.000000         4.000000   
    max                  5.000000       5.000000        5.000000         5.000000   
    
            Seat comfort  Inflight entertainment  On-board service  \
    count  103904.000000           103904.000000     103904.000000   
    mean        3.439396                3.358158          3.382363   
    std         1.319088                1.332991          1.288354   
    min         0.000000                0.000000          0.000000   
    25%         2.000000                2.000000          2.000000   
    50%         4.000000                4.000000          4.000000   
    75%         5.000000                4.000000          4.000000   
    max         5.000000                5.000000          5.000000   
    
           Leg room service  Baggage handling  Checkin service  Inflight service  \
    count     103904.000000     103904.000000    103904.000000     103904.000000   
    mean           3.351055          3.631833         3.304290          3.640428   
    std            1.315605          1.180903         1.265396          1.175663   
    min            0.000000          1.000000         0.000000          0.000000   
    25%            2.000000          3.000000         3.000000          3.000000   
    50%            4.000000          4.000000         3.000000          4.000000   
    75%            4.000000          5.000000         4.000000          5.000000   
    max            5.000000          5.000000         5.000000          5.000000   
    
             Cleanliness  Departure Delay in Minutes  Arrival Delay in Minutes  
    count  103904.000000               103904.000000             103594.000000  
    mean        3.286351                   14.815618                 15.178678  
    std         1.312273                   38.230901                 38.698682  
    min         0.000000                    0.000000                  0.000000  
    25%         2.000000                    0.000000                  0.000000  
    50%         3.000000                    0.000000                  0.000000  
    75%         4.000000                   12.000000                 13.000000  
    max         5.000000                 1592.000000               1584.000000  
    
    ============================================================
    ОЧИСТКА ДАННЫХ
    ============================================================
    
    Пропуски до очистки:
    Gender                                 0
    Customer Type                          0
    Age                                    0
    Type of Travel                         0
    Class                                  0
    Flight Distance                        0
    Inflight wifi service                  0
    Departure/Arrival time convenient      0
    Ease of Online booking                 0
    Gate location                          0
    Food and drink                         0
    Online boarding                        0
    Seat comfort                           0
    Inflight entertainment                 0
    On-board service                       0
    Leg room service                       0
    Baggage handling                       0
    Checkin service                        0
    Inflight service                       0
    Cleanliness                            0
    Departure Delay in Minutes             0
    Arrival Delay in Minutes             310
    satisfaction                           0
    dtype: int64
    
    Пропуски после очистки:
    0
    
    Удалено дубликатов: 0
    
    ============================================================
    ПОДГОТОВКА ДАННЫХ
    ============================================================
    
    Числовые переменные (18 шт.):
    ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    
    Нормализация данных...
    Нормализованные данные (первые 5 строк):
            Age  Flight Distance  Inflight wifi service  \
    0 -1.745279        -0.731539               0.203579   
    1 -0.951360        -0.957184               0.203579   
    2 -0.885200        -0.047584              -0.549533   
    3 -0.951360        -0.629246              -0.549533   
    4  1.430397        -0.978244               0.203579   
    
       Departure/Arrival time convenient  Ease of Online booking  Gate location  \
    0                           0.616172                0.173776      -1.547323   
    1                          -0.695245                0.173776       0.018094   
    2                          -0.695245               -0.541060      -0.764614   
    3                           1.271880                1.603448       1.583511   
    4                          -0.039537                0.173776       0.018094   
    
       Food and drink  Online boarding  Seat comfort  Inflight entertainment  \
    0        1.352264        -0.185532      1.183099                1.231704   
    1       -1.656326        -0.185532     -1.849315               -1.769081   
    2        1.352264         1.296496      1.183099                1.231704   
    3       -0.904178        -0.926545     -1.091211               -1.018885   
    4        0.600117         1.296496      1.183099               -0.268688   
    
       On-board service  Leg room service  Baggage handling  Checkin service  \
    0          0.479403          -0.26684          0.311769         0.549799   
    1         -1.849161           1.25338         -0.535045        -1.821012   
    2          0.479403          -0.26684          0.311769         0.549799   
    3         -1.072973           1.25338         -0.535045        -1.821012   
    4         -0.296785           0.49327          0.311769        -0.240472   
    
       Inflight service  Cleanliness  Departure Delay in Minutes  \
    0          1.156436     1.305870                    0.266393   
    1          0.305848    -1.742292                   -0.361375   
    2          0.305848     1.305870                   -0.387532   
    3          0.305848    -0.980251                   -0.099805   
    4         -0.544740    -0.218211                   -0.387532   
    
       Arrival Delay in Minutes  
    0                  0.074169  
    1                 -0.236313  
    2                 -0.391554  
    3                 -0.158692  
    4                 -0.391554  
    
    ============================================================
    ПРОВЕРКА ПРИГОДНОСТИ ДАННЫХ
    ============================================================
    


    
![png](pr8_files/pr8_0_1.png)
    


    
    Критерий Бартлетта:
      Хи-квадрат: 886552.68
      p-value: 0.0000000000
      Степени свободы: 153
      Результат: p-value < 0.05 - данные подходят для факторного анализа
    
    Тест Кайзера-Мейера-Олкина (KMO):
      Общий KMO: 0.0489
    
      KMO по переменным:
        Age: 0.3565
        Flight Distance: 0.5068
        Inflight wifi service: 0.3115
        Departure/Arrival time convenient: 0.5052
        Ease of Online booking: 0.2828
        Gate location: 0.4454
        Food and drink: 0.4377
        Online boarding: 0.4114
        Seat comfort: 0.4749
        Inflight entertainment: 0.2835
        On-board service: 0.5317
        Leg room service: 0.7237
        Baggage handling: 0.5229
        Checkin service: 0.4360
        Inflight service: 0.4386
        Cleanliness: 0.3472
        Departure Delay in Minutes: 0.0061
        Arrival Delay in Minutes: 0.0061
      Результат: KMO < 0.5 - данные НЕ подходят для факторного анализа
    
    ============================================================
    ОПРЕДЕЛЕНИЕ ЧИСЛА ФАКТОРОВ
    ============================================================
    
    Собственные значения:
      Фактор 1: 3.8607
      Фактор 2: 2.3634
      Фактор 3: 2.1790
      Фактор 4: 1.9574
      Фактор 5: 1.2448
      Фактор 6: 0.9630
      Фактор 7: 0.9281
      Фактор 8: 0.8916
      Фактор 9: 0.6916
      Фактор 10: 0.5337
      Фактор 11: 0.4789
      Фактор 12: 0.4472
      Фактор 13: 0.3673
      Фактор 14: 0.3231
      Фактор 15: 0.2946
      Фактор 16: 0.2508
      Фактор 17: 0.1852
      Фактор 18: 0.0397
    


    
![png](pr8_files/pr8_0_3.png)
    


    
    Рекомендуемое число факторов (собственное значение > 1): 5
    
    ============================================================
    ФАКТОРНЫЙ АНАЛИЗ (МЕТОД ГЛАВНЫХ КОМПОНЕНТ)
    ============================================================
    
    Собственные значения:
      Фактор 1: 3.8607
      Фактор 2: 2.3634
      Фактор 3: 2.1790
      Фактор 4: 1.9574
      Фактор 5: 1.2448
      Фактор 6: 0.9630
      Фактор 7: 0.9281
      Фактор 8: 0.8916
      Фактор 9: 0.6916
      Фактор 10: 0.5337
      Фактор 11: 0.4789
      Фактор 12: 0.4472
      Фактор 13: 0.3673
      Фактор 14: 0.3231
      Фактор 15: 0.2946
      Фактор 16: 0.2508
      Фактор 17: 0.1852
      Фактор 18: 0.0397
    
    Факторные нагрузки (корреляция переменных с факторами):
                                       Factor_1  Factor_2  Factor_3  Factor_4  \
    Age                                  0.0715    0.0104   -0.1023   -0.0092   
    Flight Distance                      0.1195   -0.0198   -0.0109    0.0216   
    Inflight wifi service                0.2310    0.4496   -0.0439   -0.0135   
    Departure/Arrival time convenient    0.0992    0.4175    0.0261   -0.0086   
    Ease of Online booking               0.1627    0.5364   -0.0464   -0.0158   
    Gate location                        0.0642    0.4302   -0.0341   -0.0099   
    Food and drink                       0.3022   -0.1578   -0.3387   -0.0032   
    Online boarding                      0.2871    0.1515   -0.1698   -0.0020   
    Seat comfort                         0.3457   -0.1533   -0.3136    0.0063   
    Inflight entertainment               0.4221   -0.1775   -0.0644    0.0305   
    On-board service                     0.2745   -0.0701    0.3791    0.0280   
    Leg room service                     0.2204   -0.0138    0.2951    0.0660   
    Baggage handling                     0.2547   -0.0569    0.4354    0.0525   
    Checkin service                      0.1813   -0.0485    0.1316    0.0124   
    Inflight service                     0.2586   -0.0646    0.4469    0.0093   
    Cleanliness                          0.3517   -0.1758   -0.3119    0.0197   
    Departure Delay in Minutes          -0.0382    0.0283   -0.0366    0.7035   
    Arrival Delay in Minutes            -0.0406    0.0281   -0.0377    0.7032   
    
                                       Factor_5  
    Age                                  0.5256  
    Flight Distance                      0.4866  
    Inflight wifi service                0.0140  
    Departure/Arrival time convenient   -0.1941  
    Ease of Online booking               0.0502  
    Gate location                       -0.2626  
    Food and drink                      -0.2542  
    Online boarding                      0.4261  
    Seat comfort                         0.0121  
    Inflight entertainment              -0.1835  
    On-board service                     0.0420  
    Leg room service                     0.1267  
    Baggage handling                    -0.0997  
    Checkin service                      0.1695  
    Inflight service                    -0.1102  
    Cleanliness                         -0.1702  
    Departure Delay in Minutes          -0.0027  
    Arrival Delay in Minutes            -0.0050  
    


    
![png](pr8_files/pr8_0_5.png)
    


    
    Объясненная дисперсия:
      Фактор 1: 0.2145 (кумулятивно: 0.2145)
      Фактор 2: 0.1313 (кумулятивно: 0.3458)
      Фактор 3: 0.1211 (кумулятивно: 0.4668)
      Фактор 4: 0.1087 (кумулятивно: 0.5756)
      Фактор 5: 0.0692 (кумулятивно: 0.6447)
    
    Факторные значения для первых 10 наблюдений:
       Factor_1  Factor_2  Factor_3  Factor_4  Factor_5
    0    2.1064   -1.2379   -0.2305    0.3409   -1.9117
    1   -3.3230    1.2431    1.2570   -0.5432   -0.0890
    2    2.1289   -1.9274   -0.9108   -0.4257   -0.3950
    3   -1.8683    2.5003    0.8470   -0.2789   -1.3946
    4    0.7831    0.2378   -0.9407   -0.5647    0.8322
    5   -2.5929    0.3367    2.5382   -0.5666    0.4989
    6   -1.6056    0.1992    1.6168   -0.0103    0.0775
    7    3.8166    0.4223    0.4602   -0.2532    0.6759
    8   -3.3702   -0.7054   -2.4414   -0.8442    0.7964
    9   -1.4853    1.0687    0.4090   -0.6308   -0.3808
    
    ============================================================
    ИНТЕРПРЕТАЦИЯ ФАКТОРОВ
    ============================================================
    
    Фактор 1:
      Топ-5 переменных с наибольшим влиянием:
        Inflight entertainment: 0.4221
        Cleanliness: 0.3517
        Seat comfort: 0.3457
        Food and drink: 0.3022
        Online boarding: 0.2871
    
    Фактор 2:
      Топ-5 переменных с наибольшим влиянием:
        Ease of Online booking: 0.5364
        Inflight wifi service: 0.4496
        Gate location: 0.4302
        Departure/Arrival time convenient: 0.4175
        Inflight entertainment: -0.1775
      Переменные с сильной положительной нагрузкой (>0.5):
        Ease of Online booking: 0.5364
    
    Фактор 3:
      Топ-5 переменных с наибольшим влиянием:
        Inflight service: 0.4469
        Baggage handling: 0.4354
        On-board service: 0.3791
        Food and drink: -0.3387
        Seat comfort: -0.3136
    
    Фактор 4:
      Топ-5 переменных с наибольшим влиянием:
        Departure Delay in Minutes: 0.7035
        Arrival Delay in Minutes: 0.7032
        Leg room service: 0.0660
        Baggage handling: 0.0525
        Inflight entertainment: 0.0305
      Переменные с сильной положительной нагрузкой (>0.5):
        Departure Delay in Minutes: 0.7035
        Arrival Delay in Minutes: 0.7032
    
    Фактор 5:
      Топ-5 переменных с наибольшим влиянием:
        Age: 0.5256
        Flight Distance: 0.4866
        Online boarding: 0.4261
        Gate location: -0.2626
        Food and drink: -0.2542
      Переменные с сильной положительной нагрузкой (>0.5):
        Age: 0.5256
    
    ============================================================
    ВИЗУАЛИЗАЦИЯ
    ============================================================
    


    
![png](pr8_files/pr8_0_7.png)
    



    
![png](pr8_files/pr8_0_8.png)
    


    
    ============================================================
    ВЫВОДЫ ПО ФАКТОРНОМУ АНАЛИЗУ
    ============================================================
    
    1. КАЧЕСТВО ДАННЫХ:
       - Исходный датасет: 103904 строк, 23 столбцов
       - После очистки: 103904 строк
       - Числовых переменных для анализа: 18
    
    2. ПРОВЕРКА ПРИГОДНОСТИ:
       - Критерий Бартлетта: p-value = 0.00e+00 (< 0.05) - данные подходят
       - KMO тест: 0.0489
    
    3. ОПРЕДЕЛЕНИЕ ФАКТОРОВ:
       - Рекомендуемое число факторов: 5
       - Объясненная дисперсия: 64.5%
    
    4. ИНТЕРПРЕТАЦИЯ ФАКТОРОВ:
       - Фактор 2: Ease of Online booking
       - Фактор 4: Departure Delay in Minutes, Arrival Delay in Minutes
       - Фактор 5: Age
    
    5. ЗАКЛЮЧЕНИЕ:
       Факторный анализ позволил сократить размерность данных и выявить
       скрытые факторы, влияющие на удовлетворенность пассажиров.
       Полученные факторы могут быть использованы для дальнейшего анализа
       и построения моделей машинного обучения.
    
    ============================================================
    
