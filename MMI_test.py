import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# FIMMWAVE
datasetfi1_2 = [0.458, 0.454, 0.46, 0.456, 0.456, 0.453, 0.454, 0.455]
datasetfi2_2 = [0.439, 0.454, 0.483, 0.469, 0.44, 0.45, 0.48, 0.472]

dataset1 =  datasetfi1_2 + datasetfi2_2
# FDTD 2d
datasetfd1_2 = [ 0.497, 0.48, 0.44, 0.495, 0.495, 0.481, 0.44, 0.494]
datasetfd2_2 = [0.448, 0.497, 0.453, 0.48, 0.45, 0.485, 0.425, 0.476]
dataset2 = datasetfd1_2 + datasetfd2_2

# Obliczenie procentowej różnicy między datasetami
mean1 = np.mean(dataset1)
mean2 = np.mean(dataset2)
percentage_difference = abs(mean1 - mean2) / ((mean1 + mean2) / 2) * 100

# Obliczenie procentowego odchylenia maksymalnego
std_dev_max1 = (max(dataset1) - mean1) / mean1 * 100
std_dev_max2 = (max(dataset2) - mean2) / mean2 * 100

# Obliczenie MAE (Mean Absolute Error)
mae = np.mean(np.abs(np.array(dataset1) - np.array(dataset2)))

# Obliczenie MRE (Maximum Relative Error)
max_relative_error = np.max(np.abs(np.array(dataset1) - np.array(dataset2)) / np.array(dataset1)) * 100
# Obliczenie MPE (Mean Percentage Error)
mpe = np.mean(((np.array(dataset1) - np.array(dataset2)) / np.array(dataset1)) * 100)

distance = np.linalg.norm(np.array(dataset1) - np.array(dataset2))
# Test t-Studenta dla dwóch niezależnych próbek
t_stat, p_value = stats.ttest_ind(dataset1, dataset2)
# Test Kolmogorova-Smirnova
ks_statistic, ks_p_value = stats.ks_2samp(dataset1, dataset2)
n_samples = 1000
mean1, std_dev1 = norm.fit(dataset1)
mean2, std_dev2 = norm.fit(dataset2)
generated_samples1 = np.random.normal(mean1, std_dev1, n_samples)
generated_samples2 = np.random.normal(mean2, std_dev2, n_samples)
hist1, bin_edges1, _ = plt.hist(generated_samples1, bins=50, alpha=0.5, label='FIMMWAVE', color='blue')
hist2, bin_edges2, _ = plt.hist(generated_samples2, bins=50, alpha=0.5, label='FDTD', color='orange')
max_y = max(max(hist1), max(hist2))
min_x = min(min(generated_samples1), min(generated_samples2))
max_x = max(max(generated_samples1), max(generated_samples2))
plt.axvline(mean1, color='blue', linestyle='dashed', linewidth=2, label=f'µ : {mean1:.3f}')
plt.axvline(mean2, color='orange', linestyle='dashed', linewidth=2, label=f'µ : {mean2:.3f}')
plt.fill_betweenx([0, max_y], mean1 - std_dev1, mean1 + std_dev1, alpha=0.7, facecolor='none', edgecolor='blue', label=f'Std : {std_dev1:.3f}', hatch='//')
plt.fill_betweenx([0, max_y], mean2 - std_dev2, mean2 + std_dev2, alpha=0.7, facecolor='none', edgecolor='orange', label=f'Std : {std_dev2:.3f}', hatch='\\')
plt.xlabel('Pfwd')
plt.ylabel('N')
plt.legend()
plt.grid()
plt.ylim(0, max_y)
plt.xlim(min_x, max_x)
plt.show()

# Analiza regresji
slope, intercept, r_value, p_value, std_err = stats.linregress(dataset1, dataset2)
plt.scatter(generated_samples1, generated_samples2, label='Pfwd', color='blue')
# Linia regresji
x = np.array(generated_samples1)
y = slope * x + intercept
plt.plot(x, y, color='red', label='reg line')

plt.xlabel('Pfwd FIMMWAVE ')
plt.ylabel('Pfwd FDTD ')
plt.legend()
plt.grid()

# Wyświetlenie wyników

print(f"Wartość statystyki t: {t_stat}")
print(f"Wartość p: {p_value}")

alpha = 0.05  # Poziom istotności w procentach %
if p_value < alpha and ks_p_value < alpha:
    print("Odrzucamy hipotezę zerową - Istnieje różnica między rozkładami.")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej - Brak różnicy między rozkładami.")

print(f"Współczynnik kierunkowy (slope): {slope:.4f}")
print(f'Wyraz wolny (intercept): {intercept:.4f}')
print(f'Współczynnik korelacji (R-squared): {r_value**2:.4f}')
print(f'Wartość p (test istotności współczynnika kierunkowego): {p_value:.4f}')
print(f'Błąd standardowy (std_err): {std_err:.4f}')
print(f"Procentowa różnica między rozkładami: {percentage_difference:.2f}%")
print(f"Procentowe odchylenie maksymalne (FIMMWAVE): {std_dev_max1:.2f}%")
print(f"Procentowe odchylenie maksymalne (FDTD): {std_dev_max2:.2f}%")
print(f'MAE: {mae:.4f}')
print(f'MPE: {abs(mpe):.4f}%')
print(f'MPE_double: {abs(mpe)*2:.4f}%')
print(f'MRE: {max_relative_error:.4f}%')
print(f'MRE_half: {max_relative_error/2:.4f}%')

plt.show()

# Konfiguracja wykresu
fig, ax = plt.subplots(figsize=(8, 6))
boxprops = dict(linestyle='-', linewidth=2, color='blue')
whiskerprops = dict(linestyle='--', linewidth=1, color='gray')
medianprops = dict(linestyle='-', linewidth=2, color='red')
capprops = dict(linestyle='-', linewidth=1, color='black')

# Tworzenie wykresu pudełkowego
boxes = ax.boxplot([dataset1, dataset2], labels=['FIMMWAVE', 'FDTD'], boxprops=boxprops, whiskerprops=whiskerprops,
                   medianprops=medianprops, capprops=capprops, patch_artist=True)

# Dodanie kolorów do pudełek
colors = ['lightblue', 'lightgreen']
for box, color in zip(boxes['boxes'], colors):
    box.set_facecolor(color)

# Etykiety na osiach
ax.set_ylabel('Pfwd')

plt.grid()
plt.show()