import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# FIMMWAVE
dataset1 = [0.439, 0.454, 0.483, 0.469, 0.44, 0.45, 0.48, 0.472, 0.458, 0.454, 0.46, 0.456, 0.456, 0.453, 0.454, 0.455]
# FDTD 2d
dataset2 = [0.448, 0.497, 0.453, 0.48, 0.45, 0.485, 0.425, 0.476, 0.497, 0.48, 0.44, 0.495, 0.495, 0.481, 0.44, 0.494]
distance = np.linalg.norm(np.array(dataset1) - np.array(dataset2))
# Test t-Studenta dla dwóch niezależnych próbek
t_stat, p_value = stats.ttest_ind(dataset1, dataset2)
# Test Kolmogorova-Smirnova
ks_statistic, ks_p_value = stats.ks_2samp(dataset1, dataset2)
n_samples = 1000
mean1, std_dev1 = norm.fit(dataset1)
mean2, std_dev2 = norm.fit(dataset2)
generated_samples1 = np.random.normal(mean1, std_dev1, n_samples)
generated_samples2 = np.random.norOdchylmal(mean2, std_dev2, n_samples)
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

print(f"Wartość statystyki t: {t_stat}")
print(f"Wartość p: {p_value}")

alpha = 0.05  # Poziom istotności
if p_value < alpha and ks_p_value < alpha:
    print("Odrzucamy hipotezę zerową - Istnieje różnica między rozkładami.")
else:
    print("Nie ma podstaw do odrzucenia hipotezy zerowej - Brak różnicy między rozkładami.")
