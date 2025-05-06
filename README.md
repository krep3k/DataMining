Laporan proyek: Penggabungan dataset sebagai dasar untuk pelatihan model klasifikasi menggunakan algoritma Decision Tree.

import library dan membaca dataset
```python
import pandas as pd

data_1 = pd.read_csv("Recon Ping Sweep.csv")
data_2 = pd.read_csv("Recon Vulnerability Scan.csv")
gabungan = pd.concat([data_1, data_2], ignore_index=True)
```
<p>data_1 & data_2: Membaca dua file CSV, gabungan: Menggabungkan dua data tersebut menjadi satu DataFrame</p>

Menentukan Fitur (X) dan Target (Y)
```python
x = gabungan.iloc[:, 7:25]  # Mengambil kolom ke-7 sampai ke-24 sebagai fitur
y = gabungan.iloc[:, 50]    # Mengambil kolom ke-50 sebagai label (target)
```
x adalah data fitur yang akan digunakan untuk pelatihan dan y adalah label yang akan diprediksi

Membagi Data
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=50)
```
Memisahkan data menjadi data latih (90%) dan data uji (10%).

Membuat dan Melatih Model Decision Tree
```python
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

alya = DecisionTreeClassifier(criterion='entropy', splitter='random')
alya.fit(x_train, y_train)
```
Membuat dan melatih model Decision Tree untuk klasifikasi

Prediksi dan Akurasi
```python
y_pred = alya.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```
Melakukan prediksi pada data uji dan menghitung tingkat akurasi

Visualisasi Decision Tree
```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 7))
tree.plot_tree(alya, feature_names=x.columns.values, class_names=np.array(['Benign Traffic', 'DDos ICMP Flood', 'DDoS UDP Flood']), filled=True)
plt.show()
```
Menampilkan visualisasi Decision Tree yang menunjukkan bagaimana model memutuskan klasifikasi

Confusion Matrix
```python
import seaborn as lol
from sklearn import metrics

label = np.array(['Recon Vulnerability Scan', 'Recon Ping Sweep'])
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
lol.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
Membuat confusion matrix dalam bentuk heatmap untuk mengevaluasi performa klasifikasi model
