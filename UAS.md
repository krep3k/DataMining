**Impor Library yang dibutuhkan**

Tujuan dari langkah ini adalah memuat semua **alat** atau modul yang kita perlukan untuk analisis data, machine learning, dan visualisasi.

```python
import pandas as pd
```

**Fungsinya yaitu,** Mengimpor library **pandas** dan memberinya nama alias **pd**. Pandas adalah library fundamental untuk manipulasi dan analisis data di Python. Kita menggunakannya untuk membuat dan mengelola data dalam format tabel yang disebut DataFrame, yang mirip seperti spreadsheet. 

```python
from sklearn.datasets import load_wine
```

**Fungsinya yaitu,** Mengimpor fungsi **load_wine** dari library **scikit-learn** (sklearn). **scikit-learn** adalah library utama untuk machine learning di Python. Fungsi **load_wine** ini secara spesifik digunakan untuk memuat dataset "Wine" yang sudah tersedia di dalamnya, jadi kita tidak perlu mengunduh file CSV terpisah.

```python
from sklearn.cluster import KMeans
```

**Fungsinya yaitu,** Mengimpor kelas **PCA** (Principal Component Analysis) dari modul "decomposition" di **scikit-learn**. PCA adalah teknik untuk mengurangi dimensi data. Kita akan menggunakannya untuk menyederhanakan 13 fitur wine menjadi 2 fitur utama agar bisa digambar dalam grafik 2D.


**Muat dan Persiapkan Dataset Wine**

Tujuan langkah ini adalah mengambil data mentah, memuatnya ke dalam program, dan menyusunnya ke dalam format yang mudah dibaca dan diolah.

```python
wine = load_wine()
```

**Fungsinya yaitu,** Menjalankan fungsi **load_wine()** dan menyimpan hasilnya ke dalam variabel **wine**. Variabel **wine** ini sekarang berisi sebuah objek yang memiliki beberapa bagian, seperti **data** (fitur-fitur numerik), **target** (label kelas asli), dan **feature_names**(nama dari setiap fitur).

```python
X = wine.data
```

**Fungsinya yaitu,** Mengambil data fitur (13 kolom berisi angka seperti kadar alkohol, malic acid, dll.) dari objek **wine** dan menyimpannya ke variabel **X**. Dalam machine learning, **X** adalah konvensi umum untuk menampung data fitur (data input

```python
df = pd.DataFrame(data=X, columns=wine.feature_names)
```

**Fungsinya yaitu,** Membuat sebuah DataFrame pandas dari data **X**.
- **data=X:** Memberi tahu pandas untuk menggunakan data fitur kita.
- **columns=wine.feature_names:** Memberi nama pada setiap kolom sesuai dengan nama fitur yang ada di dataset. Ini membuat data jauh lebih mudah dibaca daripada sekadar angka.

```python
print(df.head())
print(f"\nBentuk data: {X.shape} ...")
```

**Fungsinya yaitu,** **df.head()** menampilkan 5 baris pertama dari DataFrame untuk memastikan data termuat dengan benar. **X.shape** adalah atribut yang memberikan dimensi data dalam format **(jumlah_baris, jumlah_kolom)**. Ini digunakan untuk verifikasi ukuran dataset.


**Menentukan Jumlah Cluster (k)**

Langkah ini sangat penting dalam K-Means. Kita harus memberi tahu algoritma berapa banyak kelompok yang ingin kita temukan.

```python
k = 3
```

**Fungsinya yaitu,** Menetapkan variabel **k** dengan nilai 3. Dalam konteks ini, kita sudah tahu dari dokumentasi dataset Wine bahwa ada 3 jenis wine. Kita sengaja memilih **k=3** untuk menguji apakah K-Means dapat menemukan ketiga kelompok ini secara mandiri hanya berdasarkan karakteristik fiturnya.


**Membuat dan Melatih Model K-Means**

Di sinilah proses machine learning yang sebenarnya terjadi. Kita membuat model dan "melatihnya" dengan data kita.

```python
kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
```

**Fungsinya yaitu,** Membuat sebuah instance (objek) dari model K-Means.
- **n_clusters=k:** Parameter paling penting, menentukan jumlah cluster yang akan dibuat (dalam hal ini, 3).
- **init='k-means++':** Ini adalah metode cerdas untuk menempatkan titik pusat cluster awal, yang membantu model menemukan hasil yang lebih baik dan lebih cepat.
- **n_init='auto':** Pengaturan modern di scikit-learn untuk secara otomatis menentukan berapa kali algoritma K-Means akan dijalankan dengan titik awal yang berbeda untuk menemukan hasil terbaik.
- **random_state=42:** Ini adalah "benih" untuk generator angka acak. Mengaturnya memastikan bahwa setiap kali kode ini dijalankan, hasil clustering akan **selalu sama**. Ini sangat penting untuk reproduktibilitas. Jika tidak diatur, titik awal bisa berbeda setiap kali dan hasil clusternya bisa sedikit berbeda.

```python
kmeans.fit(X)
```

**Fungsinya yaitu,** Perintah untuk "melatih" model. Model **kmeans** akan melihat semua data di **X** dan menjalankan algoritma K-Means untuk menemukan 3 pusat cluster terbaik yang dapat meminimalkan jarak antara titik data dengan pusat clusternya masing-masing.

```python
df['cluster_kmeans'] = kmeans.labels_
```

**Fungsinya yaitu,** Setelah model dilatih (**fit**), ia akan menyimpan hasil pengelompokan di dalam atribut **.labels_.** Atribut ini berisi sebuah array di mana setiap elemennya adalah label cluster (0, 1, atau 2) untuk setiap baris data. Baris ini membuat kolom baru bernama **cluster_kmeans** di DataFrame kita dan mengisinya dengan hasil label tersebut.


**Visualisasi Hasil Clustering dengan PCA**

Data kita memiliki 13 dimensi (fitur), yang tidak mungkin kita gambar di layar. Langkah ini bertujuan untuk menyederhanakan data menjadi 2 dimensi agar bisa divisualisasikan.

```python
pca = PCA(n_components=2)
```

**Fungsinya yaitu,** Membuat sebuah instance dari model PCA dan memberitahunya bahwa kita ingin mengurangi data menjadi **2** komponen utama (dimensi).

```python
principal_components = pca.fit_transform(X)
```

**Fungsinya yaitu,** Menerapkan PCA ke data **X**. Metode **fit_transform** melakukan dua hal:
- **fit:** PCA "mempelajari" data **X** untuk menemukan 2 sumbu (komponen) baru yang dapat menangkap variasi data sebanyak mungkin.
- **transform:** PCA kemudian memproyeksikan data 13-dimensi kita ke 2 sumbu baru tersebut, menghasilkan data versi 2-dimensi.

```python
pca_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
```

**Fungsinya yaitu,** Membuat DataFrame baru, **pca_df**, yang berisi data 2-dimensi hasil PCA. Kolomnya diberi nama **pc1** (Principal Component 1) dan **pc2** (Principal Component 2).

```python
pca_df['cluster_kmeans'] = df['cluster_kmeans']
```

**Fungsinya yaitu,** Menyalin hasil label cluster dari DataFrame asli (**df**) ke DataFrame PCA (**pca_df**) agar kita bisa menggunakannya untuk pewarnaan plot.

```python
plt.figure(...)
plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_df['pc1'], pca_df['pc2'], c=pca_df['cluster_kmeans'], cmap='viridis')
...
```

**Fungsinya yaitu,** Kode ini membuat grafik pertama.
- **plt.scatter():** Membuat diagram sebar.
- **x=pca_df['pc1'], y=pca_df['pc2']:** Menggunakan dua komponen utama sebagai sumbu X dan Y.
- **c=pca_df['cluster_kmeans']:** **Ini bagian kuncinya**. **c** adalah parameter warna. Setiap titik akan diwarnai berdasarkan label cluster (0, 1, atau 2) yang ditemukan oleh K-Means.
- **cmap='viridis':** Memilih skema warna.
- Sisa barisnya (**plt.title, xlabel, ylabel, legend, grid**) adalah untuk memberi label dan mempercantik grafik.



**Bandingkan dengan Label Asli (Ground Truth)**

Langkah terakhir ini adalah momen pembuktian. Kita membandingkan hasil "tebakan" K-Means dengan jawaban yang sebenarnya.

```python
pca_df['jenis_asli'] = wine.target
```

**Fungsinya yaitu,** Membuat kolom baru **jenis_asli** di DataFrame PCA dan mengisinya dengan label asli dari dataset (**wine.target**), yang berisi kelas wine sebenarnya (0, 1, atau 2).

```python
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(pca_df['pc1'], pca_df['pc2'], c=pca_df['jenis_asli'], cmap='viridis')
...
```

**Fungsinya yaitu,** Kode ini membuat grafik kedua, persis di sebelah yang pertama.
- Titik-titik yang diplot (**pc1** vs **pc2**) sama persis dengan grafik pertama.
- **Perbedaan utamanya** ada di sini: c=pca_df['jenis_asli']. Sekarang, warna setiap titik ditentukan oleh **jenis wine yang sebenarnya**, bukan hasil tebakan K-Means.
- **plt.legend(..., labels=wine.target_names):** Legendanya sekarang menampilkan nama kelas wine asli (**class_0, class_1, class_2**).


```python
plt.tight_layout()
plt.show()
```

**Fungsinya yaitu,** **tight_layout()** mengatur ulang plot agar tidak tumpang tindih. **show()** menampilkan kedua grafik yang telah kita buat ke layar.


**Alur Penggunaan Secara Rinci (Workflow)**

Secara ringkas, alur kerja dari kode ini adalah sebagai berikut:
- **Persiapan (Setup)**: Kita memulai dengan mengimpor semua "perkakas" yang dibutuhkan: **pandas** untuk data, **matplotlib** untuk gambar, dan **sklearn** untuk model machine learning (K-Means dan PCA).
- **Pemuatan Data (Data Loading):** Dataset "Wine" yang sudah ada di **sklearn** dimuat. Data ini terdiri dari 178 sampel wine, masing-masing dengan 13 karakteristik kimia (fitur) dan satu label asli yang menunjukkan jenis winenya (salah satu dari tiga kelas).
- **Pemodelan (Modeling):**
  - Kita memutuskan untuk mencari **3 kelompok (k=3)** karena kita tahu ada 3 jenis wine. Ini adalah asumsi awal kita.
  - Sebuah model K-Means dibuat dan dikonfigurasi untuk mencari 3 cluster.
  - Model tersebut "dilatih" menggunakan 13 fitur dari data wine. Selama pelatihan, K-Means secara iteratif menemukan 3 titik pusat (centroid) dan mengelompokkan setiap sampel wine ke centroid terdekat. Proses ini dilakukan **tanpa melihat label asli sama sekali.**
- **Reduksi Dimensi untuk Visualisasi (Dimensionality Reduction):**
  - Kita menghadapi masalah: bagaimana cara melihat pengelompokan data yang memiliki 13 dimensi
  - Solusinya adalah menggunakan **PCA** untuk "meringkas" 13 fitur tersebut menjadi 2 "fitur super" (Principal Components) yang paling mewakili keragaman data. Ini memungkinkan kita memplot data pada grafik 2D.
- **Visualisasi dan Evaluasi (Visualization & Evaluation):**
  - **Grafik Kiri (Hasil K-Means)**: Kita membuat scatter plot dari data 2D hasil PCA. Warna setiap titik didasarkan pada cluster yang ditemukan oleh K-Means. Ini menunjukkan bagaimana model mengelompokkan data.
  - **Grafik Kanan (Kebenaran/Ground Truth):** Kita membuat scatter plot kedua dari data 2D yang sama. Namun, kali ini warna setiap titik didasarkan pada **jenis wine yang sebenarnya (label asli).**
  - **Perbandingan:** Dengan meletakkan kedua grafik bersebelahan, kita dapat secara visual membandingkannya. Jika pola warna di grafik kiri sangat mirip dengan pola warna di grafik kanan, itu berarti algoritma K-Means berhasil menemukan struktur alami dalam data dengan sangat baik, meskipun ia tidak pernah "diberi contekan" label aslinya.
