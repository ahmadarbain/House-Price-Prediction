# Laporan Proyek Machine Learning - Ahmad Arbain

## Domain Proyek

Bagi sebagian besar orang memiliki rumah idaman merupakan sebuah kebanggan tersendiri apalagi dengan membeli rumah dengan harga terjangkau. Namun dengan seiring berjalannya waktu hingga saat ini penjualan properti rumah yang di idam idamkan sesuai keiginan harganya kian meroket apalagi dengan kondisi saat ini yang membuat kita sulit menentukan rumah yang kita inginkan sesuai dengan baget kantong.

Melihat permasalahan ini membuat saya ingin mengembangkan sebuah proyek berbasis Ekonomi Bisnis yang dapat membantu orang orang dalam melakukan penjualan maupun pembellian rumah dengan harga yang sesuai. Dalam mengembangkan proyek ini pemilihan metode regresi merupakan solusi yang ditawarkan dimana dalam menyelesaikan permasalahan ini membandingkan 3 alogartima yang sesuai yaitu KNN, Random Forest dan Boosting. Selain itu dalam menyelesaikan permasalhan ini pula diperlukan beberapa referensi dalam penyelesainnya berikut dalam [tautan](https://journal.ithb.ac.id/telematika/article/view/321)

## Business Understanding
### Problem Statements
* Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap harga jual rumah?
* Berapa harga rumah dengan karakteristik tertentu?  

### Goals
* Mengetahui fitur yang paling berkorelasi dengan harga penjualan rumah.
* Membuat model machine learning yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
1.  Melakukan Elskpolari pada Dataset yaitu Exploratory Data Analysis
2. Mencari korelasi antar fitur yang memiliki korelasi yang paling dekat dengan fitur price
3. Menghapus fitur yang tidak memiliki korelasi yang lemah terhadap fitur price dalam kondisi ini yang nilainya dibawah 0.0 (Dapat dilihat pada tabel korelasi)
4. Membagi data menjadi data training dan data set dengan perbandingan 8:2
5. Melakukan Standarisasi pada data training agar memiliki nilai yang hampir sama sehingga mudah dalam melakukan pemrosesan.
6. Melakukan uji data dengan menerapkan tiga algoritma untuk dilakukan perbandingan mana yang lebih baik dalam hal ini menggunakan **KNN*, Random Forest, dan Boosting**
7. Melakukan Uji test data dengan melakukan Standarisasi pada data test (dalam hal ini pemisahan standarisasi dilakukan secara terpisah antara train dan test agar menghindari kebocoran data) kemudian melakukan uji testing pada data.
8. Mendapakan hasil prediksi model yang paling akurat diantara ketiga model

## Data Understanding

Pada pengembangan sistem prediksi harga rumah ini menggunakan sebuah data penjualan rumah dari website [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) di amerika dimana pada data ini terdapat 21 kolom dan 21612 entry dengan variabel dan fitur pada data sebagai berikut :
Variabel-variabel pada House Sales in King County, USA dataset adalah sebagai berikut:
-   **price** : merupakan harga rumah di King County, USA
-   **bedrooms** : merupakan kamar pada rumah
-   **bathrooms**  : merupakan kamar mandi pada rumah
-   **sqft_living** : merupakan ruang tamu pada rumah dengan luas kaki persegi
-   **sqft_lot** : merupakan lot pada rumah dalam luas kaki persegi
-   **floors** : merupakan jumlah lantai pada rumah
-   **waterfront** : merupakan rumah di kawasan waterfront
-   **view** : merupakan rumah dengan view
-   **condition** : merupakan kondisi rumah
-   **grade** : merupakan kelas pada rumah
-   **sqft_above** : merupakan luas atap pada rumah dalam satuan kaki persegi
-   **sqft_basement** : merupakan luas basement dalam satuan kaki persegi
-   **yr_built** : merupakan tahun di bangunnya rumah
-   **yr_renovated** : merupakan tahun rumah terakhir kali di renovasi
-   **zipcode** : merupkan kode zip rumah
-   **lat** : merupakan luas rumah
-   **long** : merupakan panjang rumah
-   **sqft_living15** : merupakan panjang dan lebar dalam 1500 kaki persegi
-   **sqft_lot15'** : merupakan luas lot dalam 1500 kaki persegi

## Data Preparation

Pada Data Preparation dalam mengembangkan model ini menggunakan salah satu meto pengembangan yaitu Train-Test-Split. Train-Test-Split merupakan metode untuk mambagi data menjadi data training dan data test dimana pada pengngembangan model ini menggunkan pembagian data sebesar 80% data training dan 20% data Test. Pemilihan penggunaan metode ini karena untuk membagi data menjadi data training dan data test agar model dapat melakukan training yang baik dan test dengan rasio 8:2

```
from sklearn.model_selection import train_test_split
 
X = df.drop(["price"],axis =1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```

Selain itu dalam melakukan data processing dilakukan pula proses Standarization dimana tujuannya dilakukan proses ini karena. Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Dalam menerapkan metode ini pula dilakukan dua kali proses yaitu saat data training dan saat data testing tujuannya untuk menghindari kebocoran data.

```
from sklearn.preprocessing import StandardScaler
 
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade',
       'sqft_above', 'sqft_basement', 'lat', 'sqft_living15']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
```

## Modeling
Pada tahap pengembangan model Machine Learning ini dalam membuat modelnya dilakukan tahapan menggunakan tiga model alogaritma yang kemudain dilakukan perbandingan dan memilih alogaritma terbaik yang akan dilakukan pengembangan :
-   KNN Alogarithm
-   Random Forest Alogarithm
-   Boosting Alogarithm

Tahap awal pengembangan model yaitu dilakukan tahap pengembnagan dengan membuat data frame dari ketiga algoritma **KNN, RF, Boosting**

```
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])
```

Kemudian setelah dilakukan proses dataframe selanjutnya mulai mengembangkan model KNN dengan K = 10. Pada tahap ini hanya melatih data testing dan menyimpan data tersting untuk evaluasi.
```
from sklearn.neighbors import KNeighborsRegressor
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_train)
```
Selanjutnya menggunakan model development dengan Random Forest
````
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
 
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)          
````

pada model diatas merupakan algoritma Random Forest dimana pada parameter pertama menggunakan n_estimator sebanyak 50 artinya pada penerapan prosesnya dilakukan sebanyak 50 semakin banyak yang digunakan maka hasilnya akan semakin baik tetpi akan akan membuat proses running semakin lambat. Kemudian parameter max_depth yaitu merupakan parameter untuk menentukan depthnya yitu sebanyak 16 semakin besar depthnya maka akan semakin dalam informasi yang akan di dapatkan. sedangkan random state merupakan parameter agar dataset yang di training tidak konsisten.

Selanjutnya ialah model Algoritma Boosting 
```
from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

Boosting atau yang dikenal dengan nama AdaBoost merupakan algoritma yang digunakan untuk meningkat performa model dengan menggabungkan model yang lemah menjadi model yang kuat. 

Berdasarkan hasil perbandingan ketiga model algoritma diatas hasil terbaik diperlihatkan oleh model algoritma Random Forest dimana model erronya lebih sedikit dan pada proses testing model yang mendekati hasilnya ialah random forest.

## Evaluation

Pada tahap ini menggunakan metrik mse, mse melakukan perhitngan selisih rata rata sebenernya dengan nilai prediksi. pada dasarnya setiap metrik melakukan hal yang sama yaitu jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror.

![metrik](https://lms.onnocenter.or.id/wiki/images/thumb/1/18/Metric1.png/200px-Metric1.png)

dimana : 
-   N = jumlah dataset

-   yi = nilai sebenarnya

-   y_pred = nilai prediksi

Dalam menerapkan metrik ini pada model sebelum melakukan test data hal yang dilakukan pertama ialah melakukan standarization pada data test agar hasil data test dan data training memilik nilai yang sama selain itu dengan menerapkan standarization membantu mengurangi kebocoran data. Setelah dilakukan test data didapatkan hasil dimana nilai test yang diberikan sebesar 472500.0	dan model memberikan nilai yang mendekati yaitu pada RF sebesar 433256.6	dengan nilai prediksi price 10943

