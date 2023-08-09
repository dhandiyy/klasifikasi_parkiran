# klasifikasi_parkiran
![](hasil_short.gif)
## **Klasifikasi kendaraan di parkiran dan penempatannya menggunakan TensorFlow**

Model klasfikasi untuk menentukan objek kendaraan dan kebenaran letak kendaraan dalam proses parkir. Terdapat dua model klasifikasi pada projek ini. Model pertama untuk menentukan kendaraan (kelas) yaitu Mobil, Motor, Orang dan Kosong. Model kedua untuk menentukan apakah kendaraan parkir sudah sesuai dengan garis parkiran atau tidak. 
![](video2.gif)

## ***Dataset***

Saya membuat dataset sendiri dengan menggunakan video yang sudah diambil sebelumnya. link dataset dapat di download here.
Dataset model pertama:
https://www.kaggle.com/datasets/dhandiyy/new-parkir
Dataset model kedua:
https://www.kaggle.com/datasets/dhandiyy/parkir-2

Ukuran gambar pada dataset disesuaikan sesuai dengan ukuran kotak parkir agar mendapatkan hasil yang lebih akurat. Dataset pada model pertama terdapat empat kelas yaitu Mobil, Motor, Orang dan Kosong. Dataset pada model kedua menggunakan dua kelas yaitu keadaan parkir yang benar dan salah.

## ***Model***

Pembuatan kedua model tersebut menggunakan teknik ****Transfer Learning**** memanfaatkn model yang sudah ada yaitu MobilNetV3 agar mendapatkan model yang berukuran kecil tetapi tetap dapat diandalkan. Model tersebut nantinya dapat digunakan pada perangkat yang mempunyai tingkat komputasi yang rendah.

<img src="mobil.jpeg" alt="Hasil deteksi kendaraan Mobil" width="400" height="300">
<img src="motor.jpeg" alt="Hasil deteksi kendaraan Motor" width="400" height="300">
<img src="kosong dan orang.jpeg" alt="Hasil deteksi Orang dan Keadaan kosong" width="400" height="300">

Saat model mendeteksi mobil garis kotak akan berwarna hijau, berwarna merah saat mendeteksi motor dan berwarna biru saat mendeteksi orang dan tidak ada objek. Pengaturan warna garis kotak dapat disesuaikan sesuai kebutuhan. Model kedua akan mengeluarkan informasi ketika mobil terparkir dengan benar maupun tidak

##
Pembuatan model pertama:
https://colab.research.google.com/drive/1Mo4g1imyqN90GUwIAm611FpXaBCyXgvF?usp=sharing
Pembuaan model kedua:
https://colab.research.google.com/drive/1DdNjcNxFeU-n2RwIhuf9DRajPU0G5CRl?usp=sharing



