# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Institute

## Business Understanding
Jaya Jaya Institute adalah salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Jaya Jaya Institute dikenal sebagai institusi yang menerima siswa dari berbagai macam kalangan. Institusi pendidikan ini memiliki kelas pagi dan malam dan menyediakan lebih dari 10 pilihan jurusan. 

### Permasalahan Bisnis
Permasalahan bisnis yang dihadapi oleh Jaya Jaya Institute adalah tingginya dropout rate, yang berarti siswa tidak menyelesaikan pendidikannya. Hal ini disebabkan banyaknya jurusan yang harus diawasi dan faktor-faktor lainnya. Hal tersebut mendorong pihak institusi pendidikan untuk mencari tahu penyebab tingginya dropout rate sehingga dapat meminimalisasi kemungkinan terjadinya dropout.

### Cakupan Proyek
1. Menganalisis faktor penyebab tingginya dropout rate
2. Membuat model machine learning dan prediksi sederhana dan di-deploy pada streamlit
3. Membangun dashboard menggunakan streamlit

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

Setup environment:
1. Membuat environment baru bernama newenv
```
python -m venv newenv
```
2. Aktivasi environment
```
.\newenv\Scripts\activate
```
3. Menginstal package yang dibutuhkan
```
pip install -r requirements.txt
```

## Business Dashboard

![dashboard](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/dashboard_final.png)

Pada bagian atas dashboard, terdapat empat slicer yang bertujuan untuk membantu pengguna menfilter sesuai kebutuhan. Slicer tersebut terdiri dari sebagai berikut.
1. Status - terdiri dari "Dropout" dan "Not Dropout". Jika pengguna memilih "Not Dropout", pengguna dapat memilah lebih lanjut menjadi "Enrolled" dan "Graduated".
2. Course/Jurusan - terdiri dari jurusan yang ada. Hanya dapat memilih satu jurusan dalam satu waktu.
3. Attendance Time/Tipe jurusan - terdiri dari daytime (kelas pagi) dan evening (kelas malam).
4. Gender - terdiri dari male dan female.

Di bagian bawah judul terdapat keterangan atau card yang terdiri dari dropout rate dan jumlah siswa yang ada. Keterangan tersebut bersifat dinamis dan berubah sesuai filter yang digunakan. Di bawah keterangan tersebut, terdapat tujuh grafik.
1. Diagram batang penerima beasiswa (scholarship holder) berdasarkan status.
2. Diagram batang rata-rata nilai semester 1 dan semester 2. Terdapat selisih dan persentase dari semester 1 ke semester 2.
3. Diagram batang yang menunjukkan dropout rate dari setiap jurusan. Diagram ini hanya akan muncul apabila tipe status adalah default atau "None".
4. Diagram pie kebutuhan pendidikan khusus yang terdiri dari true (benar) dan false (salah), menunjukkan apakah siswa tersebut memiliki kebutuhan pendidikan khusus atau tidak.
5. Diagram pie penghutang yang terdiri dari true (benar) dan false (salah), menunjukkan apakah siswa tersebut adalah penghutang atau bukan.
6. Diagram pie tuition fees up to date yang terdiri dari true (benar) dan false (salah), menunjukkan apakah siswa tersebut dibebani biaya pendidikan terbaru atau tidak.
7. Histogram dari umur ketika mendaftar (age_at_enrollment)

Link: https://baiwandek-ezg3si5bkkepax.streamlit.app/

## Menjalankan Sistem Machine Learning
Langkah-langkah menggunakan sistem machine learning berbasis random forest adalah sebagai berikut.

1. Membuka link https://baiwandek-ezg3si5bkkepax.streamlit.app/

2. Memilih "Prediction" pada taskbar di sisi kiri.

![dashboard](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/taskbar.png)

3. Mengisi data yang dibutuhkan. Perlu diperhatikan bahwa nilai jurusan atau Course tidak boleh 'None' serta terdapat batas minimum dan maksimum pada input numerik. Selain itu, pengguna harus menekan enter agar dapat menyimpan data numerik.

![dashboard](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/prediksi2.png)

4. Hasil prediksi akan tampil di bagian bawah. Akan tetapi, pesan error akan muncul apabila pengguna tidak memilih jurusan atau Course.

![dashboard](https://raw.githubusercontent.com/dieselgank/Media/refs/heads/main/prediksi3.png)


## Conclusion
- Faktor yang paling memengaruhi siswa untuk dropout adalah biaya pendidikan terbaru. Sebanyak 32,2% siswa dengan biaya pendidikan terbaru melakukan dropout. Bisa jadi, biaya terbaru yang ditetapkan dianggap terlalu tinggi bagi sebagian mahasiswa sehingga melakukan dropout. Ditambah lagi, apabila biaya terbaru tersebut terlalu tinggi juga akan mendorong siswa melakukan hutang, padahal 22% dari pelaku dropout adalah penghutang.
- Sebagian besar dari penerima beasiswa tidak melakukan dropout (dropout rate sebesar 10%).
- Siswa yang melakukan dropout cenderung memiliki nilai yang turun dari semester 1 ke semester 2.
- Jurusan yang memiliki dropout rate tertinggi adalah Biofuel Production Technologies (66,67%) diikuti oleh Equinculture (55,32%) dan Informatics Engineering (54,12%).

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan Jaya Jaya Institute guna menyelesaikan masalah tingginya dropout rate diantaranya sebagai berikut.
- Menyelidiki apakah biaya pendidikan terbaru yang ditetapkan merupakan keputusan yang tepat.
- Mengawasi siswa yang mengalami penurunan nilai. Selain itu, dapat juga dilakukan dengan meningkatkan tenaga kerja sehingga siswa semakin paham akan pelajaran yang diajarkan dan tidak melakukan dropout.
- Melakukan evaluasi terhadap setiap jurusan, khususnya pada Biofuel Production Technologies yang memiliki siswa relatif sedikit (12 orang) namun memiliki dropout rate yang tinggi. Evaluasi dapat dilakukan dengan meninjau kualitas pengajar dan materi yang diberikan. Jurusan Equinculture juga dapat diawasi khususnya pada siswa pria karena memiliki dropout rate yang tinggi (64,516%) padahal memiliki jumlah siswa yang relatif sedang yaitu 62 siswa. Pada jurusan Informatic Engineering juga harus dilakukan pengawasan pada siswa wanita karena 6 dari 7 siswa wanita melakukan dropout.
- Menerapkan pembatasan umur, misalnya maksimal 50 tahun.
- Memberikan lebih banyak penerima beasiswa sekaligus dapat mengurangi kemungkinan siswa berhutang.
