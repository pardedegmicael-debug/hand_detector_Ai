# Hand Motion Tracker

Program Python untuk mendeteksi hingga dua tangan dari webcam atau video, menampilkan landmark dan gesture, menghitung gerakan tangan, serta menyimpan hasil analisis ke CSV.

## Persiapan (Windows PowerShell)

Jalankan perintah berikut dari folder proyek dengan Python 3.11/3.12. Proyek memakai
API legacy `mp.solutions.hands` dan dependensi yang sudah dipastikan kompatibel untuk tracker ini:
`mediapipe==0.10.21`, `opencv-contrib-python<4.12`, `numpy<2`.
Jika folder `.venv311` yang sudah ada digunakan, pasang dependensi di environment itu:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r .\requirements_hand_tracker.txt
```

Untuk membuat environment baru, pastikan Python 3.11 terpasang:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r .\requirements_hand_tracker.txt
```

Jangan gunakan `.venv` berbasis Python 3.13 untuk tracker ini. Jika PowerShell
memblokir aktivasi virtual environment, contoh perintah di atas tetap dapat
dijalankan tanpa aktivasi.

## Menjalankan

Webcam default (kamera 0), jendela tracker, dan pencatatan CSV:

```powershell
.\.venv311\Scripts\python.exe .\hand_motion_tracker.py
```

### Memasukkan hasil tracker sebagai sumber video di OBS

OBS Virtual Camera bawaan OBS tidak dapat dipakai sebagai input kamera ke OBS itu sendiri. Untuk menambahkan
output tracker ke scene OBS, pasang driver [Unity Capture](https://github.com/schellingb/UnityCapture#installation),
lalu jalankan tracker dengan backend `unitycapture`:

```powershell
.\.venv311\Scripts\python.exe .\hand_motion_tracker.py --virtual-camera --virtual-camera-backend unitycapture
```

Di OBS, tambahkan **Video Capture Device** sebagai source dan pilih perangkat Unity Capture. Output memuat
video webcam, landmark tangan, dan panel informasi tracker. Tracker tetap membuka webcam fisik; tutup aplikasi
lain yang sedang memakai webcam tersebut. Gunakan `q`, `Esc`, atau `Ctrl+C` untuk menghentikan tracker.
Gesture pointing dan shutdown dinonaktifkan dalam mode kamera virtual.

### Memakai hasil tracker sebagai kamera di aplikasi lain

Untuk mengirim hasil tracker ke aplikasi lain (misalnya WhatsApp), gunakan OBS Virtual Camera:

1. Pasang OBS Studio di Windows agar driver OBS Virtual Camera tersedia.
2. Jalankan tracker dengan opsi kamera virtual:

   ```powershell
   .\.venv311\Scripts\python.exe .\hand_motion_tracker.py --virtual-camera --virtual-camera-backend obs
   ```

3. Di aplikasi tujuan, buka pengaturan video/kamera dan pilih **OBS Virtual Camera**.

Opsi `--virtual-camera` tanpa pilihan backend tetap memakai OBS Virtual Camera (`obs`) agar kompatibel dengan
perilaku sebelumnya. Dependensi Python untuk mode kamera virtual ikut dipasang lewat
`requirements_hand_tracker.txt`; driver virtual camera yang dipilih harus dipasang terpisah.

Memilih kamera atau ukuran gambar:

```powershell
.\.venv311\Scripts\python.exe .\hand_motion_tracker.py --camera 1 --width 1280 --height 720
```

Menganalisis file video dan menyimpan data ke CSV tanpa membuka jendela:

```powershell
.\.venv311\Scripts\python.exe .\hand_motion_tracker.py --video .\rekaman.mp4 --no-display --csv .\hasil.csv
```

Secara default gambar dicerminkan horizontal seperti tampilan selfie. Tambahkan `--no-mirror` untuk memakai orientasi sumber apa adanya. Pemrosesan video berhenti saat file mencapai akhir; opsi `--no-display` dengan webcam berhenti dengan `Ctrl+C`.

Saat gesture `rock` terdeteksi, tracker memutar `.\.venv311\Include\rock.mp3` berulang kali dan menghentikannya saat pose berubah atau tidak ada tangan yang terdeteksi.

Di webcam langsung pada Windows, pose jari tengah yang lolos pemeriksaan bentuk dan ditahan selama 1,5 detik memulai hitung mundur lokal 15 detik. Pose ini juga ditampilkan sebagai gesture `fuck`, tetapi label gesture saja tidak cukup untuk memicu shutdown. Setelah hitung mundur habis, tracker menjalankan `shutdown.exe /s /t 0 /f`, yang dapat langsung memaksa menutup aplikasi lain dan menyebabkan data yang belum disimpan hilang. Jangan aktifkan fitur ini jika tidak menginginkan shutdown komputer. Gunakan `--disable-shutdown` untuk menonaktifkannya; `x` membatalkan hitung mundur tanpa keluar, sedangkan `q`, `Esc`, atau `Ctrl+C` membatalkan hitung mundur lokal dan menghentikan tracker. Fitur ini tidak aktif untuk video atau kamera virtual.

Opsi lainnya:

- `--max-hands 1` membatasi deteksi ke satu tangan (default: 2).
- `--detection-confidence 0.7` dan `--tracking-confidence 0.7` mengatur ambang keyakinan, dalam rentang 0 sampai 1.
- `--csv nama.csv` memilih lokasi file hasil. File dengan nama yang sama akan ditimpa.
- `--no-csv` menonaktifkan pencatatan CSV.
- `--no-display` memproses sumber tanpa antarmuka visual.

## Kontrol jendela

- `q` atau `Esc`: keluar; jika hitung mundur shutdown berjalan, batalkan hitung mundur.
- `x`: batalkan hitung mundur shutdown tanpa keluar dari tracker.
- `c`: hapus trail dan reset riwayat kecepatan.
- `d`: tampilkan atau sembunyikan detail landmark.
- `s`: simpan screenshot di folder kerja.

## Data keluaran

CSV mencatat satu baris untuk setiap tangan yang terdeteksi pada setiap frame: gesture, arah dan kecepatan, status jari, sudut sendi, serta koordinat 3D ke-21 landmark. Koordinat landmark MediaPipe dinormalisasi; koordinat pusat tangan dan kecepatan dinyatakan dalam piksel serta piksel per detik.

## Menjalankan tes

```powershell
python -m unittest discover -s tests -v
```
