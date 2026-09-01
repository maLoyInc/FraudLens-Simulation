# Fraud Transaction Detection App

Aplikasi Streamlit untuk memprediksi apakah sebuah transaksi kartu kredit
**aman** atau **berpotensi fraud**, menggunakan model XGBoost yang sudah
dilatih sebelumnya (skripsi).

## Install Dependency

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run website.py
```

## Artifact yang Dibutuhkan

File berikut harus berada di folder yang sama dengan `website.py`:

- `ordinal_encoder.pkl`
- `fraud_scaler.pkl`
- `xgboost_fraud_model.pkl`
- `fraudTrain_dataset_cleaned.csv`

## Catatan Penting

- Aplikasi ini bersifat **inference-only**: `website.py` hanya melakukan
  `pickle.load()` terhadap ketiga artifact di atas dan tidak pernah melatih
  ulang model, encoder, atau scaler saat dijalankan.
- File `.pkl` di atas adalah hasil resmi dari proses training pada
  `dataset_training.ipynb` / `xgb_modeling.ipynb`. **Jangan ditimpa** oleh
  aplikasi maupun proses lain — aplikasi memang sudah tidak melakukan ini,
  tapi pastikan modifikasi di masa depan tetap mempertahankan perilaku ini.
- `fraudTrain_dataset_cleaned.csv` hanya dipakai aplikasi untuk mengisi
  opsi dropdown pada form input, bukan untuk melatih ulang apa pun.
