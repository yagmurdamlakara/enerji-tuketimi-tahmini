# Temiz veri üzerinde modellerin yeniden eğitimi

Colab: https://colab.research.google.com/github/yagmurdamlakara/enerji-tuketimi-tahmini/blob/feat/clean-data-model-rerun/clean_data_models_colab.ipynb

T4 GPU seçin, hücreleri sırayla çalıştırın. Ham dosyanın Drive yolunu veri
hücresinde ayarlayın veya açılan pencereden household_power_consumption.txt yükleyin.
Sonuçlar Drive/MyDrive/enerji_temiz_model_sonuclari altında yeni bir klasöre kaydedilir.
Eski makale.ipynb çalıştırılmaz. Bu çalışma makale ve FPGA dosyalarını değiştirmez.

## Korunan ayarlar ve kaynak

Kaynak: makale.ipynb, commit 6d74b9b; model factory ve son beş-seed deneyi.
Model sınıfları ve eğitim fonksiyonları original_model_definitions.py içine
kaynak metinleri değiştirilmeden alınmıştır. Eski notebook'ta ilk deneyler 100,
nihai beş-seed deneyi 150 epoch kullanır; burada nihai deney korunur.

- 24 saat giriş, sonraki saatin Energy_kWh hedefi, aynı sırada 12 özellik.
- MLP: 288 → 64 ReLU → 32 ReLU → 1.
- LSTM/GRU: tek katman, hidden_size=64 → 1.
- BiLSTM: tek katman, her yönde 64; h_n[0] ve h_n[1] birleştirilir → 1.
- Adam lr=0.001, MSELoss, batch_size=256, shuffle=False, max_epochs=150.
- Early stopping patience=10, en düşük validation loss ağırlıkları geri yüklenir.
- Seed: 11,22,33,44,55. Standart sapma ddof=1. Test ile model seçimi yapılmaz.
- Aynı seed farklı PyTorch/CUDA/donanım ortamlarında bit düzeyinde eşitlik garantisi değildir;
  çalışma ortamı run_summary.json içine kaydedilir. Eğitim backend ayarları değiştirilmez.

## Veri ve zaman bütünlüğü

Baseline dalındaki load_and_prepare_hourly aynen kullanılır: dört değişkende
saatlik ortalama, üç alt sayaçta toplam, yedi değişkenin her birinde 60 geçerli
dakika. İnterpolasyon, eksik doldurma veya aykırı değer silme yoktur.
Temiz saatler üzerinden önce %70/%15/%15 kronolojik sınırlar belirlenir.
İki MinMaxScaler yalnız train saatlerinde fit edilir, diğerleri transform edilir.
Pencere oluştururken t-24,...,t-1 ve t ardışık saat olmalıdır. Eksik saat üzerinden
atlayan satır pencereleri elenir; split sınırları tekrar hesaplanmaz.
Validation/test önceki split'in geçmiş ölçümlerinden yararlanabilir; gelecek veri
kullanılmaz. Test içindeki önceki gerçek ölçümler bir saat ilerisi kayan tahmin
protokolünde kullanılmaktadır; sabit başlangıçlı çok adımlı tahmin değildir.

Sağlanan ham dosya üzerindeki doğrulanmış sayılar:

| Split | Temiz saat | Pencere adayı | Boşluk nedeniyle elenen | Model örneği |
|---|---:|---:|---:|---:|
| Train | 23859 | 23835 | 1156 | 22679 |
| Validation | 5113 | 5113 | 288 | 4825 |
| Test | 5113 | 5113 | 168 | 4945 |

Train'in ilk 24 saati yalnız giriş geçmişidir. Her model aynı örnekleri kullanır.
Eski baseline ana tablosu 5113 aday hedef içeriyordu; model testi 4945 hedeftir.
Eski skorları doğrudan karşılaştırmayın. Sonraki aşamada model_test_targets.csv
ile baseline yeniden hesaplanmalı ve haftalık geçmişi eksik hedefler varsa tüm
modeller ve referanslar aynı ortak zaman damgalarına indirgenerek metrikler yeniden
hesaplanmalıdır. Bu görev baseline karşılaştırmasını henüz yapmaz.

## Çıktılar

- advisor_table.csv/.txt: dört model için MAE, RMSE, R2 ortalama ± std ve örnek sayıları.
- model_summary_numeric.csv: yuvarlanmamış ortalama ve standart sapmalar.
- per_seed_metrics.csv: 20 deney; en iyi epoch, metrikler, parametre ve örnek sayıları.
- MODEL_seed_N_predictions.csv: target_timestamp, actual_kwh, prediction_kwh, model, seed.
- model_test_targets.csv: sonraki baseline hesabı için tam hedef listesi.
- sequence_manifest.csv: tüm split'lerde giriş başlangıç/bitiş ve hedef zamanları.
- split_counts.csv, scalers.joblib, run_summary.json: veri/ölçekleyici/ortam denetimi.
- Her seed için .pth ağırlıkları ve eğitim/validation kayıpları.

Çıktı dizini boş olmalıdır; önceki deneyin dosyaları üzerine yazılmaz.
Kesinti halinde tamamlanan seed çıktıları kalır; otomatik devam özelliği yoktur.
Tam rapor sadece 20 deney tamamlandığında üretilir. Eğitim sonuçları henüz
üretilmemiştir; ön işleme sayıları model başarımı değildir.

Yerel çalıştırma (numpy, pandas, scikit-learn, joblib, torch gerekir):

```bash
python clean_data_rerun.py --data household_power_consumption.txt --output-dir clean_model_results
```

Sadece veri denetimi için ayrıca --prepare-only kullanın (torch gerektirmez).
