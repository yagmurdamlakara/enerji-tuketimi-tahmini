# Enerji Tüketimi Tahmini

Saatlik elektrik tüketimi tahmini üzerine akademik çalışma.

Bu depo, veri ön işleme ve tahmin yöntemlerine ilişkin kodların değişiklik geçmişini takip etmek için kullanılmaktadır.

## Google Colab

[Son ödevin ortak test karşılaştırmasını Colab'da aç](https://colab.research.google.com/github/yagmurdamlakara/enerji-tuketimi-tahmini/blob/main/common_test_comparison_colab.ipynb)

Son ödev yeniden eğitim gerektirmez. Tamamlanmış eğitim klasörünü seçin;
kayıtlı 20 tahmin dosyasından yedi yöntemin aynı zamanlardaki metriklerini hesaplayın.
Uygulama ve doğrulanmış sonuçlar: [ortak test karşılaştırması](COMMON_TEST_COMPARISON.md).

[Temiz veri model eğitimini Colab'da aç](https://colab.research.google.com/github/yagmurdamlakara/enerji-tuketimi-tahmini/blob/main/clean_data_models_colab.ipynb)

T4 GPU seçin ve hücreleri sırayla çalıştırın. Ham `household_power_consumption.txt`
dosyasının Drive yolunu ayarlayın veya yükleme penceresinden dosyayı seçin.
Sonuçlar Drive'da her çalıştırma için ayrı klasöre kaydedilir.

Mevcut deney korunur: 24 saat × 12 özellik, kronolojik %70/%15/%15 ayrım,
yalnız train üzerinde MinMaxScaler, aynı dört mimari ve 11/22/33/44/55 seed'leri.
Eksik saat üzerinden atlayan pencereler kullanılmaz. Eğitim ayarları Adam 0.001,
batch 256, shuffle=False, en çok 150 epoch ve patience=10'dur.

## Kod dosyaları

- `common_test_comparison_colab.ipynb`: son ödevin Colab değerlendirme arayüzü.
- `common_test_comparison.py`: kayıtlı tahminler ve baseline'lar için ortak test değerlendirmesi.
- `clean_data_models_colab.ipynb`: Colab ve Drive çalıştırma arayüzü.
- `clean_data_rerun.py`: veri hazırlama, eğitim ve sonuç kaydı.
- `original_model_definitions.py`: mevcut model ve eğitim fonksiyonları.
- `reference_baselines.py`: tam zaman eşleşmesiyle 1/24/168 saat referansları.
- `makale.ipynb`: önceki araştırma, ablasyon ve donanım deneyleri.

Güncel eğitim için küçük Colab notebook'unu kullanın. Araştırma notebook'unun
tamamını çalıştırmak ek eğitim ve HLS işlemlerini başlatır; HLS sentezi için
standart Colab ortamında bulunmayan Vitis araçları gerekir.

Kullanım ayrıntıları: [temiz veri eğitimi](CLEAN_DATA_RERUN.md) ve
[referans yöntemler](REFERENCE_BASELINES.md).

## Yerel doğrulama

Gerekli paketler: `numpy`, `pandas`, `scikit-learn`, `joblib`, `torch`.

```bash
python -m unittest discover -s tests -v
python clean_data_rerun.py --data household_power_consumption.txt --output-dir veri_kontrolu --prepare-only
```

`--prepare-only` eğitim yapmaz. Yeni bir eğitim için boş bir çıktı klasörü kullanın.
Tamamlanmış tahminleri değerlendirmek için yeniden eğitim gerekmez.
