# Ortak test kümesinde son karşılaştırma

Danışmanın son isteği: üç baseline ve dört modeli aynı test timestamp'lerinde
karşılaştırmak. Kayıtlı tahminler kullanılır; model eğitimi, makale değişikliği ve
FPGA/HLS işlemi yapılmaz. Eğitim kodu ve model ayarları değişmez.

## Kullanım

[Colab'da aç](https://colab.research.google.com/github/yagmurdamlakara/enerji-tuketimi-tahmini/blob/main/common_test_comparison_colab.ipynb)

Ham veri yolunu ve önceki tamamlanmış eğitim klasörünün `RESULTS_DIR` yolunu
ayarlayın. Bu klasörde `run_summary.json`, `model_test_targets.csv` ve dört
model × beş seed için 20 prediction CSV bulunmalıdır. Yerel kullanım:

```bash
python common_test_comparison.py --data household_power_consumption.txt --results-dir tamamlanmis_egitim --output-dir yeni_ortak_test_sonuclari
python -m unittest discover -s tests -v
```

## Hesaplama ve kontroller

Ham veri SHA256 özeti, eğitim denetim kaydıyla eşleşmelidir. Saatlik veri aynı
60 geçerli dakika kuralıyla yeniden hazırlanır. Bütün tahmin dosyaları aynı
4.945 hedefi, gerçek değerleri, model/seed kimliğini ve geçerli sayısal tahminleri
taşımalıdır. Eksik seed, farklı hedef veya gerçek değer, mükerrer timestamp
sessizce elenmez; işlem hata vererek durur.

Baseline değerleri tam olarak t−1, t−24 ve t−168 zamanlarından alınır; satır
kaydırma veya eksik doldurma uygulanmaz. Ortak küme, bu üç geçmiş değerin
tamamının mevcut olduğu model hedeflerinden oluşur. Her seed'in MAE, RMSE ve
R² değeri bu kümede yeniden hesaplanır; model satırları beş **metriğin**
aritmetik ortalamasıdır. Tahminlerin ortalamasından ensemble oluşturulmaz.
Standart sapmalar (ddof=1) ayrı arşiv dosyasında saklanır.

## 11 Ekim 2026 sonucu

Kaynak: `temiz_veri_tum_sonuclar.zip`, status complete ve 20/20 koşu.
Orijinal model hedefi 4.945; ortak hedef 4.730; 168 saatlik geçmişi bulunmayan
215 hedef bütün yöntemlerden birlikte çıkarıldı. 1 ve 24 saatlik geçmişte
eksik yok. İlk ve son ortak hedefler 2010-04-18 21:00 ve 2010-11-26 20:00'dır.

| Yöntem | Örnek sayısı | MAE kWh | RMSE kWh | R² |
|---|---:|---:|---:|---:|
| Persistence | 4730 | 0.383405 | 0.580934 | 0.322682 |
| 24 saat önce | 4730 | 0.502169 | 0.749225 | -0.126584 |
| 168 saat önce | 4730 | 0.549910 | 0.793364 | -0.263233 |
| MLP | 4730 | 0.365172 | 0.546114 | 0.401300 |
| LSTM | 4730 | 0.320910 | 0.466462 | 0.563295 |
| GRU | 4730 | 0.324226 | 0.469233 | 0.558070 |
| BiLSTM | 4730 | 0.324220 | 0.472156 | 0.552503 |

LSTM üç metrikte de en iyi sonucu verdi. Bütün modeller persistence'tan daha
iyi; LSTM'nin MAE'si yaklaşık %16,3, RMSE'si %19,7 düşük. Sonuçlar bu ortak
test kümesine aittir; başka konutlara genellenmiş başarı veya donanım
performansı anlamına gelmez. Yeni hedef kümesindeki değerler önceki 4.945
örnekli skorlarla karıştırılmamalıdır.

## Çıktılar

- `common_test_comparison.csv`: hocaya gönderilecek yedi satırlık tablo.
- `common_test_targets.csv`: ortak hedef zamanları ve gerçek değerleri.
- `common_test_predictions.csv`: aynı sırada üç baseline ve 20 model tahmini.
- `common_per_seed_metrics.csv`: ortak kümedeki 20 seed metriği.
- `common_model_variability.csv`: model başına seed standart sapmaları.
- `comparison_audit.json`: sayılar, veri ve dosya özetleri, değerlendirme kuralı.

Doğrulanmış küçük tablo, denetim kaydı ve dört cümlelik yorum
`results/common_test_20261011` altında tutulur. Hocaya yalnız sonuç CSV'si,
ortak N=4.730 ve kısa yorum gönderilir; detaylı dosyalar arşivlenir.
