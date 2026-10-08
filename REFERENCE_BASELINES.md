# Referans yöntem değerlendirmesi

Bu betik, `makale.ipynb` içindeki model eğitimi ve FPGA/HLS hücrelerini çalıştırmadan
üç temel yöntemi değerlendirir:

- Persistence: hedef saatten 1 saat önceki `Energy_kWh`
- Günlük: hedef saatten 24 saat önceki `Energy_kWh`
- Haftalık: hedef saatten 168 saat önceki `Energy_kWh`

Bütün referanslar tam zaman damgası eşleşmesiyle alınır. Saatlik eksikler doldurulmaz.
Saatlik tablo; dört değişkende ortalama, üç alt sayaçta toplam ve her yedi değişkende
saat başına 60 gerçek dakika şartıyla ham veriden yeniden hazırlanır.

## Çalıştırma

Depoyu indirin/klonlayın ve UCI veri dosyasını yerel makineye koyun. Ardından depo
kökünde:

```bash
python -m pip install numpy pandas scikit-learn
python reference_baselines.py --data household_power_consumption.txt
```

Dosya başka bir konumdaysa:

```bash
python reference_baselines.py --data "/tam/yol/household_power_consumption.txt" --output-dir reference_baseline_results
```

Google Colab’da yalnızca bu betiği ve veri hazırlama/değerlendirme akışını çalıştırın.
`makale.ipynb` içindeki “Run all” düğmesini kullanmayın. Colab’da betiği ve veri
dosyasını oturuma yükledikten sonra aynı komutu bir hücrede çalıştırın:

```python
!python reference_baselines.py --data /content/household_power_consumption.txt
```

## Üretilen dosyalar

Betik `reference_baseline_results/` dizinine üç dosya yazar:

- `reference_baseline_metrics.csv`: her yöntemin MAE (kWh), RMSE (kWh), R²,
  toplam test hedefi, 24 saatlik süreklilik durumu, değerlendirilen hedef sayısı ve
  kesintisiz adaylarda eksik geçmiş sayısı.
- `reference_baseline_common_metrics.csv`: üç yöntemi ortak geçerli test hedefleri üzerinde
  karşılaştıran MAE, RMSE ve R² tablosu.
- `reference_baseline_predictions.csv`: her test hedefinin zaman damgası, gerçek
  tüketimi, üç referans zaman damgası/tahmini, geçmiş bulunurluğu ve 24 giriş saati
  ile hedefin kesintisiz olma göstergesi.
- `reference_baseline_run_summary.json`: ham/saatlik satır sayıları, split tarihleri,
  süreklilik denetimi ve eski test döneminin kaynakta doğrulanıp doğrulanamadığı.

Ana metrikler sadece 24 giriş saati ile hedefi kesintisiz olan test hedeflerinde
hesaplanır. Süreklilik nedeniyle elenen adaylar ve bu kesintisiz adaylarda referans
geçmişi olmayan hedefler ayrı raporlanır. Ek ortak hedef metriği, üç yöntemin de
geçmiş karşılığı bulunan aynı hedef alt kümesini karşılaştırır.

Split, kabul edilmiş temizleme işleminden sonra notebook’ta tanımlı kronolojik
%70/%15/%15 satır oranını yeniden uygular. Temizleme satır sayısını değiştirebildiği
için bu test tarihleri eski deney tarihleriyle otomatik olarak aynı kabul edilmez.
Özet JSON bu belirsizliği açıkça kaydeder. 5 Ekim doğrulama sayılarıyla fark varsa
betik satır kesmez; gözlenen farkı ve sayıları raporlar.

Betik başlarken küçük bir eksik-saat örneğiyle gerçek zaman gecikmesi ve kesintisizlik
kontrolünü doğrular. Veri dosyası yoksa anlaşılır bir hata verir. Hiçbir model eğitimi,
seed araması, latency veya HLS deneyi çalıştırmaz.
