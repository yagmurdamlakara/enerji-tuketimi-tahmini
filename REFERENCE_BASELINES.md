# Referans yöntem değerlendirmesi

Bu betik model eğitmeden Persistence, 24 saat ve 168 saat referansları için MAE
(kWh), RMSE (kWh), R² ve değerlendirilemeyen hedef sayılarını hesaplar.
Hedef T için sırasıyla T−1, T−24 ve T−168 saat değerleri kullanılır.
Saatler zaman damgasıyla eşleştirilir; eksik değer doldurulmaz.

Ham veri hazırlama: dört elektriksel değişkenin saatlik ortalaması, üç alt sayacın
saatlik toplamı ve her değişkende 60 gerçek dakika koşulu korunur.
24+1 saat süreklilik kontrolü yalnızca tanısaldır: referans değerlendirmesinden
ayrıca hedef elemez. Her yöntem sadece geçmiş karşılığı olmayan hedefleri dışlar.
Ortak geçerli hedef tablosu ayrıca üretilir.

## 1. Hocanın istediği aynı model test hedefleri

Model değerlendirmesinde gerçekten kullanılan zaman damgalarını içeren
`model_test_targets.csv` dosyasını sağlayın. Zorunlu sütun `target_timestamp`;
isteğe bağlı `actual_kwh` sağlanırsa temiz verideki değerlerle de doğrulanır.

```text
target_timestamp,actual_kwh
```

Aşağıdaki hücre, YALNIZCA mevcut model değerlendirmesinde `y_test` ile
`test_df` birebir aynı sıradaki hedefleri temsil ediyorsa kullanılabilir.
Eski model sonuçlarıyla karşılaştırmak için bu değişkenler ve scaler aynı model
çalıştırmasına ait olmalıdır. Yeni split üretip eski skorlarla karşılaştırmayın.
Filtrelenmiş sequence kullanılıyorsa `test_df.index` yerine sequence oluştururken
saklanan gerçek hedef zaman damgaları gereklidir; uzunluk eşitliği tek başına
zaman eşleşmesini kanıtlamaz.

```python
import numpy as np
import pandas as pd

actual = target_scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1)).ravel()
assert len(actual) == len(test_df), "Gerçek sequence hedef zamanlarını kullanın."
assert np.allclose(actual, test_df['Energy_kWh'].to_numpy(), rtol=1e-5, atol=1e-6)
pd.DataFrame({
    'target_timestamp': test_df.index,
    'actual_kwh': actual,
}).to_csv('/content/model_test_targets.csv', index=False)
```

Betiği ve ham veri dosyasını Colab'a yükledikten sonra:

```python
!python reference_baselines.py --data /content/household_power_consumption.txt --test-targets /content/model_test_targets.csv
```

CSV hedefleri temiz veride yoksa veya verilen gerçek değerlerle uyuşmuyorsa betik
hata verir; hedefleri sessizce silmez. Test kimliğinin kaynağı kullanıcı tarafından
sağlanan CSV'dir; betik eski model tarihlerini tahmin etmez.

## 2. Güncel temiz veri için ayrı referans hesabı

Eski model test zamanları elinizde yoksa, aşağıdaki açık seçenek yeni
%70/%15/%15 bölmesiyle hesaplama yapar. Bu sonuç eski modellerle aynı test kümesi
olarak sunulamaz; çalıştırma çıktısında ve JSON'da bu durum belirtilir.

```python
!python reference_baselines.py --data /content/household_power_consumption.txt --cleaned-ratio
```

İki seçenekten tam biri zorunludur. Yerel Python'da komutun başındaki `!` kaldırılır.
Gerekli paketler: `numpy`, `pandas`, `scikit-learn`.

## Çıktılar

Varsayılan `reference_baseline_results/` klasöründeki dört dosya:

- `reference_baseline_metrics.csv`: ana üç yöntem tablosu, toplam/değerlendirilen
  hedefler, `missing_history_targets` ve tanısal süreklilik sayıları.
- `reference_baseline_common_metrics.csv`: üç yöntemin ortak geçerli hedefleri.
- `reference_baseline_predictions.csv`: hedef ve geçmiş zamanları, gerçek değerler,
  tahminler ve kullanılabilirlik göstergeleri.
- `reference_baseline_run_summary.json`: veri, test dönemi ve değerlendirme denetimi.

`continuity_invalid_targets` yalnızca bilgi verir; elenen örnek sayısı değildir.
Her yöntem için `evaluated_targets + missing_history_targets = test_targets_total`.
Ortak hedef yoksa ortak metrikler NaN olur; ana tablo yine kaydedilir.
`--output-dir` ile ayrı sonuç klasörü seçilebilir.

Notebook'un tamamını çalıştırmayın. Bu betik model eğitimi, FPGA/HLS deneyi veya
makale değişikliği yapmaz. Ham veri ve eski model test hedefleri depoya eklenmez.
