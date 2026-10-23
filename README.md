# Giyim Mağazası Müşteri–Ürün Eşleştirme Botu

# BU PROJE TASLAK AŞAMASINDADIR, ÇOĞU AI YARDIMI İLE YAPILMIŞTIR

İş Zekası ve Veri Madenciliği dersi dönem projesi.

Bir erkek giyim mağazasının 2018–2022 satış verisini analiz eden ve **yeni müşterilere ürün öneren** bir bot geliştirdik. Bot, müşterinin Türkçe yazdığı isteği anlar (ör. "lacivert gömlek, M beden, bütçem 1500"), mevcut müşterilerin alışverişlerinden öğrendiği kalıplara göre ürün önerir ve nedenini açıklar.

Projede iki bileşen birlikte çalışır:

| Bileşen | Görevi | Dosyalar |
|---|---|---|
| İstatistik ve veri madenciliği | Müşteri segmentasyonu (RFM + k-means), birliktelik kuralları (Apriori), ürün benzerliği (kosinüs), aylık ciro regresyonu, ANOVA / Kruskal-Wallis, ki-kare, müşterinin tekrar gelmesini tahmin eden sınıflandırma modelleri (lojistik regresyon, rastgele orman), öneri değerlendirmesi ve bootstrap güven aralıkları | `src/segmentation.py`, `src/association.py`, `src/recommender.py`, `src/istatistik.py`, `src/tekrar_gelme.py`, `src/evaluate.py` |
| Büyük dil modeli (LLM) | Müşterinin Türkçe mesajından tercihleri çıkarır, önerileri Türkçe açıklar | `src/llm.py` |

LLM ürün seçmez. Ürünleri sayısal modeller seçer, LLM yalnızca dili anlar ve sonucu anlatır. Bu sayede önerilerde uydurma ürün riski olmaz ve öneri kalitesi ölçülebilir kalır.

## Veri

[Clothes Retail Store (Kaggle)](https://www.kaggle.com/datasets/nishchay331/retail-store) veri seti: kurgusal bir erkek giyim mağaza zinciri, her satır bir faturadaki tek ürün kalemi.

| | |
|---|---|
| Dosyalar | `data_2018.csv` … `data_2022.csv` |
| Ham satır | 8.454.383 |
| Temizlik sonrası satır | 6.396.413 |
| Müşteri / fatura | 939.187 / 1.846.890 |

Veri büyük olduğu için depoda yok. Kaggle'dan indirip CSV dosyalarını `data/raw/` klasörüne koymak gerekir; kod klasördeki tüm CSV'leri okuyup birleştirir.

```bash
pip install kaggle
kaggle datasets download -d nishchay331/retail-store -p data/raw --unzip
```

Veri hakkında bilinmesi gerekenler:
- Veride takım elbise kategorisi yok; mağaza erkek giyim olarak ele alındı.
- Tutarların para birimi belirtilmemiş, metinlerde "birim" deniyor.
- Ürün bilgisi çoğu satırda kodlu açıklamada saklı (ör. `MSH ITMSH03378 Chiseled LS Green XL`). Kategori; kategori sütunundan, yoksa açıklamadaki anahtar kelimelerden, yoksa ürün kodu ön ekinden (`MSH` gömlek, `MTR` pantolon vb.) belirlendi. `MSH`, `MTR`, `MDE`, `MBL` ön ekleri örnek satırlarla doğrulandı. `MTS`, `MRG`, `MSS`, `MSW`, `MSO` ön ekleri tahmini eşlemedir (`config.py` içinde `PREFIX_CATEGORY`).
- Anlamı belirlenemeyen `B` ile başlayan kodlar ile çanta ve ambalaj kalemleri kapsam dışı bırakıldı.
- Eksik renk ve beden önce açıklama metninden çıkarıldı, bulunamayanlar `NOT CAPTURED` olarak işaretlendi.

## Kurulum ve çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # LLM için ANTHROPIC_API_KEY buraya yazılır (isteğe bağlı)
```

```bash
python main.py analyze        # temizleme raporu, grafikler, segmentler (PCA, ısı haritası), Apriori kuralları -> outputs/
python main.py evaluate       # öneri kalitesi (zamana göre ayırma) -> outputs/degerlendirme.json
python main.py stats          # ciro regresyonu, ANOVA, ki-kare -> outputs/
python main.py predict        # müşteri 180 gün içinde tekrar gelir mi? (lojistik regresyon + rastgele orman) -> outputs/
python main.py chat           # sohbet botu (çıkış: çık, q veya Ctrl+C)
python -m pytest              # testler
python main.py generate-data  # yalnızca kodu denemek için sentetik veri (rapor sonuçları için kullanılmaz)
```

Bota yazılabilecek örnekler:

```
lacivert gömlek arıyorum, M beden, bütçem 1500   -> yeni müşteri önerisi
müşteri <user_id>                                 -> kayıtlı müşteri önerisi
paket kazak                                       -> kazakla birlikte alınan ürünler
```

`ANTHROPIC_API_KEY` tanımlı değilse bot kural tabanlı yedek modda çalışır (anahtar kelimelerle tercih çıkarır, şablonla açıklar). Anahtar varsa LLM devreye girer (varsayılan model `claude-sonnet-5-5`, `.env` içindeki `LLM_MODEL` ile değiştirilebilir). Renkli terminal arayüzü `rich` kütüphanesi kuruluysa açılır, kurulu değilse düz metin arayüzü kullanılır.

## Yöntem

1. **Temizleme** (`data_loader.py`): geçersiz tutarları ve ürün olmayan kalemleri eler, kategori/renk/beden bilgisini tamamlar. Ürün = kategori + renk (ör. `SHIRTS | NAVY`).
2. **Özellikler** (`features.py`): her müşteri için son alışverişten bu yana geçen gün, fatura sayısı, toplam harcama ve harcamanın kategorilere dağılımı (RFM + kategori payları).
3. **Segmentasyon** (`segmentation.py`): özellikler standartlaştırılıp k-means uygulanır; küme sayısı silhouette skoruyla 3–6 arasından seçilir.
4. **Birliktelik kuralları** (`association.py`): Apriori kütüphane kullanılmadan yazıldı; fatura sepetleri üzerinden support, confidence ve lift hesaplanır.
5. **Mevcut müşteri önerisi:** `skor = 0,6 × benzer ürünler + 0,4 × kendi segmentindeki popülerlik`; müşterinin zaten aldığı ürünler önerilmez.
6. **Yeni müşteri önerisi (soğuk başlangıç):** söylenen kategorilere en yakın segment kosinüs benzerliğiyle bulunur, o segmentte popüler ürünler sıralanır. Kategori, beden ve bütçe filtredir; çok az ürün kalırsa darboğaz olan kriter gevşetilir ve müşteriye söylenir. Renk filtre değil skor artırıcıdır. İstenen renkteki ürün bütçeyi en fazla %25 aşıyorsa elenmez, uyarıyla gösterilir. Rengi bilinmeyen ürünler önerilmez.
7. **Paket önerisi:** Apriori kurallarından, seçilen kategoriyle birlikte alınan diğer kategoriler lift değerine göre listelenir.
8. **Klasik istatistik** (`istatistik.py`): ay etkili doğrusal ciro regresyonu, segmentlerin harcama farkı için ANOVA ve Kruskal-Wallis, segment ile bölge ilişkisi için ki-kare ve Cramér's V.
9. **Tekrar gelme modeli** (`tekrar_gelme.py`): verinin son 180 günü tahmin edilir. Özellikler yalnızca kesim tarihinden önceki alışverişlerden hesaplanır (RFM, ortalama fatura tutarı, müşteri süresi, indirim, kategori payları), etiket kesimden sonra geri gelip gelmediğidir. Lojistik regresyon ve rastgele orman, yalnızca son alışveriş zamanını bilen basit bir referansla ROC-AUC ve PR-AUC üzerinden karşılaştırılır.
10. **Güven aralıkları** (`evaluate.py`): hit_rate değerleri için %95 bootstrap güven aralığı ve yöntemler arasındaki farklar için eşleştirilmiş bootstrap testi hesaplanır; böylece iki yöntemin farkının tesadüf olup olmadığı görülür.

## Sonuçlar

Segmentasyon: k = 6, silhouette 0,243.

Öneri kalitesi (tarihin %80'i ile eğitim, kalanı test; hit_rate@5, parantez içinde %95 bootstrap güven aralığı):

| Durum | Yöntem | hit_rate@5 |
|---|---|---|
| Mevcut müşteri (2.000 örnek) | Genel popülerlik | 0,406 (0,385–0,427) |
| | Segment popülerliği | 0,394 (0,373–0,417) |
| | Benzer ürün | 0,419 (0,397–0,441) |
| | Hibrit | 0,417 (0,394–0,439) |
| Yeni müşteri (2.707 örnek) | Kategori popülerliği | 0,422 (0,403–0,440) |
| | Segment tabanlı | 0,411 (0,392–0,430) |

- Mevcut müşteride hibrit, genel popülerliği 1,1 puan geçiyor ama fark anlamlı değil (güven aralığı −0,2 ile +2,5). Hibrit, segment popülerliğini 2,3 puan geçiyor ve bu fark anlamlı.
- Yeni müşteride segment tabanlı yöntem, yalnızca kategori popülerliğinin 1,1 puan gerisinde kaldı ve bu küçük fark anlamlı (−1,8 ile −0,4). Yani segment bilgisi bu ölçümde isabeti artırmadı.
- Daha önceki bir ölçümde segment tabanlı yöntem 0,434 ile 0,386 gibi bir üstünlük göstermişti. Rengi bilinmeyen ürünleri öneri kataloğundan çıkardıktan sonra ölçümü yenilediğimizde bu üstünlük kayboldu; değişikliğin tek nedeninin bu olduğunu ayrıca test etmedik. Güncel ölçüm esas alınmalıdır.

Tekrar gelme modeli (kesim 2022-07-04, sonraki 180 gün, geri gelen müşteri oranı %8,6; test: 60.000 müşteri):

| Model | ROC-AUC | PR-AUC | Kesinlik | Duyarlılık |
|---|---|---|---|---|
| Referans (yalnızca son alışveriş) | 0,735 | 0,198 | 0,148 | 0,774 |
| Lojistik regresyon | 0,790 | 0,346 | 0,211 | 0,691 |
| Rastgele orman | 0,790 | 0,371 | 0,271 | 0,592 |

İki model de referanstan iyi. En önemli özellikler son alışverişten bu yana geçen süre, toplam harcama ve fatura sayısı. Kesinlik düşük (rastgele orman "geri gelir" dediklerinin yaklaşık dörtte birini doğru buluyor), çünkü geri gelenler azınlıkta.

Klasik istatistik:
- Ciro regresyonu (59 ay, 2020 Nisan ayında veri yok): R² 0,537; ciro ayda yaklaşık 1,95 milyon birim artıyor (p = 0,0001). Son 6 ayı tahmin hatası %26,3 (MAPE), yani tahmin gücü sınırlı.
- ANOVA / Kruskal-Wallis: segmentler harcamada belirgin ayrışıyor (eta² = 0,56). Segmentler harcamayla kurulduğu için bu beklenen bir sonuç.
- Ki-kare: segment ile bölge arasında ilişki var ama küçük (Cramér's V = 0,125).

## Testler

`python -m pytest` ile çalışan 40 otomatik test var:

| Dosya | Kapsam |
|---|---|
| `test_data_loader.py` | Temizleme, eksik sütun hatası, çok yıllı dosya birleştirme, kategori/renk/beden çıkarma |
| `test_association.py` | Support, confidence ve lift değerleri elle hesaplananlarla eşleşiyor |
| `test_recommender.py` | Segment atama, sahip olunan ürünün önerilmemesi, yeni müşteriye her zaman öneri, filtreler ve kriter gevşetme, rengi bilinmeyen ürünlerin elenmesi, bütçe toleransı, paket önerisi, Türkçe ayrıştırma, anahtarsız yedek mod |
| `test_evaluate_and_bot.py` | Zaman ayırmada sızıntı yok, metrikler, bootstrap karşılaştırması, botun uçtan uca çalışması, çıkış kısayolları |
| `test_tekrar_gelme.py` | Özellikler yalnızca geçmişten, etiket yalnızca gelecekten gelir; model öğrenilebilir bir kalıpta referanstan iyi; tek sınıf durumunda uyarı |
| `test_gorsel.py` | Segment grafikleri üretiliyor |
| `test_cache.py` | Model önbelleği |
| `test_istatistik.py` | Regresyon eğimi ve mevsim etkisi, eksik ay, ANOVA ve ki-kare çıktıları |

## Sınırlamalar

- Ürün kodu ön eklerinin bir kısmı tahmini eşlemedir (yukarıda belirtildi); tişört, kazak ve şort ile ilgili sonuçlar bundan etkilenebilir.
- Yeni müşteri değerlendirmesinde müşterinin tercihi olarak ilk faturasındaki kategoriler kullanıldı; gerçek bir müşteri isteğini daha dağınık ifade eder.
- Segmentler zevkten çok alışveriş sıklığı ve harcama düzeyine göre ayrışıyor; iki segment çok küçük.
- Ciro tahmini yalnızca trend ve ay etkisini içerir, kampanya gibi etkenleri bilmez.
- Tekrar gelme modeli tek bir kesim tarihiyle denendi; kampanya ve mevsim bilgisi içermiyor.
- LLM bileşeni gerçek API anahtarıyla sistematik olarak test edilmedi; kural tabanlı yedek mod test edildi.

## Klasör yapısı

```
config.py              ayarlar ve sabitler (kategori listesi, ürün kodu eşlemesi)
main.py                komutlar: generate-data, analyze, evaluate, stats, predict, chat
inspect_unmapped.py    kategorisi belirlenemeyen ürün kodlarını inceleme aracı
src/                   data_loader, features, segmentation, association, recommender,
                       llm, bot, ui, pipeline, evaluate, istatistik, tekrar_gelme, gorsel, cache, synthetic_data
tests/                 pytest testleri
data/raw/              CSV dosyaları (git'e girmez)
outputs/               grafikler ve tablolar (git'e girmez)
```