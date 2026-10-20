# Giyim Mağazası Müşteri–Ürün Eşleştirme Botu

İş Zekası ve Veri Madenciliği dersi dönem projesi.

Bir erkek giyim mağazasının 2018–2022 satış verisini analiz eden ve **yeni müşterilere ürün öneren** bir bot geliştirdik. Bot, müşterinin Türkçe yazdığı isteği anlar (ör. "lacivert gömlek, M beden, bütçem 1500"), mevcut müşterilerin alışverişlerinden öğrendiği kalıplara göre ürün önerir ve nedenini açıklar.

Projede iki bileşen birlikte çalışır:

| Bileşen | Görevi | Dosyalar |
|---|---|---|
| İstatistik ve veri madenciliği | Müşteri segmentasyonu (RFM + k-means), birliktelik kuralları (Apriori), ürün benzerliği (kosinüs), aylık ciro regresyonu, ANOVA / Kruskal-Wallis, ki-kare, öneri değerlendirmesi | `src/segmentation.py`, `src/association.py`, `src/recommender.py`, `src/istatistik.py`, `src/evaluate.py` |
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
python main.py analyze        # temizleme raporu, grafikler, segmentler, Apriori kuralları -> outputs/
python main.py evaluate       # öneri kalitesi (zamana göre ayırma) -> outputs/degerlendirme.json
python main.py stats          # ciro regresyonu, ANOVA, ki-kare -> outputs/
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

## Sonuçlar

Segmentasyon: k = 6, silhouette 0,243.

Öneri kalitesi (tarihin %80'i ile eğitim, kalanı test; hit_rate@5):

| Durum | Yöntem | hit_rate@5 |
|---|---|---|
| Mevcut müşteri | Genel popülerlik | 0,377 |
| | Benzer ürün | 0,383 |
| | Hibrit | 0,379 |
| Yeni müşteri | Kategori popülerliği | 0,386 |
| | Segment tabanlı | **0,434** |

Yeni müşteride segment bilgisi öneri isabetini artırıyor. Mevcut müşteride yöntemler arasındaki fark hata payı içinde kalıyor; bu veride popüler ürünler güçlü bir referans.

Klasik istatistik:
- Ciro regresyonu (59 ay, 2020 Nisan ayında veri yok): R² 0,537; ciro ayda yaklaşık 1,95 milyon birim artıyor (p = 0,0001). Son 6 ayı tahmin hatası %26,3 (MAPE), yani tahmin gücü sınırlı.
- ANOVA / Kruskal-Wallis: segmentler harcamada belirgin ayrışıyor (eta² = 0,56). Segmentler harcamayla kurulduğu için bu beklenen bir sonuç.
- Ki-kare: segment ile bölge arasında ilişki var ama küçük (Cramér's V = 0,125).

## Testler

`python -m pytest` ile çalışan 35 otomatik test var:

| Dosya | Kapsam |
|---|---|
| `test_data_loader.py` | Temizleme, eksik sütun hatası, çok yıllı dosya birleştirme, kategori/renk/beden çıkarma |
| `test_association.py` | Support, confidence ve lift değerleri elle hesaplananlarla eşleşiyor |
| `test_recommender.py` | Segment atama, sahip olunan ürünün önerilmemesi, yeni müşteriye her zaman öneri, filtreler ve kriter gevşetme, rengi bilinmeyen ürünlerin elenmesi, bütçe toleransı, paket önerisi, Türkçe ayrıştırma, anahtarsız yedek mod |
| `test_evaluate_and_bot.py` | Zaman ayırmada sızıntı yok, metrikler, botun uçtan uca çalışması, çıkış kısayolları |
| `test_cache.py` | Model önbelleği |
| `test_istatistik.py` | Regresyon eğimi ve mevsim etkisi, eksik ay, ANOVA ve ki-kare çıktıları |

## Sınırlamalar

- Ürün kodu ön eklerinin bir kısmı tahmini eşlemedir (yukarıda belirtildi); tişört, kazak ve şort ile ilgili sonuçlar bundan etkilenebilir.
- Yeni müşteri değerlendirmesinde müşterinin tercihi olarak ilk faturasındaki kategoriler kullanıldı; gerçek bir müşteri isteğini daha dağınık ifade eder.
- Segmentler zevkten çok alışveriş sıklığı ve harcama düzeyine göre ayrışıyor; iki segment çok küçük.
- Ciro tahmini yalnızca trend ve ay etkisini içerir, kampanya gibi etkenleri bilmez.
- LLM bileşeni gerçek API anahtarıyla sistematik olarak test edilmedi; kural tabanlı yedek mod test edildi.

## Klasör yapısı

```
config.py              ayarlar ve sabitler (kategori listesi, ürün kodu eşlemesi)
main.py                komutlar: generate-data, analyze, evaluate, stats, chat
inspect_unmapped.py    kategorisi belirlenemeyen ürün kodlarını inceleme aracı
src/                   data_loader, features, segmentation, association, recommender,
                       llm, bot, ui, pipeline, evaluate, istatistik, cache, synthetic_data
tests/                 pytest testleri
data/raw/              CSV dosyaları (git'e girmez)
outputs/               grafikler ve tablolar (git'e girmez)
```