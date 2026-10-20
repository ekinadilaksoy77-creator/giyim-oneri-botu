# Giyim Mağazası: Müşteri–Ürün Eşleştirme Botu

**Ders:** İş Zekası ve Veri Madenciliği

Mevcut müşterilerin alışveriş verisinden öğrenip **yeni müşterilere ürün öneren** bir bot.
Projede hocanın istediği iki parça birlikte çalışır:

| Parça | Ne yapar? | Nerede? |
|---|---|---|
| **İstatistik / veri madenciliği** | Müşteri segmentasyonu (RFM + k-means), birliktelik kuralları (Apriori), ürün benzerliği (kosinüs), öneri skorları, değerlendirme metrikleri | `src/segmentation.py`, `src/association.py`, `src/recommender.py`, `src/evaluate.py` |
| **LLM** | Müşterinin Türkçe mesajından tercihi çıkarır (renk, kategori, beden, bütçe) ve önerileri Türkçe açıklar | `src/llm.py` |

> **Tasarım kararı:** Ürünleri LLM **seçmez**. Ürünleri sayısal modeller seçer; LLM yalnızca
> dili anlar ve sonucu anlatır. Böylece LLM'in uydurma (halüsinasyon) riski düşer ve öneriler ölçülebilir kalır.

## Veri seti

[Clothes Retail Store (Kaggle)](https://www.kaggle.com/datasets/nishchay331/retail-store): kurgusal bir **erkek giyim**
mağaza zinciri, 2022 yılı satışları. Her satır bir fatura (`bill_id`) içindeki tek ürün kalemidir.

Beklenen sütunlar: `user_id, bill_id, line_item_amount, bill_discount, transaction_date, description,
inventory_category, colour, size, zone_name, store_name, year`

**İndirme (Kaggle hesabı gerekir):**

```bash
pip install kaggle
# Kaggle > Settings > API > "Create New Token" ile kaggle.json'u indir, ~/.kaggle/ içine koy
kaggle datasets download -d nishchay331/retail-store -p data/raw --unzip
```

Ya da siteden elle indirip CSV'leri `data/raw/` klasörüne koy. Birden fazla yıl varsa (2019, 2020, ...) **hepsini** koyabilirsin: kod klasördeki tüm CSV'leri okuyup birleştirir (sütunlar aynı olmalı). Birleştirme, aynı müşterinin yıllar içindeki tekrar alışverişlerini görmemizi sağlar.

`python main.py analyze` çalışınca `outputs/temizleme_raporu.json` içinde **kategorisi belirlenemeyip atılan** en sık ürün türleri de listelenir; önemli bir ürün grubu (örn. kravat) orada görünüyorsa `src/data_loader.py` içindeki `CATEGORY_KEYWORDS` listesine ekle.

**Bilmen gerekenler**
- Bu veri setinde "takım elbise" kategorisi **yok**; gömlek, eşofman altı, ceket vb. var. Rapordaki işletme tanımını buna göre "erkek giyim mağazası" yap.
- Tutarların para birimi veri setinde belirtilmiyor; metinlerde "birim" diyoruz, TL varsayma.
- Çoğu müşteri yalnızca bir kez alışveriş yapmış. Bu yüzden "yeni müşteri / soğuk başlangıç" yaklaşımı bu veriye çok uygun, ama kişiselleştirme gücü sınırlı kalabilir.
- Ham CSV'nin kategori/renk/beden sütunlarında eksik ve dağınık değerler olabilir; `src/data_loader.py` bunları temizler ve kaç satır attığını raporlar.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # LLM kullanacaksan ANTHROPIC_API_KEY'i buraya yaz
```

## Çalıştırma (adım adım)

```bash
python main.py generate-data   # (İsteğe bağlı) gerçek veri yoksa sentetik deneme verisi üretir
python main.py analyze         # temizleme raporu, grafikler, segmentler, Apriori kuralları -> outputs/
python main.py evaluate        # öneri kalitesi (zamana göre ayırma) -> outputs/degerlendirme.json
python main.py chat            # eşleştirme botu
python -m pytest               # testler
```

`data/raw/` içinde gerçek CSV varsa o kullanılır; yoksa `sentetik_...csv`. Sentetik veri **uydurmadır**, sadece kodu denemek içindir:
rapordaki sonuçları gerçek veriyle üret.

**Bota yazabileceklerin**

```
İş için lacivert bir gömlek arıyorum, M beden, bütçem 1500     -> yeni müşteri önerisi
müşteri U00012                                                  -> kayıtlı müşteri önerisi (kendi ID'nle dene)
paket gömlek                                                    -> gömlekle birlikte alınan ürünler
çık
```

`ANTHROPIC_API_KEY` yoksa bot **kural tabanlı yedek** ile çalışır (anahtar kelimelerle tercih çıkarır, şablonla açıklar).
Anahtar varsa LLM devreye girer (varsayılan model `claude-sonnet-5-5`, `.env` içinde `LLM_MODEL` ile değiştirilir).

## Nasıl çalışıyor?

```
CSV ──► temizleme ──► müşteri özellikleri (RFM + kategori payları)
                         │
                         ├─► k-means segmentleri ──────────────┐
                         ├─► Apriori kuralları (paket önerisi) │
                         └─► müşteri×ürün matrisi ─► ürün-ürün benzerliği
                                                               ▼
                                              Hibrit öneri motoru
                                                               ▲
Müşteri mesajı ─► LLM: tercih çıkar ─► Preferences ────────────┘
                                                               │
                                       öneri + nedenler ─► LLM: Türkçe açıklama
```

1. **Temizleme** (`data_loader.py`): geçersiz tutarları atar, eksik kategoriyi `description` metninden türetir, eksik renk/bedeni `NOT CAPTURED` yapar. "Ürün" = kategori + renk (örn. `SHIRTS | NAVY`).
2. **Özellikler** (`features.py`): her müşteri için *recency* (son alışverişten bu yana gün), *frequency* (fatura sayısı), *monetary* (toplam harcama) ve harcamanın kategorilere dağılımı.
3. **Segmentasyon** (`segmentation.py`): özellikler standartlaştırılır, k-means çalışır; k değeri **silhouette skoru** ile seçilir (3–6 arası).
4. **Birliktelik kuralları** (`association.py`): Apriori elle yazıldı. *support, confidence, lift* hesaplanır.
5. **Mevcut müşteri**: `skor = 0.6 × benzer ürünler + 0.4 × kendi segmentindeki popülerlik`; zaten aldığı ürünler önerilmez.
6. **Yeni müşteri (soğuk başlangıç)**: söylediği kategorilere en çok benzeyen segment bulunur (kosinüs benzerliği), o segmentte popüler ürünler sıralanır.
   - Kategori, beden ve bütçe **filtredir**; çok az ürün kalırsa, darboğaz olan kriter gevşetilir ve bu müşteriye **söylenir**.
   - Renk **skor artırıcıdır** (uyan ürünler en üste çıkar).
7. **Değerlendirme** (`evaluate.py`): tarihin %80'ine kadar eğit, sonrasını tahmin et. *hit_rate@5* ve *precision@5*; popülerlik, segment popülerliği, benzer ürün ve hibrit yöntem karşılaştırılır.

## Test case'ler

`tests/` klasöründeki testler, analiz dokümanındaki kabul kriterlerinin bir kısmını otomatik doğrular:

| Test | Doğruladığı kriter |
|---|---|
| `test_data_loader.py` | Temizleme sonrası geçersiz tutar/eksik değer kalmaz; eksik sütunda anlaşılır hata |
| `test_association.py` | Support/confidence/lift elle hesaplanan değerlerle eşleşir |
| `test_recommender.py` | Her müşteri tek segmente atanır; mevcut müşteriye sahip olduğu ürün önerilmez; yeni müşteriye **her zaman** öneri döner; öneriler yalnızca katalogdan gelir; bütçe/kategori kısıtı uygulanır; Türkçe mesaj doğru ayrıştırılır; anahtarsız yedek mod çalışır |
| `test_evaluate_and_bot.py` | Zaman ayırmada sızıntı yok; metrikler hesaplanır; bot uçtan uca çalışır |

## Raporda tartışabileceğin sınırlamalar

- Sentetik veride yapılan denemeler yalnızca kodun çalıştığını gösterir; bilimsel sonuç değildir.
- Yeni müşteri için segment tahmini, yalnızca söylenen kategorilere dayanır (kaba bir tahmindir). Gerçek veride segment tabanlı yöntemin "kategori popülerliği" yöntemini geçip geçmediğini `evaluate` ile ölç; geçmiyorsa bunu dürüstçe yaz.
- Çoğu müşterinin tek alışverişi olduğu için "mevcut müşteri" değerlendirmesinde yeterli örnek çıkmayabilir.
- LLM yanılabilir ve API maliyeti vardır; bu yüzden ürün seçimini LLM'e bırakmadık.

## Klasör yapısı

```
config.py            ayarlar ve sabitler
main.py              komut satırı (generate-data / analyze / evaluate / chat)
src/                 data_loader, synthetic_data, features, segmentation, association,
                     recommender, llm, bot, pipeline, evaluate
tests/               pytest testleri
data/raw/            CSV buraya (git'e girmez)
outputs/             grafikler ve tablolar (git'e girmez)
```
