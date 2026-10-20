"""Komut satırı girişi.

  python main.py generate-data     # (isteğe bağlı) sentetik deneme verisi üret
  python main.py analyze           # temizleme raporu, EDA grafikleri, segmentler, kurallar
  python main.py evaluate          # öneri kalitesini ölç (zamana göre ayırma)
  python main.py stats             # klasik istatistik: ciro regresyonu, ANOVA, ki-kare
  python main.py chat              # eşleştirme botunu başlat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from dotenv import load_dotenv

load_dotenv()  # .env içindeki ANTHROPIC_API_KEY'i ortama yükler
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from src import data_loader, synthetic_data  # noqa: E402
from src.pipeline import build  # noqa: E402


def _load(path):
    df, report = data_loader.load_clean(Path(path) if path else None)
    print(f"Veri: {report['dosya']} | temiz satır: {report['temiz_satir']:,} | müşteri: {report['musteri_sayisi']:,}")
    if report["dosya"].startswith("sentetik"):
        print("UYARI: SENTETİK veri kullanılıyor. Raporda gerçek (Kaggle) veri sonuçlarını kullan.")
    return df, report


def cmd_generate(args):
    path, df = synthetic_data.write_csv(n_users=args.users)
    print(f"{len(df):,} satır yazıldı -> {path}")


def cmd_analyze(args):
    df, report = _load(args.data)
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "temizleme_raporu.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTemizleme raporu:", json.dumps(report, ensure_ascii=False, indent=2))

    # --- Keşifsel analiz (betimleyici istatistik) ---
    by_cat = df.groupby("category")["amount"].agg(["count", "sum", "mean", "median"]).sort_values("sum", ascending=False)
    by_cat.round(2).to_csv(out / "kategori_ozeti.csv", encoding="utf-8-sig")
    monthly = df.groupby(df["transaction_date"].dt.to_period("M"))["amount"].sum()
    monthly.to_csv(out / "aylik_ciro.csv", encoding="utf-8-sig")
    print("\nKategori özeti (ciroya göre):\n", by_cat.round(1).head(20))

    fig, ax = plt.subplots(figsize=(8, 4))
    by_cat["sum"].plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_title("Kategoriye göre toplam ciro")
    fig.tight_layout()
    fig.savefig(out / "kategori_ciro.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(ax=ax, marker="o")
    ax.set_title("Aylık ciro")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out / "aylik_ciro.png", dpi=130)
    plt.close(fig)

    # --- Segmentasyon + kurallar ---
    art = build(df, min_support=args.min_support)
    seg = art.seg
    seg.profiles.to_csv(out / "segmentler.csv", encoding="utf-8-sig")
    art.rules.to_csv(out / "birliktelik_kurallari.csv", index=False, encoding="utf-8-sig")
    print(f"\nSeçilen küme sayısı k={seg.k}  (silhouette: {seg.silhouette_by_k or 'k elle verildi'})")
    print(seg.profiles[["musteri_sayisi", "recency_days", "frequency", "monetary", "isim"]].to_string())
    print("\nEn güçlü 15 birliktelik kuralı:")
    print(art.rules.head(15).to_string() if len(art.rules) else "Kural bulunamadı (min_support'u düşür).")
    print(f"\nÇıktılar: {out}")


def cmd_evaluate(args):
    from src.evaluate import evaluate
    df, _ = _load(args.data)
    res = evaluate(df, k=args.k)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "degerlendirme.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_stats(args):
    from src import cache, istatistik as ist
    from src.features import build_customer_features
    df, _ = _load(args.data)
    paths = [Path(args.data)] if args.data else data_loader.find_data_files()
    sig = cache.signature(paths)
    art = cache.load(sig)
    if art is None:
        print("Segmentler hazırlanıyor (kayıtlı model yok)...")
        art = build(df)
        cache.save(sig, art)
    labels = art.seg.labels
    features = build_customer_features(df)
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # --- 1) Aylık ciro regresyonu ---
    r = ist.revenue_regression(df)
    o = r["ozet"]
    r["katsayilar"].round(4).to_csv(out / "regresyon_katsayilari.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(r["gercek"].index.to_timestamp(), r["gercek"].values, marker="o", label="Gerçek ciro")
    ax.plot(r["uyum"].index.to_timestamp(), r["uyum"].values, "--", label="Model (trend + mevsim)")
    ax.plot(r["tahmin"].index.to_timestamp(), r["tahmin"].values, ":", marker="x", color="red", label="İleri tahmin")
    ax.set_title("Aylık ciro: doğrusal regresyon"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "ciro_regresyon.png", dpi=130); plt.close(fig)
    print("\n=== 1) Aylık ciro regresyonu (trend + mevsimsellik) ===")
    if o["eksik_aylar"]:
        print(f"UYARI: veride şu aylar yok (ciro satırı yok), regresyonda atlandı: {', '.join(o['eksik_aylar'])}")
    print(f"Ay sayısı: {o['n_ay']} | R²: {o['r2']:.3f} (düzeltilmiş: {o['duzeltilmis_r2']:.3f})")
    yon = "artıyor" if o["trend_katsayisi_aylik"] > 0 else "azalıyor"
    anlam = "anlamlı" if o["trend_p"] < 0.05 else "anlamlı DEĞİL"
    print(f"Trend: ciro ayda ortalama {o['trend_katsayisi_aylik']:,.0f} birim {yon} (p={o['trend_p']:.4f}, {anlam}).")
    if o.get("holdout_ay"):
        print(f"Son {o['holdout_ay']} ayı saklayıp tahmin ettik: ortalama yüzde hata (MAPE) = %{o['holdout_mape'] * 100:.1f}")
    print("İleri tahmin (sonraki 6 ay):"); print(r["tahmin"].round(0).to_string())

    # --- 2) Segmentler arası harcama farkı ---
    a = ist.segment_spend_tests(features, labels)
    print("\n=== 2) Segmentler harcamada farklı mı? (ANOVA + Kruskal-Wallis) ===")
    print(f"ANOVA F={a['anova_F']:,.1f} (p={a['anova_p']:.3g}) | Kruskal-Wallis H={a['kruskal_H']:,.1f} (p={a['kruskal_p']:.3g})")
    print(f"Etki büyüklüğü eta²={a['eta_kare']:.3f} -> {ist.yorum_etki(a['eta_kare'], 'eta')}")
    print("Not: segmentler harcama bilgisiyle kurulduğu için fark beklenir; asıl bilgi etki büyüklüğüdür.")

    # --- 3) Segment x bölge ---
    z = ist.segment_zone_test(df, labels)
    print("\n=== 3) Segment ile bölge ilişkili mi? (ki-kare bağımsızlık testi) ===")
    if "not" in z:
        print(z["not"])
    else:
        z["tablo"].to_csv(out / "segment_bolge_tablosu.csv", encoding="utf-8-sig")
        print(f"χ²={z['ki_kare']:,.1f}, sd={z['serbestlik']}, p={z['p']:.3g} | Cramér's V={z['cramer_v']:.3f} -> {ist.yorum_etki(z['cramer_v'], 'v')}")
        z = {k: v for k, v in z.items() if k != "tablo"}
    summary = {"regresyon": o, "anova": a, "ki_kare": z}
    (out / "istatistik_ozeti.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nÇıktılar: {out}")


def cmd_chat(args):
    from src import cache
    from src.bot import StyleBot
    paths = [Path(args.data)] if args.data else data_loader.find_data_files()
    sig = cache.signature(paths)
    art = None if args.rebuild else cache.load(sig)
    if art is not None:
        print("Kayıtlı model yüklendi (veri değişmediği için yeniden eğitilmedi; zorlamak için --rebuild).")
    else:
        df, _ = _load(args.data)
        print("Model hazırlanıyor (segmentasyon + öneri motoru); büyük veride birkaç dakika sürebilir...")
        art = build(df)
        cache.save(sig, art)
    StyleBot(art).run()


def main():
    p = argparse.ArgumentParser(description="Giyim mağazası müşteri-ürün eşleştirme projesi")
    p.add_argument("--data", help="Kullanılacak CSV yolu (verilmezse data/raw içinden seçilir)")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-data"); g.add_argument("--users", type=int, default=6000); g.set_defaults(fn=cmd_generate)
    a = sub.add_parser("analyze"); a.add_argument("--min-support", type=float, default=0.002); a.set_defaults(fn=cmd_analyze)
    e = sub.add_parser("evaluate"); e.add_argument("-k", type=int, default=config.DEFAULT_TOP_K); e.set_defaults(fn=cmd_evaluate)
    st = sub.add_parser("stats"); st.set_defaults(fn=cmd_stats)
    c = sub.add_parser("chat"); c.add_argument("--rebuild", action="store_true", help="Kayıtlı modeli yok say, yeniden eğit"); c.set_defaults(fn=cmd_chat)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()