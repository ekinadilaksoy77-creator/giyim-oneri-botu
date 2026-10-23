from src import gorsel


def test_all_plots_are_created(artifacts, tmp_path):
    info = gorsel.all_plots(artifacts.features, artifacts.seg, tmp_path)
    for name in ["segment_pca.png", "segment_kategori_isi_haritasi.png", "segment_buyuklukleri.png"]:
        f = tmp_path / name
        assert f.exists() and f.stat().st_size > 5000
    assert 0 < info["pca_aciklanan_varyans"] <= 1
