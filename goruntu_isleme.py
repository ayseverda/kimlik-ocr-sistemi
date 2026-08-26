import cv2
import numpy as np
import os
import time


# =========================================================
# SABİTLER
# =========================================================

# SIFT maliyeti piksel sayisiyla hizla buyur. Kimlik tespiti icin 1300 uzun
# kenar, en kucuk referansin ayrintilarini korurken 1800'e gore islenecek alani
# yaklasik %48 azaltir. OCR her zaman asil/yuksek cozunurluklu warp'ta yapilir;
# bu sinir yalnizca belge tipi ve geometri tespitini etkiler.
MAX_CALISMA_BOYUTU = 1300
SIFT_AZAMI_OZELLIK = 3200

# Referans adlari calisma dizinine degil, bu modulun bulundugu dizine gore
# cozulur. Boylece uygulama farkli bir dizinden baslatildiginda da ayni
# dosyalar kullanilir.
MODUL_DIZINI = os.path.dirname(os.path.abspath(__file__))

BELGE_REFERANSLARI = {
    "tc": "referans_kimlik.jpg",
    "gocmen": "referans_gocmen.jpg",
    "eski_tc": "referans_eski_tc.jpg",
}


_DETECTOR = None
_DETECTOR_TIPI = None
_CLAHE = None
_FLANN = None
_REFERANS_CACHE = {}


def referans_yolunu_coz(yol):
    """Mutlak olmayan referans yollarini modul dizinine gore cozer."""
    yol = os.fspath(yol)
    if not os.path.isabs(yol):
        yol = os.path.join(MODUL_DIZINI, yol)
    return os.path.abspath(yol)


# =========================================================
# DETECTOR / CLAHE / MATCHER
# =========================================================

def detector_getir():
    global _DETECTOR, _DETECTOR_TIPI

    if _DETECTOR is not None:
        return _DETECTOR, _DETECTOR_TIPI

    if hasattr(cv2, "SIFT_create"):
        _DETECTOR = cv2.SIFT_create(
            nfeatures=SIFT_AZAMI_OZELLIK,
            contrastThreshold=0.018,
            edgeThreshold=12,
            sigma=1.6,
        )
        _DETECTOR_TIPI = "SIFT"
    else:
        _DETECTOR = cv2.ORB_create(nfeatures=7000, scaleFactor=1.2, nlevels=10, fastThreshold=5)
        _DETECTOR_TIPI = "ORB"

    return _DETECTOR, _DETECTOR_TIPI


def clahe_getir():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def matcher_getir(detector_tipi):
    global _FLANN

    if detector_tipi == "SIFT":
        if _FLANN is None:
            # 3200 ozellikte 4 agac/40 kontrol, belge siniflandirmasi icin
            # yeterli komsuluk kalitesini korurken onceki 5/60 taramasindan
            # daha az descriptor karsilastirir.
            _FLANN = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 40})
        return _FLANN

    return cv2.BFMatcher(cv2.NORM_HAMMING)


def sayfa_matcher_hazirla(des_resim, detector_tipi):
    """Ayni sayfanin train indeksini uc referans icin yalnizca bir kez kurar."""
    if des_resim is None:
        return None
    try:
        if detector_tipi == "SIFT":
            matcher = cv2.FlannBasedMatcher(
                {"algorithm": 1, "trees": 4}, {"checks": 40}
            )
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matcher.add([des_resim])
        matcher.train()
        return matcher
    except Exception:
        # Eski OpenCV derlemelerinde dogrudan knnMatch yoluna geri donulur.
        return None


# =========================================================
# REFERANS
# =========================================================

def referansi_hazirla(belge_tipi, yol):
    """Her referans türü kendi cache girdisinde tutulur; dosya değişmediği
    sürece yeniden okunmaz/yeniden feature çıkarılmaz."""
    global _REFERANS_CACHE

    try:
        abs_yol = referans_yolunu_coz(yol)
    except (TypeError, ValueError, OSError) as exc:
        return None, None, None, f"Geçersiz referans yolu ({belge_tipi}): {exc}"

    if not os.path.isfile(abs_yol):
        return None, None, None, f"Referans bulunamadı ({belge_tipi}): {abs_yol}"

    try:
        mtime = os.path.getmtime(abs_yol)
    except OSError as exc:
        return None, None, None, f"Referans bilgisi okunamadı ({belge_tipi}): {abs_yol} ({exc})"
    cache = _REFERANS_CACHE.get(belge_tipi)

    if cache is not None and cache.get("yol") == abs_yol and cache.get("mtime") == mtime:
        return cache["resim"], cache["kp"], cache["des"], None

    try:
        resim = cv2.imread(abs_yol)
    except Exception as exc:
        return None, None, None, f"Referans resmi okunamadı ({belge_tipi}): {abs_yol} ({exc})"
    if resim is None:
        return None, None, None, f"Referans resmi okunamadı ({belge_tipi}): {abs_yol}"

    try:
        gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
        detector, _ = detector_getir()
        kp, des = detector.detectAndCompute(gri, None)
    except Exception as exc:
        return None, None, None, f"Referans özellikleri çıkarılamadı ({belge_tipi}): {abs_yol} ({exc})"

    if des is None:
        return None, None, None, f"Referansta özellik bulunamadı ({belge_tipi}): {abs_yol}"

    _REFERANS_CACHE[belge_tipi] = {"yol": abs_yol, "mtime": mtime, "resim": resim, "kp": kp, "des": des}
    return resim, kp, des, None


# =========================================================
# ÇALIŞMA GÖRÜNTÜSÜNÜN DESCRIPTOR'LARI (SAYFA BAŞINA 1 KEZ)
# =========================================================
#
# ÖNEMLİ HIZ DÜZELTMESİ: eskiden her sayfa için 3 referans (tc, göçmen,
# eski_tc) sırayla test ediliyordu ve her testte 'calisma' görüntüsünün SIFT
# keypoint/descriptor'ları BAŞTAN çıkarılıyordu — yani aynı fotoğraf için
# detectAndCompute() 3 kere çalışıyordu. Bu, sayfa başına en pahalı adımın
# gereksiz yere 3 katına çıkması demekti. Descriptor'lar yalnızca görüntüye
# bağlı olduğu için (hangi referansla karşılaştırıldığından bağımsız),
# sentetik veriyle doğrulandığı gibi 1 kez çıkarılıp 3 referansa karşı da
# kullanılması BİREBİR AYNI sonucu ~2.5x daha hızlı verir.

def resim_descriptor_cikar(resim):
    detector, detector_tipi = detector_getir()
    gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
    kp, des = detector.detectAndCompute(gri, None)
    return kp, des, detector_tipi


def referansla_eslestir(
    kp_ref, des_ref, kp_resim, des_resim, detector_tipi, hazir_matcher=None
):
    if des_resim is None:
        return None, [], 0

    matcher = hazir_matcher if hazir_matcher is not None else matcher_getir(detector_tipi)

    try:
        if hazir_matcher is not None:
            raw = matcher.knnMatch(des_ref, k=2)
        else:
            raw = matcher.knnMatch(des_ref, des_resim, k=2)
    except cv2.error:
        return None, [], 0

    ratio = 0.74 if detector_tipi == "SIFT" else 0.80
    iyi = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]

    if len(iyi) < 8:
        return None, iyi, 0

    src = np.float32([kp_ref[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2)
    dst = np.float32([kp_resim[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, iyi, 0

    return H, iyi, int(mask.sum())


# =========================================================
# SKOR
# =========================================================

def eslesme_skoru(iyi, inlier):
    if iyi <= 0:
        return 0.0
    oran = inlier / iyi
    return inlier * 3.5 + iyi * 0.30 + oran * 100


# =========================================================
# TEK REFERANS TEST
# =========================================================

def tek_referansi_test_et(
    belge_tipi, referans_yolu, kp_resim, des_resim, detector_tipi,
    hazir_matcher=None,
):
    referans, kp_ref, des_ref, hata = referansi_hazirla(belge_tipi, referans_yolu)
    if hata:
        return {"belge_tipi": belge_tipi, "basarili": False, "skor": 0.0, "hata": hata}

    H, iyi, inlier = referansla_eslestir(
        kp_ref, des_ref, kp_resim, des_resim, detector_tipi, hazir_matcher
    )
    iyi_sayisi = len(iyi)
    inlier_orani = inlier / iyi_sayisi if iyi_sayisi else 0.0
    skor = eslesme_skoru(iyi_sayisi, inlier)

    if belge_tipi == "eski_tc":
        # Yedi rastgele inlier; gocmen formundaki ortak arma, tablo ve Bakanlik
        # yazilarini eski kimlik olarak secmeye yetiyordu. Bu kadar zayif bir
        # aday homografiye sokulmadan elenir; yanlis pozitif yerine kontrollu
        # bir "bulunamadi" sonucu tercih edilir.
        basarili = H is not None and inlier >= 9 and inlier_orani >= 0.18
    elif belge_tipi == "gocmen":
        basarili = H is not None and inlier >= 10 and inlier_orani >= 0.12
    else:
        basarili = H is not None and inlier >= 8

    return {
        "belge_tipi": belge_tipi, "basarili": basarili, "referans": referans,
        "kp_ref": kp_ref, "kp_resim": kp_resim, "H": H, "iyi": iyi,
        "iyi_eslesme": iyi_sayisi, "inlier": inlier, "inlier_orani": inlier_orani,
        "skor": skor, "detector": detector_tipi,
    }


# =========================================================
# BELGE TİPİ
# =========================================================

def belge_tipini_bul(resim):
    h, w = resim.shape[:2]
    olcek = 1.0

    if max(h, w) > MAX_CALISMA_BOYUTU:
        olcek = MAX_CALISMA_BOYUTU / max(h, w)
        calisma = cv2.resize(resim, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    else:
        calisma = resim

    # Sayfa başına TEK sefer: 3 referansın hepsi bu descriptor'ları kullanır.
    kp_resim, des_resim, detector_tipi = resim_descriptor_cikar(calisma)
    hazir_matcher = sayfa_matcher_hazirla(des_resim, detector_tipi)

    sonuclar = {
        tip: tek_referansi_test_et(
            tip, yol, kp_resim, des_resim, detector_tipi, hazir_matcher
        )
        for tip, yol in BELGE_REFERANSLARI.items()
    }

    adaylar = [x for x in sonuclar.values() if x.get("basarili", False)]

    if not adaylar:
        return {
            "basarili": False, "belge_tipi": None, "adaylar": [],
            "sonuclar": sonuclar, "calisma": calisma, "olcek": olcek,
        }

    # Ilk aday daha onceki davranisla ayni kalir. Sirali listenin tutulmasi,
    # yalnizca en iyi aday geometrik olarak kullanilamazsa bir sonraki adayi
    # kontrollu bicimde deneyebilmemizi saglar.
    adaylar = sorted(adaylar, key=lambda x: x["skor"], reverse=True)
    en_iyi = adaylar[0]
    return {
        "basarili": True, "belge_tipi": en_iyi["belge_tipi"], "en_iyi": en_iyi,
        "adaylar": adaylar, "sonuclar": sonuclar, "calisma": calisma, "olcek": olcek,
    }


# =========================================================
# REFERANS KÖŞELERİ
# =========================================================

def koseleri_bul(referans, H):
    h, w = referans.shape[:2]
    # Sıra referans yönünü temsil ediyor: TL -> TR -> BR -> BL. Sonradan sıralanmıyor.
    ref_koseler = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ref_koseler, H).reshape(4, 2)


def _gocmen_affine_koseleri_bul(aday, olcek, hedef_shape=None):
    """Mevcut eslesmelerden sikica dogrulanmis, ucuz bir affine yedegi uretir.

    Bu yol yeni descriptor cikartmaz ve yalnizca gocmen homografisi
    kullanilamadiginda cagrilir. Partial affine; duz tarama, kucuk donme ve
    olcek farklarini duzeltirken hatali bir eslesme kumesinin serbestce
    yamulmasina izin vermez.
    """
    metrikler = {
        "eslesme": 0,
        "inlier": 0,
        "inlier_orani": 0.0,
        "referans_x_yayilimi": 0.0,
        "referans_y_yayilimi": 0.0,
        "hedef_x_yayilimi": 0.0,
        "hedef_y_yayilimi": 0.0,
        "medyan_reprojeksiyon_hatasi": None,
    }

    try:
        olcek = float(olcek)
    except (TypeError, ValueError):
        return None, metrikler, "Affine calisma olcegi gecersiz."
    if not np.isfinite(olcek) or olcek <= 0.0:
        return None, metrikler, "Affine calisma olcegi gecersiz."

    referans = aday.get("referans")
    kp_ref = aday.get("kp_ref")
    kp_resim = aday.get("kp_resim")
    iyi = aday.get("iyi")
    if (
        not isinstance(referans, np.ndarray)
        or referans.ndim < 2
        or kp_ref is None
        or kp_resim is None
        or iyi is None
        or len(iyi) < 8
    ):
        return None, metrikler, "Affine icin yeterli eslesme verisi yok."

    try:
        src = np.float32([kp_ref[m.queryIdx].pt for m in iyi]).reshape(-1, 2)
        dst = np.float32([kp_resim[m.trainIdx].pt for m in iyi]).reshape(-1, 2)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return None, metrikler, f"Affine eslesme noktalari okunamadi: {exc}"

    metrikler["eslesme"] = int(len(src))
    if len(src) < 8 or not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        return None, metrikler, "Affine eslesme noktalari yetersiz veya gecersiz."

    try:
        affine, maske = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=1200,
            confidence=0.995,
            refineIters=10,
        )
    except Exception as exc:
        return None, metrikler, f"Affine model hesaplanamadi: {exc}"

    if affine is None or maske is None:
        return None, metrikler, "Affine model bulunamadi."

    affine = np.asarray(affine, dtype=np.float64)
    maske = np.asarray(maske).reshape(-1).astype(bool)
    if affine.shape != (2, 3) or not np.all(np.isfinite(affine)) or maske.size != len(src):
        return None, metrikler, "Affine model veya inlier maskesi gecersiz."

    inlier = int(np.count_nonzero(maske))
    inlier_orani = float(inlier / max(1, len(src)))
    metrikler["inlier"] = inlier
    metrikler["inlier_orani"] = inlier_orani
    if inlier < 8 or inlier_orani < 0.45:
        return None, metrikler, "Affine inlier destegi guvenilir degil."

    h_ref, w_ref = referans.shape[:2]
    src_inlier = src[maske]
    x_yayilimi = float(np.ptp(src_inlier[:, 0]) / max(1, w_ref - 1))
    y_yayilimi = float(np.ptp(src_inlier[:, 1]) / max(1, h_ref - 1))
    metrikler["referans_x_yayilimi"] = x_yayilimi
    metrikler["referans_y_yayilimi"] = y_yayilimi
    if x_yayilimi < 0.30 or y_yayilimi < 0.30:
        return None, metrikler, "Affine inlier noktalari karta yeterince yayilmiyor."

    if hedef_shape is not None and len(hedef_shape) >= 2:
        hedef_h = float(hedef_shape[0]) * olcek
        hedef_w = float(hedef_shape[1]) * olcek
        dst_inlier = dst[maske]
        hedef_x_yayilimi = float(np.ptp(dst_inlier[:, 0]) / max(1.0, hedef_w - 1.0))
        hedef_y_yayilimi = float(np.ptp(dst_inlier[:, 1]) / max(1.0, hedef_h - 1.0))
        metrikler["hedef_x_yayilimi"] = hedef_x_yayilimi
        metrikler["hedef_y_yayilimi"] = hedef_y_yayilimi
        # Bu oranlar yalnız debug metriğidir. A4 içindeki küçük fakat okunabilir
        # bir kartın inlier'ları doğal olarak tüm sayfanın %12'sinden azına
        # yayılabilir. Güvenlik; referans üzerindeki yayılım, reprojeksiyon,
        # aşağıdaki köşe/alan kontrolleri ve warp sonrası tablo imzasıyla sağlanır.

    dogrusal = affine[:, :2]
    determinant = float(np.linalg.det(dogrusal))
    if not np.isfinite(determinant) or determinant <= 1e-8:
        return None, metrikler, "Affine donusumu ters veya cokmus."

    tahmin = src_inlier @ dogrusal.T + affine[:, 2]
    hatalar = np.linalg.norm(tahmin - dst[maske], axis=1)
    medyan_hata = float(np.median(hatalar))
    metrikler["medyan_reprojeksiyon_hatasi"] = medyan_hata
    if not np.isfinite(medyan_hata) or medyan_hata > 3.5:
        return None, metrikler, "Affine reprojeksiyon hatasi yuksek."

    ref_koseler = np.float32(
        [[0, 0], [w_ref - 1, 0], [w_ref - 1, h_ref - 1], [0, h_ref - 1]]
    )
    koseler = ref_koseler @ dogrusal.T + affine[:, 2]
    return np.asarray(koseler / olcek, dtype=np.float32), metrikler, ""


def _dortgeni_tl_tr_br_bl_sirala(noktalar):
    """Dort noktayi ekran koordinatinda TL, TR, BR, BL sirasina koyar."""
    noktalar = np.asarray(noktalar, dtype=np.float32).reshape(-1, 2)
    if noktalar.shape != (4, 2) or not np.all(np.isfinite(noktalar)):
        return None

    merkez = np.mean(noktalar, axis=0)
    acilar = np.arctan2(noktalar[:, 1] - merkez[1], noktalar[:, 0] - merkez[0])
    sirali = noktalar[np.argsort(acilar)]
    ilk = int(np.argmin(np.sum(sirali, axis=1)))
    sirali = np.roll(sirali, -ilk, axis=0)
    if _cokgen_alani(sirali) <= 1.0:
        return None
    return np.asarray(sirali, dtype=np.float32)


def _gocmen_kontur_aday_metrikleri(koseler, shape):
    """Kontur adayini warp yapmadan once ucuz sekil olcutleriyle puanlar."""
    bos = {"alan_orani": 0.0, "en_boy_orani": 0.0, "skor": -1.0}
    koseler = np.asarray(koseler, dtype=np.float64)
    if koseler.shape != (4, 2) or not np.all(np.isfinite(koseler)):
        return bos

    h, w = shape[:2]
    ust = float(np.linalg.norm(koseler[1] - koseler[0]))
    alt = float(np.linalg.norm(koseler[2] - koseler[3]))
    sag = float(np.linalg.norm(koseler[2] - koseler[1]))
    sol = float(np.linalg.norm(koseler[3] - koseler[0]))
    genislik = (ust + alt) * 0.5
    yukseklik = (sol + sag) * 0.5
    if min(genislik, yukseklik) <= 1.0:
        return bos

    oran = genislik / yukseklik
    alan_orani = _cokgen_alani(koseler) / max(1.0, float(h * w))
    # Gocmen referansi 639/974 ~= 0.656. Farkli baski/tarama kenarlari icin
    # aralik genis tutulur; asil kabul warp sonrasi tablo imzasiyla yapilir.
    oran_yakinligi = max(0.0, 1.0 - abs(oran - 0.656) / 0.656)
    alan_puani = min(1.0, alan_orani / 0.55)
    return {
        "alan_orani": float(alan_orani),
        "en_boy_orani": float(oran),
        "skor": float(oran_yakinligi * 0.70 + alan_puani * 0.30),
    }


def _gocmen_kenar_zarfi_adayi(kenarlar, orijinal_shape):
    """Kenar piksellerinin yogun zarfindan eksen hizali ucuz bir aday uretir."""
    kenarlar = np.asarray(kenarlar)
    if kenarlar.ndim != 2 or kenarlar.size == 0:
        return None
    ys, xs = np.nonzero(kenarlar)
    if xs.size < 120:
        return None

    h, w = kenarlar.shape
    # Tekil leke/el yazilarini disarida birak; formun yogun kenarlari piksel
    # dagiliminin orta %99'unu rahatlikla doldurur.
    x0, x1 = np.percentile(xs, (0.5, 99.5))
    y0, y1 = np.percentile(ys, (0.5, 99.5))
    bw, bh = float(x1 - x0), float(y1 - y0)
    if bw < w * 0.25 or bh < h * 0.25:
        return None

    x_marj = max(2.0, bw * 0.025)
    y_marj = max(2.0, bh * 0.025)
    x0, x1 = max(0.0, x0 - x_marj), min(float(w - 1), x1 + x_marj)
    y0, y1 = max(0.0, y0 - y_marj), min(float(h - 1), y1 + y_marj)

    oh, ow = orijinal_shape[:2]
    sx = float(ow) / max(1.0, float(w))
    sy = float(oh) / max(1.0, float(h))
    return np.asarray(
        [[x0 * sx, y0 * sy], [x1 * sx, y0 * sy],
         [x1 * sx, y1 * sy], [x0 * sx, y1 * sy]],
        dtype=np.float32,
    )


def _gocmen_kontur_kose_adaylari(resim, azami_aday=3):
    """Gocmen formunun dis cercevesinden dogrulanabilir crop adaylari bulur.

    Yeni SIFT/descriptor veya OCR calistirmaz. En fazla 1100 piksellik gri
    goruntude Canny + kontur islemleri yapar. Ana akis, donen adaylari genel
    warp kalitesi ve gocmen tam-form imzasindan gecmeden kabul etmez.
    """
    metrikler = {
        "calisma_boyutu": None,
        "kenar_pikseli": 0,
        "ham_kontur": 0,
        "gecerli_aday": 0,
    }
    if not isinstance(resim, np.ndarray) or resim.ndim not in (2, 3) or resim.size == 0:
        return [], metrikler, "Kontur fallback girdisi gecersiz."

    try:
        h, w = resim.shape[:2]
        olcek = min(1.0, 1100.0 / max(h, w))
        if olcek < 1.0:
            calisma = cv2.resize(
                resim, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA
            )
        else:
            calisma = resim

        if calisma.ndim == 2:
            gri = calisma
        else:
            gri = cv2.cvtColor(calisma[:, :, :3], cv2.COLOR_BGR2GRAY)
        gri = cv2.GaussianBlur(gri, (3, 3), 0)
        p10, p90 = np.percentile(gri[::2, ::2], (10, 90))
        dinamik = max(20.0, float(p90 - p10))
        alt_esik = int(np.clip(dinamik * 0.12, 20, 55))
        ust_esik = int(np.clip(dinamik * 0.36, 65, 145))
        kenarlar = cv2.Canny(gri, alt_esik, ust_esik)
        cekirdek = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bagli = cv2.morphologyEx(
            kenarlar, cv2.MORPH_CLOSE, cekirdek, iterations=2
        )
        bulunan = cv2.findContours(
            bagli, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        konturlar = bulunan[-2]
    except Exception as exc:
        return [], metrikler, f"Kontur fallback calistirilamadi: {exc}"

    ch, cw = gri.shape[:2]
    metrikler.update({
        "calisma_boyutu": [int(cw), int(ch)],
        "kenar_pikseli": int(np.count_nonzero(kenarlar)),
        "ham_kontur": int(len(konturlar)),
    })
    if metrikler["kenar_pikseli"] < 120:
        return [], metrikler, "Form cercevesi icin yeterli kenar bulunamadi."

    adaylar = []

    def aday_ekle(koseler, kaynak):
        sirali = _dortgeni_tl_tr_br_bl_sirala(koseler)
        if sirali is None:
            return
        sirali = sirali / max(olcek, 1e-9)
        sekil = _gocmen_kontur_aday_metrikleri(sirali, resim.shape)
        if not (0.08 <= sekil["alan_orani"] <= 1.05):
            return
        if not (0.43 <= sekil["en_boy_orani"] <= 0.90):
            return
        if not koseler_gecerli_mi(sirali, resim.shape, "gocmen"):
            return

        for mevcut in adaylar:
            fark = float(np.mean(np.linalg.norm(mevcut["koseler"] - sirali, axis=1)))
            if fark <= max(h, w) * 0.025:
                mevcut_kapali = mevcut.get("kaynak") == "dortgen_kontur"
                yeni_kapali = kaynak == "dortgen_kontur"
                # SIKI on tespit gercek kapali dortgen kanitini arar. Ayni
                # cercevenin minAreaRect kopyasi birkac puan daha yuksek diye
                # bu kaynak bilgisini kaybetmesin.
                if yeni_kapali and not mevcut_kapali:
                    mevcut.update({"koseler": sirali, "kaynak": kaynak, **sekil})
                elif not mevcut_kapali and sekil["skor"] > mevcut["skor"]:
                    mevcut.update({"koseler": sirali, "kaynak": kaynak, **sekil})
                return
        adaylar.append({"koseler": sirali, "kaynak": kaynak, **sekil})

    alan_esigi = float(ch * cw) * 0.06
    try:
        sirali_konturlar = sorted(
            konturlar, key=lambda k: float(cv2.contourArea(k)), reverse=True
        )[:24]
        for kontur in sirali_konturlar:
            alan = float(cv2.contourArea(kontur))
            if alan < alan_esigi:
                break
            cevre = float(cv2.arcLength(kontur, True))
            if cevre <= 0.0:
                continue
            yaklasik = cv2.approxPolyDP(kontur, 0.025 * cevre, True)
            if len(yaklasik) == 4 and cv2.isContourConvex(yaklasik):
                aday_ekle(np.asarray(yaklasik).reshape(4, 2), "dortgen_kontur")
            kutu = cv2.boxPoints(cv2.minAreaRect(kontur))
            aday_ekle(kutu, "donuk_dikdortgen")
    except Exception:
        # Tek bir bozuk kontur, asagidaki kenar-zarfi adayini engellememeli.
        pass

    zarf = _gocmen_kenar_zarfi_adayi(kenarlar, resim.shape)
    if zarf is not None:
        # Zarf asil goruntu koordinatindadir; aday_ekle calisma olcegini geri
        # alacagi icin once calisma koordinatina cevir.
        aday_ekle(zarf * olcek, "kenar_zarfi")

    adaylar.sort(key=lambda aday: aday["skor"], reverse=True)
    adaylar = adaylar[:max(1, int(azami_aday))]
    metrikler["gecerli_aday"] = int(len(adaylar))
    if not adaylar:
        return [], metrikler, "Gocmen formuna benzeyen guvenli kontur bulunamadi."
    return adaylar, metrikler, ""


# =========================================================
# GEOMETRİ KONTROL
# =========================================================

def _vektorel_carpim(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _cokgen_alani(noktalar):
    """Nokta sirasi korunarak cokgenin mutlak alanini hesaplar."""
    noktalar = np.asarray(noktalar, dtype=np.float64)
    if noktalar.ndim != 2 or noktalar.shape[0] < 3 or noktalar.shape[1] != 2:
        return 0.0
    x = noktalar[:, 0]
    y = noktalar[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _nokta_segment_uzerinde_mi(nokta, bas, son, eps):
    if abs(_vektorel_carpim(son - bas, nokta - bas)) > eps:
        return False
    return (
        min(bas[0], son[0]) - eps <= nokta[0] <= max(bas[0], son[0]) + eps
        and min(bas[1], son[1]) - eps <= nokta[1] <= max(bas[1], son[1]) + eps
    )


def _segmentler_kesisiyor_mu(a, b, c, d, eps):
    """Ortak ucu olmayan iki kenarin kesisip kesismedigini bulur."""
    o1 = _vektorel_carpim(b - a, c - a)
    o2 = _vektorel_carpim(b - a, d - a)
    o3 = _vektorel_carpim(d - c, a - c)
    o4 = _vektorel_carpim(d - c, b - c)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    return (
        (abs(o1) <= eps and _nokta_segment_uzerinde_mi(c, a, b, eps))
        or (abs(o2) <= eps and _nokta_segment_uzerinde_mi(d, a, b, eps))
        or (abs(o3) <= eps and _nokta_segment_uzerinde_mi(a, c, d, eps))
        or (abs(o4) <= eps and _nokta_segment_uzerinde_mi(b, c, d, eps))
    )


def _cokgeni_eksende_kirp(noktalar, eksen, sinir, alt_sinir):
    """Sutherland-Hodgman ile cokgeni tek bir goruntu sinirina kirpar."""
    if len(noktalar) == 0:
        return noktalar

    def iceride(nokta):
        return nokta[eksen] >= sinir if alt_sinir else nokta[eksen] <= sinir

    def kesim(bas, son):
        fark = son[eksen] - bas[eksen]
        if abs(fark) < 1e-12:
            return bas.copy()
        oran = (sinir - bas[eksen]) / fark
        return bas + oran * (son - bas)

    cikti = []
    onceki = noktalar[-1]
    onceki_iceride = iceride(onceki)
    for simdiki in noktalar:
        simdiki_iceride = iceride(simdiki)
        if simdiki_iceride:
            if not onceki_iceride:
                cikti.append(kesim(onceki, simdiki))
            cikti.append(simdiki)
        elif onceki_iceride:
            cikti.append(kesim(onceki, simdiki))
        onceki = simdiki
        onceki_iceride = simdiki_iceride
    return np.asarray(cikti, dtype=np.float64)


def _goruntu_icindeki_cokgen_alani(koseler, w, h):
    kirpilmis = np.asarray(koseler, dtype=np.float64)
    for eksen, sinir, alt_sinir in (
        (0, 0.0, True), (0, float(w - 1), False),
        (1, 0.0, True), (1, float(h - 1), False),
    ):
        kirpilmis = _cokgeni_eksende_kirp(kirpilmis, eksen, sinir, alt_sinir)
        if len(kirpilmis) < 3:
            return 0.0
    return _cokgen_alani(kirpilmis)


def koseler_gecerli_mi(koseler, shape, belge_tipi):
    if koseler is None:
        return False

    koseler = np.asarray(koseler, dtype=np.float32)
    if koseler.shape != (4, 2):
        return False
    if not np.all(np.isfinite(koseler)):
        return False

    h, w = shape[:2]
    if h < 2 or w < 2:
        return False

    # TL -> TR -> BR -> BL sirasi bozuldugunda perspectiveTransform yine bir
    # sonuc uretebilir; fakat bow-tie/konkav dortgen warp'ta goruntuyu ince bir
    # serit halinde yayar. Karsi kenar kesismesi ve ayni isaretli donusler bu
    # durumu warp'tan once, ucuzca eler.
    koseler64 = koseler.astype(np.float64)
    kenar_vektorleri = np.roll(koseler64, -1, axis=0) - koseler64
    kenarlar = np.linalg.norm(kenar_vektorleri, axis=1)
    en_uzun_kenar = float(np.max(kenarlar))
    eps = max(1e-6, en_uzun_kenar * en_uzun_kenar * 1e-7)

    if _segmentler_kesisiyor_mu(koseler64[0], koseler64[1], koseler64[2], koseler64[3], eps):
        return False
    if _segmentler_kesisiyor_mu(koseler64[1], koseler64[2], koseler64[3], koseler64[0], eps):
        return False

    donusler = np.asarray([
        _vektorel_carpim(kenar_vektorleri[i], kenar_vektorleri[(i + 1) % 4])
        for i in range(4)
    ])
    if not (np.all(donusler > eps) or np.all(donusler < -eps)):
        return False

    alan = _cokgen_alani(koseler64)
    alan_orani = alan / max(1, h * w)

    # Eski TC / göçmen için daha toleranslı: eğik/yan/fotoğraf şeklinde gelebiliyor.
    if belge_tipi == "eski_tc":
        if not (0.01 <= alan_orani <= 2.0):
            return False
        tol_x, tol_y = w * 0.60, h * 0.60
    elif belge_tipi == "gocmen":
        if not (0.01 <= alan_orani <= 1.8):
            return False
        tol_x, tol_y = w * 0.45, h * 0.45
    else:
        if not (0.02 <= alan_orani <= 1.3):
            return False
        tol_x, tol_y = w * 0.25, h * 0.25

    if np.any(koseler[:, 0] < -tol_x) or np.any(koseler[:, 0] > w + tol_x):
        return False
    if np.any(koseler[:, 1] < -tol_y) or np.any(koseler[:, 1] > h + tol_y):
        return False

    tl, tr, br, bl = koseler64
    ust, sag, alt, sol = kenarlar

    if min(ust, alt, sol, sag) < 20:
        return False

    # Tek bir kenarin cok kuculmesi, ortalama genislik/yukseklik oraninda
    # saklanabiliyordu. Ozellikle gocmen referansinda bu durum beyaz/serit bir
    # warp uretiyor. Karsi kenar ve tum kenar oranlari bunu yakalar.
    karsi_kenar_orani = max(ust, alt) / min(ust, alt)
    karsi_diger_oran = max(sol, sag) / min(sol, sag)
    tum_kenar_orani = max(ust, alt, sol, sag) / min(ust, alt, sol, sag)

    # Dortgenin acilari sifira/180 dereceye yaklastiginda homografi sayisal
    # olarak kararsizlasir. Her iki yonu de kabul edip yalnizca asiri acilari
    # reddediyoruz.
    acilar = []
    for i in range(4):
        onceki = koseler64[(i - 1) % 4] - koseler64[i]
        sonraki = koseler64[(i + 1) % 4] - koseler64[i]
        payda = np.linalg.norm(onceki) * np.linalg.norm(sonraki)
        if payda <= 1e-9:
            return False
        kosinus = np.clip(float(np.dot(onceki, sonraki) / payda), -1.0, 1.0)
        acilar.append(float(np.degrees(np.arccos(kosinus))))

    bbox_w = float(np.ptp(koseler64[:, 0]))
    bbox_h = float(np.ptp(koseler64[:, 1]))
    bbox_doluluk = alan / max(1.0, bbox_w * bbox_h)
    kompaktlik = alan / max(1.0, en_uzun_kenar * en_uzun_kenar)
    gorunen_alan = _goruntu_icindeki_cokgen_alani(koseler64, w, h)
    gorunen_oran = gorunen_alan / max(1.0, alan)

    if belge_tipi == "gocmen":
        if (
            min(acilar) < 12.0 or max(acilar) > 168.0
            or max(karsi_kenar_orani, karsi_diger_oran) > 5.0
            or tum_kenar_orani > 10.0
            or bbox_doluluk < 0.18 or kompaktlik < 0.06
            or gorunen_oran < 0.38
        ):
            return False
    elif belge_tipi == "eski_tc":
        if (
            min(acilar) < 8.0 or max(acilar) > 172.0
            or max(karsi_kenar_orani, karsi_diger_oran) > 8.0
            or tum_kenar_orani > 14.0
            or bbox_doluluk < 0.12 or kompaktlik < 0.025
            or gorunen_oran < 0.20
        ):
            return False
    else:
        # Yeni T.C. mutlu yolunu korumak icin sinirlar bilerek genis; yalnizca
        # matematiksel olarak cokmus homografiler elenir.
        if (
            min(acilar) < 6.0 or max(acilar) > 174.0
            or max(karsi_kenar_orani, karsi_diger_oran) > 10.0
            or tum_kenar_orani > 15.0
            or bbox_doluluk < 0.08 or kompaktlik < 0.015
            or gorunen_oran < 0.20
        ):
            return False

    genislik, yukseklik = (ust + alt) / 2.0, (sol + sag) / 2.0
    aspect = max(genislik, yukseklik) / min(genislik, yukseklik)

    if belge_tipi == "eski_tc":
        return 1.05 <= aspect <= 2.10
    if belge_tipi == "tc":
        return 1.15 <= aspect <= 2.20
    if belge_tipi == "gocmen":
        return 1.00 <= aspect <= 3.30
    return False


# =========================================================
# PERSPEKTİF DÜZELT
# =========================================================

def perspektif_duzelt(resim, koseler, belge_tipi):
    """koseler homography'den referans TL->TR->BR->BL sırasıyla gelir; kart
    görüntüde 90 derece dönmüş olsa bile correspondence doğru kalır, bu
    yüzden köşeler burada tekrar sıralanmıyor."""
    src = np.asarray(koseler, dtype=np.float32)

    if belge_tipi == "eski_tc":
        hedef_w, hedef_h = 750, 1050
    elif belge_tipi == "gocmen":
        # Referans 639x974 (oran 0.656). Onceki 800x1100 hedefi karti yatayda
        # yaklasik %11 esnetiyor, kucuk harfleri bozuyor ve gereksiz OCR pikseli
        # uretiyordu. 720x1100 referans oranini korur.
        hedef_w, hedef_h = 720, 1100
    else:
        hedef_w, hedef_h = 1000, 630

    dst = np.float32([[0, 0], [hedef_w - 1, 0], [hedef_w - 1, hedef_h - 1], [0, hedef_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        resim, M, (hedef_w, hedef_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# =========================================================
# DEBUG — yalnızca kart sınırı (basit, ucuz); SIFT eşleşme çizgisi kaldırıldı
# =========================================================
#
# Not: iki görüntü arasında eşleşen keypoint'leri çizgilerle birleştiren
# 'match_debug_resmi' (cv2.drawMatches) kaldırıldı. Bu görsel yalnızca
# geliştirme sırasında "SIFT hangi noktaları eşleştirdi" diye incelemek
# içindi; TC/ad/soyad çıkarımını hiçbir şekilde etkilemiyordu, ekstra kod ve
# (açıldığında) ekstra render maliyeti dışında bir katkısı yoktu. Kart
# sınırını (bulunan dörtgeni) tek görüntü üzerinde gösteren 'kart_debug_resmi'
# — asıl işe yarayan, "kart doğru bulundu mu" sorusuna cevap veren debug —
# olduğu gibi kalıyor.

def kart_debug_resmi(resim, koseler):
    debug = resim.copy()
    poly = np.asarray(koseler, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [poly], True, (0, 255, 0), 5, cv2.LINE_AA)

    for isim, nokta in zip(("TL", "TR", "BR", "BL"), koseler):
        x, y = map(int, nokta)
        cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(debug, isim, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return debug


# =========================================================
# ANA
# =========================================================

def _girdi_goruntusunu_hazirla(resim):
    """Ana akisin kullanabilecegi BGR/uint8 goruntuyu ve hata metnini dondurur."""
    if resim is None:
        return None, "Görüntü bulunamadı."
    if not isinstance(resim, np.ndarray):
        return None, "Görüntü NumPy dizisi olmalıdır."
    if resim.size == 0:
        return None, "Görüntü boş."
    if resim.ndim not in (2, 3):
        return None, f"Geçersiz görüntü şekli: {resim.shape}."
    if resim.shape[0] < 2 or resim.shape[1] < 2:
        return None, f"Görüntü boyutları çok küçük: {resim.shape}."
    if resim.dtype != np.uint8:
        return None, f"Geçersiz görüntü veri tipi: {resim.dtype}; uint8 bekleniyor."

    try:
        if resim.ndim == 2:
            hazir = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
        elif resim.shape[2] == 1:
            hazir = cv2.cvtColor(resim[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif resim.shape[2] == 3:
            hazir = resim
        elif resim.shape[2] == 4:
            hazir = cv2.cvtColor(resim, cv2.COLOR_BGRA2BGR)
        else:
            return None, f"Geçersiz kanal sayısı: {resim.shape[2]}; 1, 3 veya 4 bekleniyor."
    except Exception as exc:
        return None, f"Görüntü kanalları hazırlanamadı: {exc}"

    return np.ascontiguousarray(hazir), None


def _referans_hatalarini_al(tip_sonuc):
    return {
        tip: aday.get("hata")
        for tip, aday in tip_sonuc.get("sonuclar", {}).items()
        if aday.get("hata")
    }


def _kart_ciktisi_gecerli_mi(kart):
    return (
        isinstance(kart, np.ndarray)
        and kart.size > 0
        and kart.ndim in (2, 3)
        and kart.shape[0] > 1
        and kart.shape[1] > 1
    )


def _aktif_aralik_orani(aktif):
    indisler = np.flatnonzero(aktif)
    if indisler.size == 0:
        return 0.0
    return float(indisler[-1] - indisler[0] + 1) / max(1, aktif.size)


def _radyal_yakinsama_metrikleri(ornek):
    """Kenarlarin tek bir sonlu noktadan isin gibi yayilmasini olcer.

    Cokmus bir homografi, kart ayrintisini bir koseye dogru uzanan fan/
    starburst cizgilerine cevirir. Boyle bir cikti mevcut doluluk ve ayrinti
    kapsami kontrollerini gecebilir; cunku bozukluk butun karta yayilmistir.

    Her kuvvetli kenar pikselinde gradyan, o kenarin normalidir. Ayni merkezden
    cikan isinlarda bilinmeyen merkez bu normallerin tanimladigi dogrularin
    ortak cozumudur. Iki degiskenli agirlikli en kucuk kareler hesabi sabit
    boyutlu (en fazla 240x240) ornek uzerinde yapilir ve OCR'a gore cok ucuzdur.
    """
    ornek = np.asarray(ornek, dtype=np.float64)
    bos_sonuc = {
        "radyal_yakinsama_skoru": 0.0,
        "radyal_ayrinti_kapsami": 0.0,
        "radyal_merkez_x": None,
        "radyal_merkez_y": None,
    }
    if ornek.ndim != 2 or min(ornek.shape) < 12 or not np.all(np.isfinite(ornek)):
        return bos_sonuc

    # Merkezi fark gradyani; dis cerceve warp/borderMode kaynakli tek bir
    # yapay kenarla skoru sisirmesin diye iki piksellik sinir kullanilmaz.
    gx = np.zeros_like(ornek)
    gy = np.zeros_like(ornek)
    gx[:, 1:-1] = (ornek[:, 2:] - ornek[:, :-2]) * 0.5
    gy[1:-1, :] = (ornek[2:, :] - ornek[:-2, :]) * 0.5
    buyukluk = np.hypot(gx, gy)
    ic = buyukluk[2:-2, 2:-2]
    pozitif = ic[ic > 1e-6]
    if pozitif.size < 80:
        return bos_sonuc

    esik = max(4.0, float(np.percentile(pozitif, 82.0)))
    maske = buyukluk >= esik
    maske[:2, :] = False
    maske[-2:, :] = False
    maske[:, :2] = False
    maske[:, -2:] = False
    ys, xs = np.nonzero(maske)
    if xs.size < 80:
        return bos_sonuc

    # Hesabi her kosulda sinirli tutmak icin deterministik seyrek ornekleme.
    if xs.size > 6000:
        adim = int(np.ceil(xs.size / 6000.0))
        xs, ys = xs[::adim], ys[::adim]

    kuvvet = buyukluk[ys, xs]
    nx = gx[ys, xs] / np.maximum(kuvvet, 1e-9)
    ny = gy[ys, xs] / np.maximum(kuvvet, 1e-9)
    agirlik = np.clip(kuvvet / esik, 1.0, 4.0)

    # n . merkez = n . piksel. Farkli yonlerde yeterli kenar yoksa matris
    # tekildir; paralel metin/tablo cizgileri radyal bozukluk sayilmaz.
    a00 = float(np.sum(agirlik * nx * nx))
    a01 = float(np.sum(agirlik * nx * ny))
    a11 = float(np.sum(agirlik * ny * ny))
    matris = np.asarray([[a00, a01], [a01, a11]], dtype=np.float64)
    ozdegerler = np.linalg.eigvalsh(matris)
    if ozdegerler[-1] <= 1e-9 or ozdegerler[0] / ozdegerler[-1] < 0.06:
        return bos_sonuc

    sag_taraf = np.asarray([
        np.sum(agirlik * nx * (nx * xs + ny * ys)),
        np.sum(agirlik * ny * (nx * xs + ny * ys)),
    ], dtype=np.float64)
    try:
        merkez = np.linalg.solve(matris, sag_taraf)
    except np.linalg.LinAlgError:
        return bos_sonuc

    h, w = ornek.shape
    # Paralel cizgilerin sayisal "kesisimi" cok uzakta olur. Ekran
    # goruntulerindeki fan merkezi ise kartin icinde veya hemen disindadir.
    merkez_sonucu = {
        "radyal_yakinsama_skoru": 0.0,
        "radyal_ayrinti_kapsami": 0.0,
        "radyal_merkez_x": float(merkez[0] / max(1, w - 1)),
        "radyal_merkez_y": float(merkez[1] / max(1, h - 1)),
    }
    if not (-0.35 * w <= merkez[0] <= 1.35 * w and -0.35 * h <= merkez[1] <= 1.35 * h):
        return merkez_sonucu

    artik = np.abs(nx * (merkez[0] - xs) + ny * (merkez[1] - ys))
    tolerans = max(1.5, min(h, w) * 0.012)
    uyumlu = artik <= tolerans
    skor = float(np.sum(agirlik[uyumlu]) / max(1e-9, np.sum(agirlik)))

    # Kucuk bir logo/filigran da kendi merkezine yakinsayabilir. Radyal
    # bozulmada uyumlu kenarlar kartin buyuk bolumune yayilir; 6x6 hucre
    # kapsami bu ikisini ayirir.
    hucre_y = np.minimum(5, (ys[uyumlu] * 6 // max(1, h)).astype(np.int32))
    hucre_x = np.minimum(5, (xs[uyumlu] * 6 // max(1, w)).astype(np.int32))
    hucreler = np.unique(hucre_y * 6 + hucre_x) if hucre_x.size else np.asarray([])
    kapsam = float(hucreler.size / 36.0)

    return {
        "radyal_yakinsama_skoru": skor,
        "radyal_ayrinti_kapsami": kapsam,
        "radyal_merkez_x": merkez_sonucu["radyal_merkez_x"],
        "radyal_merkez_y": merkez_sonucu["radyal_merkez_y"],
    }


def _kart_ciktisi_kalite_metrikleri(kart):
    """Warp ciktisinda ayrintinin karta yayilip yayilmadigini olcer.

    Hesap sabit boyutlu OCR goruntusunun seyrek bir ornegi uzerinde yapilir;
    boylece kontrol OCR'a gore cok ucuz kalir. Metrikler metin okumaz, yalnizca
    ekrandaki gibi buyuk bos alan + dar cizgisel serit sonucunu ayirt eder.
    """
    if not _kart_ciktisi_gecerli_mi(kart):
        return {}

    if kart.ndim == 2:
        gri = kart.astype(np.float32)
    elif kart.shape[2] == 1:
        gri = kart[:, :, 0].astype(np.float32)
    elif kart.shape[2] >= 3:
        bgr = kart[:, :, :3].astype(np.float32)
        gri = bgr[:, :, 0] * 0.114 + bgr[:, :, 1] * 0.587 + bgr[:, :, 2] * 0.299
    else:
        return {}

    # En fazla yaklasik 240x240 piksel: percentile/tile kontrollerinin maliyeti
    # giris cozunurluguyle buyumez.
    adim_y = max(1, int(np.ceil(gri.shape[0] / 240.0)))
    adim_x = max(1, int(np.ceil(gri.shape[1] / 240.0)))
    ornek = gri[::adim_y, ::adim_x]
    if ornek.size == 0 or not np.all(np.isfinite(ornek)):
        return {}

    p05, p95 = np.percentile(ornek, (5, 95))
    dinamik_aralik = float(p95 - p05)
    standart_sapma = float(np.std(ornek))
    arka_plan = float(np.median(ornek))
    fark_esigi = max(6.0, dinamik_aralik * 0.08)
    ayrinti = np.abs(ornek - arka_plan) >= fark_esigi

    satir_min = max(1, int(np.ceil(ornek.shape[1] * 0.012)))
    sutun_min = max(1, int(np.ceil(ornek.shape[0] * 0.012)))
    aktif_satirlar = np.count_nonzero(ayrinti, axis=1) >= satir_min
    aktif_sutunlar = np.count_nonzero(ayrinti, axis=0) >= sutun_min

    hucre_sayisi = 0
    bilgili_hucre = 0
    hucre_std_esigi = max(4.0, dinamik_aralik * 0.035)
    hucre_gradyan_esigi = max(0.8, dinamik_aralik * 0.006)
    for y_indisleri in np.array_split(np.arange(ornek.shape[0]), 6):
        for x_indisleri in np.array_split(np.arange(ornek.shape[1]), 6):
            if y_indisleri.size == 0 or x_indisleri.size == 0:
                continue
            hucre = ornek[np.ix_(y_indisleri, x_indisleri)]
            yatay = float(np.mean(np.abs(np.diff(hucre, axis=1)))) if hucre.shape[1] > 1 else 0.0
            dikey = float(np.mean(np.abs(np.diff(hucre, axis=0)))) if hucre.shape[0] > 1 else 0.0
            hucre_sayisi += 1
            if float(np.std(hucre)) >= hucre_std_esigi and max(yatay, dikey) >= hucre_gradyan_esigi:
                bilgili_hucre += 1

    # Sekiz gri seviyesini tek kovada toplayarak devasa tek renkli alanlari
    # JPEG gurultusune takilmadan olceriz.
    kovalar = np.clip(np.floor(ornek / 8.0).astype(np.int16), 0, 31)
    _, sayilar = np.unique(kovalar, return_counts=True)
    baskin_ton_orani = float(np.max(sayilar) / ornek.size) if sayilar.size else 1.0
    radyal_metrikler = _radyal_yakinsama_metrikleri(ornek)

    return {
        "dinamik_aralik": dinamik_aralik,
        "standart_sapma": standart_sapma,
        "bilgili_hucre_orani": float(bilgili_hucre / max(1, hucre_sayisi)),
        "aktif_satir_orani": float(np.mean(aktif_satirlar)),
        "aktif_sutun_orani": float(np.mean(aktif_sutunlar)),
        "dikey_ayrinti_kapsami": _aktif_aralik_orani(aktif_satirlar),
        "yatay_ayrinti_kapsami": _aktif_aralik_orani(aktif_sutunlar),
        "baskin_ton_orani": baskin_ton_orani,
        **radyal_metrikler,
    }


def _kart_ciktisi_kaliteli_mi(kart, belge_tipi):
    metrikler = _kart_ciktisi_kalite_metrikleri(kart)
    if not metrikler:
        return False, metrikler

    # Yeni T.C. icin esikler bilerek daha yumusak tutuldu. Gocmen karti uzun
    # form yapisinda oldugundan yazi/cizgi/fotografin iki eksene de yayilmasi
    # beklenir; dar bir diagonal serit bu esikleri gecemez.
    if belge_tipi == "gocmen":
        sinirlar = {
            "bilgili_hucre_orani": 0.24,
            "aktif_satir_orani": 0.16,
            "aktif_sutun_orani": 0.16,
            "dikey_ayrinti_kapsami": 0.50,
            "yatay_ayrinti_kapsami": 0.50,
        }
        baskin_ton_ust_siniri = 0.94
    elif belge_tipi == "eski_tc":
        sinirlar = {
            "bilgili_hucre_orani": 0.16,
            "aktif_satir_orani": 0.12,
            "aktif_sutun_orani": 0.12,
            "dikey_ayrinti_kapsami": 0.45,
            "yatay_ayrinti_kapsami": 0.45,
        }
        baskin_ton_ust_siniri = 0.96
    else:
        sinirlar = {
            "bilgili_hucre_orani": 0.10,
            "aktif_satir_orani": 0.08,
            "aktif_sutun_orani": 0.08,
            "dikey_ayrinti_kapsami": 0.35,
            "yatay_ayrinti_kapsami": 0.35,
        }
        baskin_ton_ust_siniri = 0.985

    temel_ok = metrikler["dinamik_aralik"] >= 10.0 and metrikler["standart_sapma"] >= 3.0
    yayilim_ok = all(metrikler[ad] >= sinir for ad, sinir in sinirlar.items())
    ton_ok = metrikler["baskin_ton_orani"] <= baskin_ton_ust_siniri
    # Esikler belge turunden bagimsizdir: duzeltilmis hicbir dokumanin baskin
    # kenarlari tek bir sonlu noktadan kartin yaklasik yarisina yayilmamali.
    radyal_ok = not (
        metrikler["radyal_yakinsama_skoru"] >= 0.58
        # 720x1100 gocmen hedefinde 6x6 izgara nicemlemesi ayni fan icin
        # kapsami ~0.47 olcebiliyor. Skor esigi zaten yuksek oldugundan 0.45
        # kapsam, normal logo/metin kenarlarini reddetmeden bu ciktiyi yakalar.
        and metrikler["radyal_ayrinti_kapsami"] >= 0.45
    )
    metrikler["radyal_gecerli"] = bool(radyal_ok)
    metrikler["gecerli"] = bool(temel_ok and yayilim_ok and ton_ok and radyal_ok)
    return metrikler["gecerli"], metrikler


def _gocmen_tablo_imzasi_gecerli_mi(kart):
    """Affine gocmen crop'unda deger ayirici ve yatay grid kaniti arar.

    Kontrol metin okumaz. En fazla yaklasik 360x550 gri pikselde basit
    gradyan kapsamlari hesapladigi icin yalnizca fallback'te cok ucuzdur.
    Esikler soluk/fotokopi cizgilerinde kesintiye izin verecek kadar yumusak,
    fakat rastgele metin kenarlarini tablo saymayacak kadar mekansaldir.
    """
    metrikler = {
        "yatay_cizgi_grubu": 0,
        "yatay_grid_yayilimi": 0.0,
        "deger_ayirici_kapsami": 0.0,
        "deger_ayirici_x": None,
        "ortak_yatay_cizgi_grubu": 0,
        "ortak_yatay_cizgi_orani": 0.0,
        "ortak_yatay_grid_yayilimi": 0.0,
        "gecerli": False,
    }
    if not _kart_ciktisi_gecerli_mi(kart):
        return False, metrikler

    if kart.ndim == 2:
        gri = kart.astype(np.float32)
    elif kart.shape[2] == 1:
        gri = kart[:, :, 0].astype(np.float32)
    elif kart.shape[2] >= 3:
        bgr = kart[:, :, :3].astype(np.float32)
        gri = bgr[:, :, 0] * 0.114 + bgr[:, :, 1] * 0.587 + bgr[:, :, 2] * 0.299
    else:
        return False, metrikler

    adim_y = max(1, int(np.ceil(gri.shape[0] / 550.0)))
    adim_x = max(1, int(np.ceil(gri.shape[1] / 360.0)))
    ornek = gri[::adim_y, ::adim_x]
    h, w = ornek.shape
    if h < 30 or w < 30 or not np.all(np.isfinite(ornek)):
        return False, metrikler

    y0, y1 = int(round(h * 0.32)), int(round(h * 0.96))
    x0, x1 = int(round(w * 0.04)), int(round(w * 0.96))
    tablo = ornek[y0:y1, x0:x1]
    if min(tablo.shape) < 20:
        return False, metrikler

    p10, p90 = np.percentile(tablo, (10, 90))
    gradyan_esigi = max(4.0, float(p90 - p10) * 0.025)
    dikey_fark = np.abs(np.diff(tablo, axis=0))
    yatay_fark = np.abs(np.diff(tablo, axis=1))

    # Bir yatay grid cizgisi, komsu satirla arasinda kart genisliginin kayda
    # deger bolumunde ton gecisi olusturur. Birbirine 2 satirdan yakin
    # cevaplari ayni basili cizginin iki kenari olarak tek grupta toplariz.
    satir_kapsami = np.mean(dikey_fark >= gradyan_esigi, axis=1)
    # Dokulu/fotokopi zeminde sabit esik cok sayida komsu satiri tek gruba
    # birlestirebilir. En kuvvetli yuzde 15'i almak gercek grid tepelerini
    # ayirir; temiz taramada ise 0.22 tabani soluk/kesik cizgileri korur.
    satir_esigi = max(0.22, float(np.percentile(satir_kapsami, 85.0)))
    aktif_satirlar = np.flatnonzero(satir_kapsami >= satir_esigi)
    satir_gruplari = []
    if aktif_satirlar.size:
        bas = onceki = int(aktif_satirlar[0])
        for indis in aktif_satirlar[1:]:
            indis = int(indis)
            if indis - onceki > 2:
                satir_gruplari.append((bas + onceki) * 0.5)
                bas = indis
            onceki = indis
        satir_gruplari.append((bas + onceki) * 0.5)

    yatay_yayilim = 0.0
    if len(satir_gruplari) >= 2:
        yatay_yayilim = float(
            (satir_gruplari[-1] - satir_gruplari[0]) / max(1, tablo.shape[0] - 1)
        )

    # Bir-iki piksellik egim/kesintiyi tolere etmek icin her x konumunda
    # bes sutunluk komsuluktaki en kuvvetli yatay gradyani kullaniriz.
    genisletilmis = yatay_fark.copy()
    for kayma in (1, 2):
        genisletilmis[:, kayma:] = np.maximum(
            genisletilmis[:, kayma:], yatay_fark[:, :-kayma]
        )
        genisletilmis[:, :-kayma] = np.maximum(
            genisletilmis[:, :-kayma], yatay_fark[:, kayma:]
        )
    sutun_kapsami = np.mean(genisletilmis >= gradyan_esigi, axis=0)
    merkez_bas = int(round(sutun_kapsami.size * 0.22))
    merkez_son = int(round(sutun_kapsami.size * 0.78))
    merkez = sutun_kapsami[merkez_bas:merkez_son]
    if merkez.size:
        yerel_indis = int(np.argmax(merkez))
        ayirici_kapsami = float(merkez[yerel_indis])
        ayirici_indis = merkez_bas + yerel_indis
        ayirici_x = float((x0 + ayirici_indis) / max(1, w - 1))
    else:
        ayirici_kapsami = 0.0
        ayirici_x = None

    # Genel satir kapsami ile ayirici kanitini ayri ayri aramak yeterli
    # degildir: yan yana on+arka kart bulunan bir sayfada yatay satirlar sol
    # karttan, kuvvetli dikey cizgi ise iki kartin sinirindan gelebilir. Gercek
    # deger ayiricisinda ayni grid satirlari cizginin hem solunda hem saginda
    # devam eder. Bu metrik genel tablo kapisini degistirmez; post-SIFT kontur
    # secimindeki daha ozgul tek-form kontrolu tarafindan kullanilir.
    ortak_satirlar = []
    if ayirici_x is not None and satir_gruplari:
        tablo_w = int(tablo.shape[1])
        ayirici = int(np.clip(ayirici_indis, 0, max(0, tablo_w - 1)))
        bosluk = max(2, int(round(tablo_w * 0.012)))
        yan_genislik = max(8, int(round(tablo_w * 0.24)))
        sol0, sol1 = max(0, ayirici - yan_genislik), max(0, ayirici - bosluk)
        sag0, sag1 = min(tablo_w, ayirici + bosluk), min(
            tablo_w, ayirici + yan_genislik
        )
        if sol1 - sol0 >= 6 and sag1 - sag0 >= 6:
            for satir in satir_gruplari:
                merkez_y = int(round(satir))
                sy0 = max(0, merkez_y - 2)
                sy1 = min(dikey_fark.shape[0], merkez_y + 3)
                if sy1 <= sy0:
                    continue
                yerel_tepe = np.max(dikey_fark[sy0:sy1], axis=0)
                sol_destek = float(np.mean(yerel_tepe[sol0:sol1] >= gradyan_esigi))
                sag_destek = float(np.mean(yerel_tepe[sag0:sag1] >= gradyan_esigi))
                if min(sol_destek, sag_destek) >= 0.20:
                    ortak_satirlar.append(float(satir))

    ortak_oran = float(len(ortak_satirlar) / max(1, len(satir_gruplari)))
    ortak_yayilim = 0.0
    if len(ortak_satirlar) >= 2:
        ortak_yayilim = float(
            (ortak_satirlar[-1] - ortak_satirlar[0])
            / max(1, tablo.shape[0] - 1)
        )

    gecerli = (
        len(satir_gruplari) >= 3
        and yatay_yayilim >= 0.20
        and ayirici_kapsami >= 0.28
        and ayirici_x is not None
        and 0.36 <= ayirici_x <= 0.64
    )
    metrikler.update({
        "yatay_cizgi_grubu": int(len(satir_gruplari)),
        "yatay_grid_yayilimi": yatay_yayilim,
        "deger_ayirici_kapsami": ayirici_kapsami,
        "deger_ayirici_x": ayirici_x,
        "ortak_yatay_cizgi_grubu": int(len(ortak_satirlar)),
        "ortak_yatay_cizgi_orani": ortak_oran,
        "ortak_yatay_grid_yayilimi": ortak_yayilim,
        "gecerli": bool(gecerli),
    })
    return bool(gecerli), metrikler


def _tespit_suresini_tamamla(sonuc, baslangic):
    sonuc["tespit_suresi"] = max(0.0, float(time.perf_counter() - baslangic))
    return sonuc


def _koselerle_karti_duzeltmeyi_dene(resim, koseler, belge_tipi):
    """Ayni geometri ve kalite kapilariyla tek bir koseler kumesini dener."""
    deneme = {
        "basarili": False,
        "kart": None,
        "geometri_gecerli": False,
        "cikti_kalitesi_gecerli": False,
        "kalite_metrikleri": {},
        "hata": "",
    }
    try:
        geometri_ok = koseler_gecerli_mi(koseler, resim.shape, belge_tipi)
    except Exception as exc:
        deneme["hata"] = f"Geometri kontrolu calistirilamadi: {exc}"
        return deneme

    deneme["geometri_gecerli"] = bool(geometri_ok)
    if not geometri_ok:
        deneme["hata"] = "Kart geometrisi guvenilir degil."
        return deneme

    try:
        kart = perspektif_duzelt(resim, koseler, belge_tipi)
        if not _kart_ciktisi_gecerli_mi(kart):
            raise ValueError("perspektif duzeltme bos veya gecersiz goruntu uretti")

        kalite_ok, kalite_metrikleri = _kart_ciktisi_kaliteli_mi(kart, belge_tipi)
        deneme["kalite_metrikleri"] = kalite_metrikleri
        deneme["cikti_kalitesi_gecerli"] = bool(kalite_ok)
        if not kalite_ok:
            raise ValueError("perspektif duzeltme cikti kalitesi guvenilir degil")
    except Exception as exc:
        deneme["hata"] = f"Perspektif duzeltilemedi: {exc}"
        return deneme

    deneme.update({"basarili": True, "kart": kart})
    return deneme


def _gocmen_hizli_form_imzasi_gecerli_mi(kart):
    """Tam gocmen on formunu tablo ve koyu ust bantla sikica dogrular."""
    tablo_ok, tablo = _gocmen_tablo_imzasi_gecerli_mi(kart)
    metrikler = {
        "tablo": tablo,
        "ust_bant_koyu_orani": 0.0,
        "ust_bant_asgari_parca_orani": 0.0,
        "govde_koyu_orani": 0.0,
        "ust_bant_ortalama_farki": 0.0,
        "gecerli": False,
    }
    if not tablo_ok or not _kart_ciktisi_gecerli_mi(kart):
        return False, metrikler

    if kart.ndim == 2:
        gri = kart.astype(np.float32)
    elif kart.shape[2] == 1:
        gri = kart[:, :, 0].astype(np.float32)
    else:
        bgr = kart[:, :, :3].astype(np.float32)
        gri = bgr[:, :, 0] * 0.114 + bgr[:, :, 1] * 0.587 + bgr[:, :, 2] * 0.299

    adim_y = max(1, int(np.ceil(gri.shape[0] / 360.0)))
    adim_x = max(1, int(np.ceil(gri.shape[1] / 260.0)))
    ornek = gri[::adim_y, ::adim_x]
    h, w = ornek.shape
    if h < 40 or w < 40:
        return False, metrikler

    bant = ornek[max(0, int(h * 0.01)):max(1, int(h * 0.14)), :]
    govde = ornek[int(h * 0.17):max(int(h * 0.18) + 1, int(h * 0.34)), :]
    if bant.size == 0 or govde.size == 0:
        return False, metrikler

    p10, p90 = np.percentile(ornek, (10, 90))
    koyu_esik = float(p10 + max(18.0, (p90 - p10) * 0.34))
    bant_koyu = bant <= koyu_esik
    govde_koyu = govde <= koyu_esik
    parcalar = np.array_split(bant_koyu, 4, axis=1)
    parca_oranlari = [float(np.mean(parca)) for parca in parcalar if parca.size]
    bant_orani = float(np.mean(bant_koyu))
    govde_orani = float(np.mean(govde_koyu))
    ortalama_farki = float(np.mean(govde) - np.mean(bant))

    # Preflight bilincli olarak siki: koyu bant genisligin tamaminda bulunmali
    # ve alttaki beyaz bilgi govdesinden belirgin bicimde koyu olmali. Tablo
    # tarafinda da en az yedi ayri yatay satir istenir.
    gecerli = (
        tablo.get("yatay_cizgi_grubu", 0) >= 7
        and tablo.get("yatay_grid_yayilimi", 0.0) >= 0.32
        and tablo.get("deger_ayirici_kapsami", 0.0) >= 0.30
        and bant_orani >= 0.38
        and (min(parca_oranlari) if parca_oranlari else 0.0) >= 0.24
        and (bant_orani - govde_orani >= 0.12 or ortalama_farki >= 18.0)
    )
    metrikler.update({
        "ust_bant_koyu_orani": bant_orani,
        "ust_bant_asgari_parca_orani": min(parca_oranlari) if parca_oranlari else 0.0,
        "govde_koyu_orani": govde_orani,
        "ust_bant_ortalama_farki": ortalama_farki,
        "gecerli": bool(gecerli),
    })
    return bool(gecerli), metrikler


def _gocmen_tek_form_uzamsal_imzasi_gecerli_mi(kart):
    """Yan yana kartlari tek gocmen on formundan ayiran ucuz uzamsal kapi."""
    form_ok, form_metrikleri = _gocmen_hizli_form_imzasi_gecerli_mi(kart)
    tablo = form_metrikleri.get("tablo", {})
    ortak_grup = int(tablo.get("ortak_yatay_cizgi_grubu", 0))
    ortak_oran = float(tablo.get("ortak_yatay_cizgi_orani", 0.0))
    ortak_yayilim = float(tablo.get("ortak_yatay_grid_yayilimi", 0.0))
    gecerli = bool(
        form_ok
        and ortak_grup >= 3
        and ortak_oran >= 0.24
        and ortak_yayilim >= 0.18
    )
    return gecerli, {
        "form": form_metrikleri,
        "ortak_yatay_cizgi_grubu": ortak_grup,
        "ortak_yatay_cizgi_orani": ortak_oran,
        "ortak_yatay_grid_yayilimi": ortak_yayilim,
        "gecerli": gecerli,
    }


def _gocmen_hizli_on_tespit_dene(resim):
    """Acik bir gocmen formunda SIFT'e girmeden kontur + layout ile tespit."""
    sonuc = {
        "basarili": False,
        "kart": None,
        "koseler": None,
        "kaynak": None,
        "kontur_metrikleri": {},
        "form_metrikleri": {},
        "deneme_sayisi": 0,
        "siki_aday_sayisi": 0,
        "dogrulanan_aday_sayisi": 0,
        "hata": "",
    }
    adaylar, kontur_metrikleri, hata = _gocmen_kontur_kose_adaylari(
        resim, azami_aday=4
    )
    sonuc["kontur_metrikleri"] = kontur_metrikleri
    sonuc["hata"] = hata
    dogrulananlar = []
    for aday in adaylar:
        # Kenar zarfi ve minAreaRect, SIFT sonrasi bilinen gocmen tipini
        # kurtarmakta kullanilabilir; fakat tek basina belge siniflandirmasi
        # yapacak kadar ozgul degildir. SIFT oncesi yalnizca gercek, kapali ve
        # konveks dortgen konturu kabul edilir.
        if aday.get("kaynak") != "dortgen_kontur":
            continue
        aday_koseleri = np.asarray(aday.get("koseler"), dtype=np.float64)
        if aday_koseleri.shape != (4, 2) or not np.all(np.isfinite(aday_koseleri)):
            continue
        sekil = _gocmen_kontur_aday_metrikleri(aday_koseleri, resim.shape)
        alan = _cokgen_alani(aday_koseleri)
        gorunen = _goruntu_icindeki_cokgen_alani(
            aday_koseleri,
            resim.shape[1], resim.shape[0],
        ) / max(1.0, alan)
        if not (
            0.12 <= sekil["alan_orani"] <= 1.01
            and 0.54 <= sekil["en_boy_orani"] <= 0.76
            and gorunen >= 0.95
        ):
            continue
        sonuc["siki_aday_sayisi"] += 1
        sonuc["deneme_sayisi"] += 1
        deneme = _koselerle_karti_duzeltmeyi_dene(
            resim, aday["koseler"], "gocmen"
        )
        if not deneme["basarili"]:
            sonuc["hata"] = deneme["hata"]
            continue
        imza_ok, form_metrikleri = _gocmen_tek_form_uzamsal_imzasi_gecerli_mi(
            deneme["kart"]
        )
        sonuc["form_metrikleri"] = form_metrikleri
        if not imza_ok:
            sonuc["hata"] = "Tek gocmen formunun uzamsal imzasi dogrulanamadi."
            continue
        dogrulananlar.append({
            "basarili": True,
            "kart": deneme["kart"],
            "koseler": aday["koseler"],
            "kaynak": aday.get("kaynak"),
            "kalite_metrikleri": deneme["kalite_metrikleri"],
            "form_metrikleri": form_metrikleri,
            "hata": "",
        })
    sonuc["dogrulanan_aday_sayisi"] = len(dogrulananlar)
    if len(dogrulananlar) == 1:
        sonuc.update(dogrulananlar[0])
    elif len(dogrulananlar) > 1:
        sonuc["hata"] = "Birden fazla gocmen konturu dogrulandi; SIFT gerekli."
    elif sonuc["siki_aday_sayisi"] == 0:
        sonuc["hata"] = "SIFT oncesi guvenilir kapali gocmen dortgeni bulunamadi."
    return sonuc


def kart_tespit_et_ve_duzelt(resim, debug_kart=False):
    baslangic = time.perf_counter()
    sonuc = {
        "basarili": False, "kart": None, "belge_tipi": None, "debug_resmi": None,
        "koseler": None, "fallback": False, "duzeltildi": False,
        "iyi_eslesme": 0, "inlier": 0, "inlier_orani": 0.0, "skor": 0.0, "mesaj": "",
        "referans_hatalari": {}, "hatalar": {}, "aday_sonuclari": [],
        "aday_sirasi": None, "debug_hatasi": "", "kalite_metrikleri": {},
        "duzeltme_yontemi": None, "tespit_suresi": 0.0,
        "on_tespit_metrikleri": {},
    }

    resim, girdi_hatasi = _girdi_goruntusunu_hazirla(resim)
    if girdi_hatasi:
        sonuc["mesaj"] = girdi_hatasi
        sonuc["hatalar"]["girdi"] = girdi_hatasi
        return _tespit_suresini_tamamla(sonuc, baslangic)

    # Net ve standart gocmen formlarinda en pahali adim olan SIFT'e gerek yok.
    # Yanlis pozitif riskini dusurmek icin kontur; genel kalite, cok satirli
    # tablo, orta ayirici ve tum genislige yayilan koyu ust bantla dogrulanir.
    try:
        on_tespit = _gocmen_hizli_on_tespit_dene(resim)
    except Exception as exc:
        on_tespit = {"basarili": False, "hata": f"On tespit calistirilamadi: {exc}"}
    sonuc["on_tespit_metrikleri"] = {
        anahtar: deger
        for anahtar, deger in on_tespit.items()
        if anahtar not in {"kart", "koseler"}
    }
    if on_tespit.get("basarili", False):
        koseler = on_tespit["koseler"]
        sonuc.update({
            "basarili": True,
            "kart": on_tespit["kart"],
            "belge_tipi": "gocmen",
            "koseler": koseler,
            "fallback": False,
            "duzeltildi": True,
            "aday_sirasi": 1,
            "kalite_metrikleri": on_tespit.get("kalite_metrikleri", {}),
            "duzeltme_yontemi": "kontur_on_tespit",
            "mesaj": "Gocmen karti hizli cerceve ve tablo imzasiyla tespit edildi.",
        })
        sonuc["aday_sonuclari"].append({
            "sira": 1,
            "belge_tipi": "gocmen",
            "basarili": True,
            "geometri_gecerli": True,
            "cikti_kalitesi_gecerli": True,
            "toleransli": False,
            "yontem": "kontur_on_tespit",
            "denenen_yontemler": ["kontur_on_tespit"],
            "kontur_kaynak": on_tespit.get("kaynak"),
            "kontur_metrikleri": on_tespit.get("kontur_metrikleri", {}),
            "kontur_tablo_metrikleri": on_tespit.get("form_metrikleri", {}),
            "kalite_metrikleri": on_tespit.get("kalite_metrikleri", {}),
            "hata": "",
        })
        if debug_kart:
            try:
                sonuc["debug_resmi"] = kart_debug_resmi(resim, koseler)
            except Exception as exc:
                debug_hatasi = f"Debug gorseli uretilemedi: {exc}"
                sonuc["debug_hatasi"] = debug_hatasi
                sonuc["hatalar"]["debug"] = debug_hatasi
        return _tespit_suresini_tamamla(sonuc, baslangic)

    try:
        tip_sonuc = belge_tipini_bul(resim)
    except Exception as exc:
        hata = f"Belge tipi tespiti çalıştırılamadı: {exc}"
        sonuc["mesaj"] = hata
        sonuc["hatalar"]["tespit"] = hata
        return _tespit_suresini_tamamla(sonuc, baslangic)

    referans_hatalari = _referans_hatalarini_al(tip_sonuc)
    sonuc["referans_hatalari"] = referans_hatalari
    sonuc["hatalar"].update({f"referans_{tip}": hata for tip, hata in referans_hatalari.items()})

    if not tip_sonuc.get("basarili", False):
        if referans_hatalari:
            ayrinti = " | ".join(referans_hatalari.values())
            sonuc["mesaj"] = f"Belge tipi bulunamadı. Referans hataları: {ayrinti}"
        else:
            sonuc["mesaj"] = "Belge tipi bulunamadı."
        return _tespit_suresini_tamamla(sonuc, baslangic)

    adaylar = tip_sonuc.get("adaylar") or [tip_sonuc["en_iyi"]]
    ilk_aday = adaylar[0]

    sonuc.update({
        "belge_tipi": ilk_aday["belge_tipi"],
        "iyi_eslesme": ilk_aday.get("iyi_eslesme", 0),
        "inlier": ilk_aday.get("inlier", 0),
        "inlier_orani": ilk_aday.get("inlier_orani", 0.0),
        "skor": ilk_aday.get("skor", 0.0),
    })

    for sira, aday in enumerate(adaylar, start=1):
        belge_tipi = aday["belge_tipi"]
        aday_sonucu = {
            "sira": sira, "belge_tipi": belge_tipi, "skor": aday.get("skor", 0.0),
            "basarili": False, "geometri_gecerli": False, "toleransli": False,
            "cikti_kalitesi_gecerli": False, "kalite_metrikleri": {}, "hata": "",
            "yontem": None, "denenen_yontemler": ["homografi"],
            "homografi_hatasi": "", "homografi_geometri_gecerli": False,
            "homografi_cikti_kalitesi_gecerli": False,
            "affine_hatasi": "", "affine_metrikleri": {},
            "affine_geometri_gecerli": False,
            "affine_cikti_kalitesi_gecerli": False,
            "affine_kalite_metrikleri": {},
            "affine_tablo_imzasi_gecerli": False,
            "affine_tablo_metrikleri": {},
            "kontur_hatasi": "", "kontur_metrikleri": {},
            "kontur_aday_sayisi": 0, "kontur_deneme_sayisi": 0,
            "kontur_kaynak": None, "kontur_geometri_gecerli": False,
            "kontur_cikti_kalitesi_gecerli": False,
            "kontur_kalite_metrikleri": {},
            "kontur_tablo_imzasi_gecerli": False,
            "kontur_tablo_metrikleri": {},
            "kontur_form_imzasi_gecerli": False,
            "kontur_form_metrikleri": {},
        }
        sonuc["aday_sonuclari"].append(aday_sonucu)

        olcek = tip_sonuc["olcek"]
        koseler = None
        try:
            koseler = koseleri_bul(aday["referans"], aday["H"])
            if olcek != 1.0:
                koseler = koseler / olcek
        except Exception as exc:
            aday_sonucu["homografi_hatasi"] = f"Koseler hesaplanamadi: {exc}"

        homografi_deneme = None
        if koseler is not None:
            homografi_deneme = _koselerle_karti_duzeltmeyi_dene(
                resim, koseler, belge_tipi
            )
            aday_sonucu["homografi_geometri_gecerli"] = homografi_deneme[
                "geometri_gecerli"
            ]
            aday_sonucu["homografi_cikti_kalitesi_gecerli"] = homografi_deneme[
                "cikti_kalitesi_gecerli"
            ]
            aday_sonucu["geometri_gecerli"] = homografi_deneme["geometri_gecerli"]
            aday_sonucu["cikti_kalitesi_gecerli"] = homografi_deneme[
                "cikti_kalitesi_gecerli"
            ]
            aday_sonucu["kalite_metrikleri"] = homografi_deneme["kalite_metrikleri"]
            aday_sonucu["homografi_hatasi"] = homografi_deneme["hata"]

        secilen_deneme = (
            homografi_deneme
            if homografi_deneme is not None and homografi_deneme["basarili"]
            else None
        )
        secilen_yontem = "homografi" if secilen_deneme is not None else None

        # Basarili homografi mutlu yolu birebir ayni kalir. Bu ucuz yedek
        # yalnizca gocmen adayinin geometri/warp kalite kapisinda elenmesiyle
        # calisir ve mevcut keypoint eslesmelerini yeniden kullanir.
        if secilen_deneme is None and belge_tipi == "gocmen":
            aday_sonucu["denenen_yontemler"].append("affine_partial")
            affine_koseler, affine_metrikleri, affine_hatasi = _gocmen_affine_koseleri_bul(
                aday, olcek, resim.shape
            )
            aday_sonucu["affine_metrikleri"] = affine_metrikleri
            aday_sonucu["affine_hatasi"] = affine_hatasi

            if affine_koseler is not None:
                affine_deneme = _koselerle_karti_duzeltmeyi_dene(
                    resim, affine_koseler, belge_tipi
                )
                aday_sonucu["affine_geometri_gecerli"] = affine_deneme[
                    "geometri_gecerli"
                ]
                aday_sonucu["affine_cikti_kalitesi_gecerli"] = affine_deneme[
                    "cikti_kalitesi_gecerli"
                ]
                aday_sonucu["affine_kalite_metrikleri"] = affine_deneme[
                    "kalite_metrikleri"
                ]
                aday_sonucu["affine_hatasi"] = affine_deneme["hata"]
                if affine_deneme["basarili"]:
                    tablo_ok, tablo_metrikleri = _gocmen_tablo_imzasi_gecerli_mi(
                        affine_deneme["kart"]
                    )
                    aday_sonucu["affine_tablo_imzasi_gecerli"] = bool(tablo_ok)
                    aday_sonucu["affine_tablo_metrikleri"] = tablo_metrikleri
                    if not tablo_ok:
                        aday_sonucu["affine_hatasi"] = (
                            "Gocmen tablo imzasi guvenilir degil."
                        )
                    else:
                        secilen_deneme = affine_deneme
                        secilen_yontem = "affine_partial"
                        koseler = affine_koseler
                        aday_sonucu["geometri_gecerli"] = True
                        aday_sonucu["cikti_kalitesi_gecerli"] = True
                        aday_sonucu["kalite_metrikleri"] = affine_deneme[
                            "kalite_metrikleri"
                        ]

        # Tekrarlanan tablo cizgileri SIFT homografisini, perspektif de
        # partial-affine inlier oranini bozabilir. Son care olan bu yol OCR ya
        # da yeni descriptor calistirmaz; en fazla uc ucuz cerceve adayi dener.
        # Yalnizca tablo cizgileri yeterli degildir: on ve arka karti birlikte
        # iceren PDF sayfasi da soldaki tablodan bu yumusak imzayi uretebilir.
        # Bu nedenle aday; koyu ust bant ve secilen ayiricinin iki yaninda ayni
        # grid satirlarinin devamini arayan tek-form imzasini gecmeden secilmez.
        # Gecmeyen adaydan sonra gercek on kart adayi denenir.
        if secilen_deneme is None and belge_tipi == "gocmen":
            aday_sonucu["denenen_yontemler"].append("kontur_tablo")
            kontur_adaylari, kontur_metrikleri, kontur_hatasi = (
                _gocmen_kontur_kose_adaylari(resim, azami_aday=3)
            )
            aday_sonucu["kontur_metrikleri"] = kontur_metrikleri
            aday_sonucu["kontur_hatasi"] = kontur_hatasi
            aday_sonucu["kontur_aday_sayisi"] = len(kontur_adaylari)

            son_kontur_hatasi = kontur_hatasi
            for kontur_adayi in kontur_adaylari:
                aday_sonucu["kontur_deneme_sayisi"] += 1
                kontur_koseler = kontur_adayi["koseler"]
                kontur_deneme = _koselerle_karti_duzeltmeyi_dene(
                    resim, kontur_koseler, belge_tipi
                )
                aday_sonucu["kontur_geometri_gecerli"] = bool(
                    aday_sonucu["kontur_geometri_gecerli"]
                    or kontur_deneme["geometri_gecerli"]
                )
                aday_sonucu["kontur_cikti_kalitesi_gecerli"] = bool(
                    aday_sonucu["kontur_cikti_kalitesi_gecerli"]
                    or kontur_deneme["cikti_kalitesi_gecerli"]
                )
                aday_sonucu["kontur_kalite_metrikleri"] = kontur_deneme[
                    "kalite_metrikleri"
                ]
                if not kontur_deneme["basarili"]:
                    son_kontur_hatasi = kontur_deneme["hata"]
                    continue

                form_ok, form_metrikleri = _gocmen_tek_form_uzamsal_imzasi_gecerli_mi(
                    kontur_deneme["kart"]
                )
                temel_form_metrikleri = form_metrikleri.get("form", {})
                tablo_metrikleri = temel_form_metrikleri.get("tablo", {})
                tablo_ok = bool(tablo_metrikleri.get("gecerli", False))
                aday_sonucu["kontur_tablo_imzasi_gecerli"] = bool(tablo_ok)
                aday_sonucu["kontur_tablo_metrikleri"] = tablo_metrikleri
                aday_sonucu["kontur_form_imzasi_gecerli"] = bool(form_ok)
                aday_sonucu["kontur_form_metrikleri"] = form_metrikleri
                if not form_ok:
                    son_kontur_hatasi = "Gocmen tam form imzasi guvenilir degil."
                    continue

                secilen_deneme = kontur_deneme
                secilen_yontem = "kontur_tablo"
                koseler = kontur_koseler
                aday_sonucu["kontur_kaynak"] = kontur_adayi.get("kaynak")
                aday_sonucu["geometri_gecerli"] = True
                aday_sonucu["cikti_kalitesi_gecerli"] = True
                aday_sonucu["kalite_metrikleri"] = kontur_deneme[
                    "kalite_metrikleri"
                ]
                son_kontur_hatasi = ""
                break

            aday_sonucu["kontur_hatasi"] = son_kontur_hatasi

        if secilen_deneme is None:
            nedenler = []
            if aday_sonucu["homografi_hatasi"]:
                nedenler.append(f"Homografi: {aday_sonucu['homografi_hatasi']}")
            if aday_sonucu["affine_hatasi"]:
                nedenler.append(f"Affine: {aday_sonucu['affine_hatasi']}")
            if aday_sonucu["kontur_hatasi"]:
                nedenler.append(f"Kontur: {aday_sonucu['kontur_hatasi']}")
            aday_sonucu["hata"] = " | ".join(nedenler) or "Kart duzeltilemedi."
            continue

        kart = secilen_deneme["kart"]
        toleransli = secilen_yontem != "homografi"
        aday_sonucu["yontem"] = secilen_yontem
        aday_sonucu.update({"basarili": True, "toleransli": toleransli})
        alternatif_aday = sira > 1
        fallback = alternatif_aday or toleransli

        sonuc.update({
            "kart": kart, "basarili": True, "duzeltildi": True, "fallback": fallback,
            "belge_tipi": belge_tipi, "koseler": koseler, "aday_sirasi": sira,
            "iyi_eslesme": aday.get("iyi_eslesme", 0), "inlier": aday.get("inlier", 0),
            "inlier_orani": aday.get("inlier_orani", 0.0), "skor": aday.get("skor", 0.0),
            "kalite_metrikleri": aday_sonucu["kalite_metrikleri"],
            "duzeltme_yontemi": secilen_yontem,
        })

        if alternatif_aday:
            sonuc["mesaj"] = (
                f"En yüksek skorlu {ilk_aday['belge_tipi']} adayı düzeltilemedi; "
                f"{sira}. sıradaki {belge_tipi} adayıyla kart tespit edildi ve yönü düzeltildi."
            )
        else:
            if secilen_yontem == "kontur_tablo":
                sonuc["mesaj"] = (
                    "Referans donusumu guvenilir sonuc vermedi; gocmen karti "
                    "cerceve ve tablo yapisiyla tespit edilip duzeltildi."
                )
            elif secilen_yontem == "affine_partial":
                sonuc["mesaj"] = (
                    "Homografi güvenilir sonuç vermedi; göçmen kartı sıkıca "
                    "doğrulanan affine yedeğiyle tespit edildi ve yönü düzeltildi."
                )
            else:
                sonuc["mesaj"] = "Kart tespit edildi ve yönü düzeltildi."

        # Debug cizimi asil kart sonucundan tamamen bagimsizdir. Cizim hatasi
        # basarili tespit/warp sonucunu gecersiz kilmaz.
        if debug_kart:
            try:
                sonuc["debug_resmi"] = kart_debug_resmi(resim, koseler)
            except Exception as exc:
                debug_hatasi = f"Debug görseli üretilemedi: {exc}"
                sonuc["debug_hatasi"] = debug_hatasi
                sonuc["hatalar"]["debug"] = debug_hatasi

        return _tespit_suresini_tamamla(sonuc, baslangic)

    sonuc["mesaj"] = (
        f"Belge tipi eşleşmesi bulundu ({ilk_aday['belge_tipi']}), "
        "ancak hiçbir aday güvenilir şekilde düzeltilemedi."
    )
    return _tespit_suresini_tamamla(sonuc, baslangic)
