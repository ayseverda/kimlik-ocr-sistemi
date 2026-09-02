import cv2
import numpy as np
import os
import sys
import time


# =========================================================
# SABİTLER
# =========================================================

MAX_CALISMA_BOYUTU = 1300
SIFT_AZAMI_OZELLIK = 3200

# Hiç kart bulunamayan sayfalar için "zor mod": aynı akış, ama çok daha
# geniş özellik bütçesiyle. Pahalı olduğu için SADECE boş dönen sayfalarda
# (ölçümde sayfaların ~%4'ü) devreye girer.
SIFT_ZOR_MOD_OZELLIK = 9000

# Bir sayfada birden fazla kimlik olabilir (ör. tek A4'e basılmış 2-3 kart).
# Aynı sayfada en fazla kaç kart aranacağı ve kabul edilmiş bir kartla bu
# orandan fazla örtüşen adayın (aynı kartın ikinci kez bulunması) elenmesi.
AZAMI_KART_SAYISI = 8
KART_ORTUSME_ESIGI = 0.30

# Tum sayfa taraması bitince, sayfada kart olabilecek ama henüz kapsınmamış
# bloklara YAKINDAN bakılır (blok kırpılıp tek başına aranır). Küçük kartlar
# sayfanın tamamı küçültüldüğünde çok az özellik bırakıyor; kırpılmış blokta
# aynı kart tam çözünürlükte ve çok daha yoğun özellikle taranıyor.
# Bir turda en iyi adayin homografisi geometri kontrolunu geceMEZse, o
# eslesmeler atilip tur tekrarlanir (sequential RANSAC). Yan yana duran iki
# kartin ozelliklerini karistiran bozuk bir homografi, eskiden o bolgedeki
# TUM kartlarin kaybina yol aciyordu; artik sadece kendi eslesmelerini goturur.
AZAMI_BASARISIZ_ELEME = 2

AZAMI_ADAY_BOLGE = 5
ADAY_BOLGE_ASGARI_ALAN = 0.012      # sayfa alanına oran
ADAY_BOLGE_AZAMI_ALAN = 0.90
ADAY_BOLGE_ASGARI_KENAR = 130       # piksel (tam çözünürlükte)
ADAY_BOLGE_AZAMI_ORAN = 4.5         # en/boy
ADAY_BOLGE_KENAR_PAYI = 0.08

def _modul_dizini_bul():
    """PyInstaller ile .exe'ye paketlendiğinde __file__ güvenilir bir yol
    vermeyebiliyor (donmuş modüllerde gerçek bir dosya olarak var olmaz).
    sys.frozen + sys._MEIPASS, PyInstaller'ın resmi olarak önerdiği,
    "gerçekten çalıştırılabilir dosyanın veri klasörü neresi" sorusunun
    standart cevabı. Normal (donmuş olmayan) çalıştırmada eskisi gibi
    __file__'a dayanır — davranış hiç değişmiyor."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


MODUL_DIZINI = _modul_dizini_bul()

# =========================================================
# BELGE TİPİ KONFİGÜRASYONU
# =========================================================
#
# Eskiden TC için tek temiz bir yol (homografi -> geometri kontrolü -> warp),
# göçmen için ONA EK olarak 3 fazla katman (affine yedek, kontur+tablo-imza
# tespiti, kontur-ön-tespit — SIFT'i tamamen atlayan ayrı bir yol) ve toplam
# 5 farklı kalite metriği (radyal yakınsama, tablo çizgisi tespiti, baskın
# ton oranı, vb. ~700 satır) vardı. Üç belge tipi de şimdi AYNI tek pipeline'ı
# (kartlari_tespit_et_ve_duzelt) kullanıyor; farklar sadece bu sözlükte.

BELGE_KONFIG = {
    "tc": {
        "referans": "referans_kimlik.jpg",
        "hedef_w": 1000, "hedef_h": 630,
        "min_inlier": 8, "min_inlier_orani": 0.0,
        "aspect_min": 1.15, "aspect_max": 2.20,
    },
    "eski_tc": {
        "referans": "referans_eski_tc.jpg",
        "hedef_w": 750, "hedef_h": 1050,
        "min_inlier": 9, "min_inlier_orani": 0.18,
        "aspect_min": 1.05, "aspect_max": 2.10,
    },
    "gocmen": {
        "referans": "referans_gocmen.jpg",
        "hedef_w": 720, "hedef_h": 1100,
        "min_inlier": 10, "min_inlier_orani": 0.12,
        "aspect_min": 1.00, "aspect_max": 3.30,
    },
}


_DETECTORLER = {}
_CLAHE = None
_FLANN = None
_REFERANS_CACHE = {}


def referans_yolunu_coz(yol):
    """Mutlak olmayan referans yollarını modül dizinine göre çözer."""
    yol = os.fspath(yol)
    if not os.path.isabs(yol):
        yol = os.path.join(MODUL_DIZINI, yol)
    return os.path.abspath(yol)


# =========================================================
# DETECTOR / CLAHE / MATCHER
# =========================================================

def detector_getir(mod="normal"):
    """mod="normal" günlük akış; mod="zor" aynı detector'ün çok daha geniş
    özellik bütçeli sürümü (yalnızca hiçbir kart bulunamayan sayfalarda).
    İkisi de ayrı ayrı cache'lenir."""
    global _DETECTORLER

    if mod in _DETECTORLER:
        return _DETECTORLER[mod]

    ozellik = SIFT_ZOR_MOD_OZELLIK if mod == "zor" else SIFT_AZAMI_OZELLIK
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(
            nfeatures=ozellik, contrastThreshold=0.018, edgeThreshold=12, sigma=1.6,
        )
        tipi = "SIFT"
    else:
        detector = cv2.ORB_create(
            nfeatures=max(7000, ozellik), scaleFactor=1.2, nlevels=10, fastThreshold=5
        )
        tipi = "ORB"

    _DETECTORLER[mod] = (detector, tipi)
    return _DETECTORLER[mod]


def clahe_getir():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def matcher_getir(detector_tipi):
    global _FLANN
    if detector_tipi == "SIFT":
        if _FLANN is None:
            _FLANN = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 40})
        return _FLANN
    return cv2.BFMatcher(cv2.NORM_HAMMING)


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

    mtime = os.path.getmtime(abs_yol)
    cache = _REFERANS_CACHE.get(belge_tipi)
    if cache is not None and cache.get("yol") == abs_yol and cache.get("mtime") == mtime:
        return cache["resim"], cache["kp"], cache["des"], None

    try:
        # cv2.imread(yol) Windows'ta, yolda Türkçe/Unicode karakter varsa
        # (ör. "Masaüstü") dosya gerçekten var olsa bile SESSİZCE None
        # döndürüyor — bilinen bir OpenCV/Windows sınırlaması. Dosyayı
        # Python'ın kendi (Unicode-güvenli) okuma fonksiyonuyla byte olarak
        # alıp OpenCV'ye sadece bellekten çözdürmek bu sorunu ortadan kaldırır.
        veri = np.fromfile(abs_yol, dtype=np.uint8)
        resim = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    except Exception as exc:
        return None, None, None, f"Referans resmi okunamadı ({belge_tipi}): {abs_yol} ({exc})"
    if resim is None:
        return None, None, None, f"Referans resmi okunamadı ({belge_tipi}): {abs_yol}"

    gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
    detector, _ = detector_getir("normal")
    kp, des = detector.detectAndCompute(gri, None)
    if des is None:
        return None, None, None, f"Referansta özellik bulunamadı ({belge_tipi}): {abs_yol}"

    _REFERANS_CACHE[belge_tipi] = {"yol": abs_yol, "mtime": mtime, "resim": resim, "kp": kp, "des": des}
    return resim, kp, des, None


# =========================================================
# ÇALIŞMA GÖRÜNTÜSÜNÜN DESCRIPTOR'LARI (SAYFA BAŞINA 1 KEZ)
# =========================================================
#
# Aynı fotoğraf için detectAndCompute() 3 referans için 3 kere değil, TEK
# sefer çalışır; 3 referansın hepsi bu descriptor'ları kullanır (~2.5x hız —
# daha önce sentetik veriyle doğrulandı).

def resim_descriptor_cikar(resim, mod="normal"):
    detector, detector_tipi = detector_getir(mod)
    gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
    kp, des = detector.detectAndCompute(gri, None)
    return kp, des, detector_tipi


def referansla_eslestir(kp_ref, des_ref, kp_resim, des_resim, detector_tipi):
    """Doner: (H, iyi_eslesmeler, inlier_sayisi, inlier_keypoint_indeksleri).

    Son deger, homografiyi ureten goruntu keypoint'lerinin indeksleridir;
    aday reddedilirse tam olarak bu keypoint'ler atilip tur tekrarlanabilsin
    diye tutuluyor."""
    if des_resim is None:
        return None, [], 0, []

    matcher = matcher_getir(detector_tipi)
    try:
        raw = matcher.knnMatch(des_ref, des_resim, k=2)
    except cv2.error:
        return None, [], 0, []

    ratio = 0.74 if detector_tipi == "SIFT" else 0.80
    iyi = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(iyi) < 8:
        return None, iyi, 0, []

    src = np.float32([kp_ref[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2)
    dst = np.float32([kp_resim[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, iyi, 0, []

    duz = mask.ravel().astype(bool)
    inlier_idx = [m.trainIdx for m, secili in zip(iyi, duz) if secili]
    return H, iyi, int(duz.sum()), inlier_idx


def eslesme_skoru(iyi, inlier):
    if iyi <= 0:
        return 0.0
    return inlier * 3.5 + iyi * 0.30 + (inlier / iyi) * 100


def tek_referansi_test_et(belge_tipi, kp_resim, des_resim, detector_tipi):
    konfig = BELGE_KONFIG[belge_tipi]
    referans, kp_ref, des_ref, hata = referansi_hazirla(belge_tipi, konfig["referans"])
    if hata:
        return {"belge_tipi": belge_tipi, "basarili": False, "skor": 0.0, "hata": hata}

    H, iyi, inlier, inlier_idx = referansla_eslestir(
        kp_ref, des_ref, kp_resim, des_resim, detector_tipi
    )
    iyi_sayisi = len(iyi)
    inlier_orani = inlier / iyi_sayisi if iyi_sayisi else 0.0
    basarili = H is not None and inlier >= konfig["min_inlier"] and inlier_orani >= konfig["min_inlier_orani"]

    return {
        "belge_tipi": belge_tipi, "basarili": basarili, "referans": referans, "H": H,
        "iyi_eslesme": iyi_sayisi, "inlier": inlier, "inlier_orani": inlier_orani,
        "inlier_idx": inlier_idx,
        "skor": eslesme_skoru(iyi_sayisi, inlier), "detector": detector_tipi,
    }


def calisma_goruntusu(resim):
    """Büyük fotoğrafları feature çıkarımı için küçültür; (görüntü, ölçek)."""
    h, w = resim.shape[:2]
    if max(h, w) > MAX_CALISMA_BOYUTU:
        olcek = MAX_CALISMA_BOYUTU / max(h, w)
        return cv2.resize(resim, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA), olcek
    return resim, 1.0


def adaylari_sirala(kp_resim, des_resim, detector_tipi):
    """Eldeki keypoint kümesini 3 referansla da dener, skora göre sıralar.
    Çok kimlikli sayfada bu fonksiyon her turda yeniden çağrılır — her tur
    bir önceki turda bulunan kartın keypoint'leri çıkarılmış olarak gelir."""
    adaylar = [
        tek_referansi_test_et(tip, kp_resim, des_resim, detector_tipi)
        for tip in BELGE_KONFIG
    ]
    basarili_adaylar = sorted(
        (a for a in adaylar if a["basarili"]), key=lambda x: x["skor"], reverse=True
    )
    return basarili_adaylar, {a["belge_tipi"]: a.get("hata") for a in adaylar if a.get("hata")}


def belge_tipini_bul(resim, mod="normal"):
    calisma, olcek = calisma_goruntusu(resim)
    kp_resim, des_resim, detector_tipi = resim_descriptor_cikar(calisma, mod)
    basarili_adaylar, hatalar = adaylari_sirala(kp_resim, des_resim, detector_tipi)
    return basarili_adaylar, olcek, hatalar


# =========================================================
# GEOMETRİ + PERSPEKTİF (tek, genel — belge tipine özel dallanma yok,
# sadece BELGE_KONFIG'teki sayılar farklı)
# =========================================================

def koseleri_bul(referans, H):
    h, w = referans.shape[:2]
    ref_koseler = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ref_koseler, H).reshape(4, 2)


def koseler_gecerli_mi(koseler, shape, konfig):
    koseler = np.asarray(koseler, dtype=np.float32)
    if koseler.shape != (4, 2) or not np.all(np.isfinite(koseler)):
        return False

    h, w = shape[:2]
    kontur = koseler.astype(np.int32).reshape(-1, 1, 2)

    # Bow-tie/konkav dörtgen warp'ta görüntüyü ince bir şerit haline getirir;
    # cv2'nin kendi dışbükeylik kontrolü bunu ucuzca eler.
    if not cv2.isContourConvex(kontur):
        return False

    alan = abs(cv2.contourArea(kontur))
    if not 0.02 <= alan / max(1, h * w) <= 1.5:
        return False

    tol_x, tol_y = w * 0.35, h * 0.35
    if np.any(koseler[:, 0] < -tol_x) or np.any(koseler[:, 0] > w + tol_x):
        return False
    if np.any(koseler[:, 1] < -tol_y) or np.any(koseler[:, 1] > h + tol_y):
        return False

    tl, tr, br, bl = koseler
    ust, sag = np.linalg.norm(tr - tl), np.linalg.norm(br - tr)
    alt, sol = np.linalg.norm(br - bl), np.linalg.norm(bl - tl)
    if min(ust, alt, sol, sag) < 20:
        return False

    genislik, yukseklik = (ust + alt) / 2.0, (sol + sag) / 2.0
    aspect = max(genislik, yukseklik) / min(genislik, yukseklik)
    return konfig["aspect_min"] <= aspect <= konfig["aspect_max"]


def perspektif_duzelt(resim, koseler, konfig):
    hedef_w, hedef_h = konfig["hedef_w"], konfig["hedef_h"]
    src = np.asarray(koseler, dtype=np.float32)
    dst = np.float32([[0, 0], [hedef_w - 1, 0], [hedef_w - 1, hedef_h - 1], [0, hedef_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        resim, M, (hedef_w, hedef_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def cikti_kaliteli_mi(kart):
    """Warp çıktısının çökmüş/dejenere olmadığını (büyük tek renk alan +
    ince çizgisel şerit değil) tek, ucuz bir kontrolle doğrular. Eskiden
    beş ayrı metrik (radyal yakınsama, tablo imzası, baskın ton oranı,
    hücre-bazlı ayrıntı kapsamı...) vardı; burada tek bir std/dinamik-aralık
    kontrolü var — çok daha az kod, "çökmüş warp"ı yakalamak için yeterli."""
    if kart is None or kart.size == 0:
        return False
    gri = cv2.cvtColor(kart, cv2.COLOR_BGR2GRAY) if kart.ndim == 3 else kart
    ornek = gri[::4, ::4]  # ucuz olsun diye seyrek örnekle
    if ornek.size == 0:
        return False
    p05, p95 = np.percentile(ornek, (5, 95))
    return (p95 - p05) >= 12.0 and float(np.std(ornek)) >= 4.0


# =========================================================
# DEBUG (yalnızca kart sınırı — basit, ucuz)
# =========================================================

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
# GİRDİ DOĞRULAMA
# =========================================================

def _girdi_hazirla(resim):
    if resim is None:
        return None, "Görüntü bulunamadı."
    if not isinstance(resim, np.ndarray) or resim.size == 0:
        return None, "Görüntü geçersiz veya boş."
    if resim.ndim not in (2, 3) or resim.shape[0] < 2 or resim.shape[1] < 2:
        return None, f"Geçersiz görüntü şekli: {resim.shape}."
    if resim.dtype != np.uint8:
        return None, f"Geçersiz görüntü veri tipi: {resim.dtype}; uint8 bekleniyor."

    if resim.ndim == 2:
        hazir = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    elif resim.shape[2] == 1:
        hazir = cv2.cvtColor(resim[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif resim.shape[2] == 3:
        hazir = resim
    elif resim.shape[2] == 4:
        hazir = cv2.cvtColor(resim, cv2.COLOR_BGRA2BGR)
    else:
        return None, f"Geçersiz kanal sayısı: {resim.shape[2]}."

    return np.ascontiguousarray(hazir), None


# =========================================================
# ÇOK KARTLI SAYFA YARDIMCILARI
# =========================================================

def dortgen_ortusme_orani(a, b):
    """İki dörtgenin kesişim alanının, küçük olanın alanına oranı.
    Aynı kartın ikinci bir referansla yeniden bulunmasını elemek için."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    try:
        kesisim_alani, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    if kesisim_alani <= 0:
        return 0.0
    kucuk = min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b)))
    return float(kesisim_alani / kucuk) if kucuk > 0 else 0.0


def ozellikleri_ele(kp, des, atilacak_indeksler):
    """Verilen indekslerdeki keypoint/descriptor'ları atar.
    Dönüş: (kalan_kp, kalan_des, eleme_oldu_mu)"""
    if des is None or not len(kp) or not len(atilacak_indeksler):
        return kp, des, False

    idx = np.asarray(sorted({int(i) for i in atilacak_indeksler}), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(kp))]
    if not len(idx):
        return kp, des, False

    tut = np.ones(len(kp), dtype=bool)
    tut[idx] = False
    if not tut.any():
        return [], None, True

    return [k for k, sec in zip(kp, tut) if sec], des[tut], True


def kartin_disindaki_ozellikler(kp, des, koseler):
    """Bulunan kartın İÇİNDE kalan keypoint'leri atar; geriye sayfanın geri
    kalanı kalır. Bir sonraki tur bu kalanla çalışır, böylece aynı kart
    tekrar tekrar bulunmaz ve ikinci/üçüncü kimlik ortaya çıkar.
    Dönüş: (kalan_kp, kalan_des, eleme_oldu_mu)"""
    if des is None or not len(kp):
        return kp, des, False

    kontur = np.asarray(koseler, dtype=np.float32).reshape(-1, 1, 2)
    icerideler = [i for i, nokta in enumerate(kp)
                  if cv2.pointPolygonTest(kontur, nokta.pt, False) >= 0]
    return ozellikleri_ele(kp, des, icerideler)


def sayfa_sirasina_diz(sonuclar):
    """Bulunan kartlari sayfadaki KONUMLARINA göre dizer: yukarıdan aşağıya,
    aynı hizadakiler soldan sağa. Tespit doğal olarak eşleşme SKORUNA göre
    olur (en net kart önce bulunur); kullanıcıya gösterilen sıra ise sayfayı
    gözle takip eden sıra olsun diye burada yeniden diziliyor.

    Aynı satırda mı kararı, kartların medyan yüksekliğinin yarısı kadar bir
    bant toleransıyla verilir; böylece hafif eğik taranmış sayfalarda yan yana
    duran iki kart yanlışlıkla alt alta sayılmaz."""
    if len(sonuclar) < 2:
        return sonuclar

    kutular = []
    for sonuc in sonuclar:
        koseler = np.asarray(sonuc["koseler"], dtype=np.float32)
        kutular.append({
            "sonuc": sonuc,
            "cy": float(koseler[:, 1].mean()),
            "cx": float(koseler[:, 0].mean()),
            "yukseklik": float(koseler[:, 1].max() - koseler[:, 1].min()),
        })

    tolerans = 0.5 * float(np.median([k["yukseklik"] for k in kutular]))

    bantlar = []
    for kutu in sorted(kutular, key=lambda k: k["cy"]):
        if bantlar and (kutu["cy"] - bantlar[-1]["cy"]) <= tolerans:
            bantlar[-1]["kutular"].append(kutu)
        else:
            bantlar.append({"cy": kutu["cy"], "kutular": [kutu]})

    return [
        kutu["sonuc"]
        for bant in bantlar
        for kutu in sorted(bant["kutular"], key=lambda k: k["cx"])
    ]


def _bos_sonuc(mesaj="", referans_hatalari=None):
    return {
        "basarili": False, "kart": None, "belge_tipi": None, "debug_resmi": None,
        "koseler": None, "iyi_eslesme": 0, "inlier": 0, "inlier_orani": 0.0,
        "skor": 0.0, "mesaj": mesaj, "referans_hatalari": referans_hatalari or {},
        "tespit_suresi": 0.0, "kart_sirasi": 1, "kart_sayisi": 0, "arama_modu": None,
        "kurtarma_bolgeleri": [],
    }


# =========================================================
# ADAY BÖLGELER (SAYFAYI YAKINDAN TARAMAK İÇİN)
# =========================================================

def _koseler_kutusu(koseler):
    k = np.asarray(koseler, dtype=np.float32).reshape(-1, 2)
    return (float(k[:, 0].min()), float(k[:, 1].min()),
            float(k[:, 0].max()), float(k[:, 1].max()))


def _kutu_ortusme_orani(kutu, digeri):
    """Kesişim / küçük kutunun alanı."""
    x0 = max(kutu[0], digeri[0]); y0 = max(kutu[1], digeri[1])
    x1 = min(kutu[2], digeri[2]); y1 = min(kutu[3], digeri[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    kesisim = (x1 - x0) * (y1 - y0)
    a = (kutu[2] - kutu[0]) * (kutu[3] - kutu[1])
    b = (digeri[2] - digeri[0]) * (digeri[3] - digeri[1])
    kucuk = min(a, b)
    return float(kesisim / kucuk) if kucuk > 0 else 0.0


def kart_adayi_bolgeler(resim, bulunanlar=(), azami=AZAMI_ADAY_BOLGE):
    """Sayfada "kart büyüklüğünde bir şey" duran, HENÜZ bulunmamış bölgeleri
    döndürür. Fotokopi sayfalarında kartlar beyaz zemin üzerinde ayrı ayrı
    lekeler oluşturur; Otsu eşiği + morfolojik kapama bu lekeleri tek parça
    haline getirir, findContours da her birini verir.

    Bu bir kart TESPİTİ değil, sadece "buraya yakından bak" önerisi: bulunan
    her bölge kırpılıp normal eşleştirme akışından geçirilir, kart olmayan
    bloklar (kimliğin arka yüzü, imza, kaşe...) orada elenir."""
    h, w = resim.shape[:2]
    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY) if resim.ndim == 3 else resim
    gri = cv2.GaussianBlur(gri, (5, 5), 0)

    try:
        _, ikili = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    except cv2.error:
        return []

    cekirdek = max(9, int(min(h, w) * 0.02) | 1)
    kapali = cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, np.ones((cekirdek, cekirdek), np.uint8))
    konturlar, _ = cv2.findContours(kapali, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bulunan_kutular = [_koseler_kutusu(s["koseler"]) for s in bulunanlar if s.get("koseler") is not None]
    sayfa_alani = float(max(1, h * w))
    adaylar = []

    for kontur in konturlar:
        x, y, bw, bh = cv2.boundingRect(kontur)
        if not (ADAY_BOLGE_ASGARI_ALAN <= (bw * bh) / sayfa_alani <= ADAY_BOLGE_AZAMI_ALAN):
            continue
        if min(bw, bh) < ADAY_BOLGE_ASGARI_KENAR:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > ADAY_BOLGE_AZAMI_ORAN:
            continue

        kutu = (float(x), float(y), float(x + bw), float(y + bh))
        if any(_kutu_ortusme_orani(kutu, b) > 0.5 for b in bulunan_kutular):
            continue

        pay = int(ADAY_BOLGE_KENAR_PAYI * max(bw, bh))

        def _payli(bx, by, bw2, bh2):
            return (max(0, bx - pay), max(0, by - pay),
                    min(w, bx + bw2 + pay), min(h, by + bh2 + pay))

        adaylar.append(_payli(x, y, bw, bh))

    adaylar.sort(key=lambda k: (k[2] - k[0]) * (k[3] - k[1]), reverse=True)
    return adaylar[:azami]


# =========================================================
# TEK PIPELINE (SAYFADA BİRDEN FAZLA KART)
# =========================================================

def _bolgede_kart_ara(arama_resmi, ofset, sayfa_resmi, sonuclar, azami_kart,
                      debug_kart, referans_hatalari, mod="normal", eleme_hakki=0):
    """arama_resmi üzerinde kart arar ve bulduklarını `sonuclar` listesine ekler.

    arama_resmi tüm sayfa da olabilir, sayfadan kırpılmış bir blok da; köşeler
    `ofset` ile sayfa koordinatına çevrilir ve perspektif düzeltme HER ZAMAN
    tam çözünürlüklü sayfa üzerinden yapılır (kırpma yüzünden kalite düşmez).

    Döndürdüğü değer: ilk turda bulunan aday listesi (hata mesajı için)."""
    calisma, olcek = calisma_goruntusu(arama_resmi)
    kp, des, detector_tipi = resim_descriptor_cikar(calisma, mod)
    ofset_vektoru = np.float32(ofset)
    ilk_adaylar = []
    basarisiz_eleme = 0
    bulunan_bu_bolgede = 0

    while len(sonuclar) < azami_kart:
        adaylar, hatalar = adaylari_sirala(kp, des, detector_tipi)
        referans_hatalari.update(hatalar)
        if not ilk_adaylar:
            ilk_adaylar = adaylar
        if not adaylar:
            break

        kabul = None
        for aday in adaylar:
            konfig = BELGE_KONFIG[aday["belge_tipi"]]

            koseler_calisma = koseleri_bul(aday["referans"], aday["H"])
            koseler_bolge = koseler_calisma / olcek if olcek != 1.0 else koseler_calisma

            # Geometri, ARANAN görüntüye göre doğrulanır (kırpılmış blokta kart
            # bloğun büyük kısmını kaplar; sayfaya göre bakılsa alan oranı çok
            # küçük kalırdı).
            if not koseler_gecerli_mi(koseler_bolge, arama_resmi.shape, konfig):
                continue

            koseler_sayfa = koseler_bolge + ofset_vektoru
            if any(dortgen_ortusme_orani(koseler_sayfa, s["koseler"]) > KART_ORTUSME_ESIGI
                   for s in sonuclar):
                continue

            kart = perspektif_duzelt(sayfa_resmi, koseler_sayfa, konfig)
            if not cikti_kaliteli_mi(kart):
                continue

            kabul = (aday, kart, koseler_sayfa, koseler_calisma)
            break

        if kabul is None:
            # Hicbir aday gecmedi. En iyi adayin homografisi bu bolgedeki
            # gercek bir kartla ortusmuyor demektir (or. iki kartin
            # ozelliklerini karistirmis). SADECE o eslesmeleri atip tekrar
            # dene; eskiden burada dongu bitiyor ve bolgedeki diger kartlar
            # da kayboluyordu. Bu bolgede zaten kart bulunduysa tekrar
            # denemiyoruz: kazanci olculmedi, suresi ise belirgin.
            if basarisiz_eleme >= eleme_hakki or bulunan_bu_bolgede:
                break
            kp, des, elendi = ozellikleri_ele(kp, des, adaylar[0].get("inlier_idx", []))
            if not elendi or des is None or len(kp) < 8:
                break
            basarisiz_eleme += 1
            continue

        aday, kart, koseler_sayfa, koseler_calisma = kabul
        sonuc = _bos_sonuc()
        sonuc.update({
            "basarili": True, "kart": kart, "belge_tipi": aday["belge_tipi"],
            "koseler": koseler_sayfa, "iyi_eslesme": aday["iyi_eslesme"],
            "inlier": aday["inlier"], "inlier_orani": aday["inlier_orani"],
            "skor": aday["skor"], "arama_modu": mod,
            "mesaj": "Kart tespit edildi ve yönü düzeltildi.",
            # Geçici bulunma sırası; en sonda sayfa sırasına göre yeniden yazılır.
            "kart_sirasi": len(sonuclar) + 1,
        })
        if debug_kart:
            sonuc["debug_resmi"] = kart_debug_resmi(sayfa_resmi, koseler_sayfa)
        sonuclar.append(sonuc)
        bulunan_bu_bolgede += 1

        kp, des, elendi = kartin_disindaki_ozellikler(kp, des, koseler_calisma)
        if not elendi or des is None or len(kp) < 8:
            break

    return ilk_adaylar


def kartlari_tespit_et_ve_duzelt(resim, debug_kart=False, azami_kart=AZAMI_KART_SAYISI,
                                 ek_tarama=True):
    """Bir sayfadaki TÜM kimlikleri bulur ve düzeltir; sonuç listesi döner.

    Üç aşamalı:
    1) TÜM SAYFA: referanslarla eşleştir -> geometri -> örtüşme -> warp ->
       kalite; kabul edilen kartın keypoint'leri atılıp tur tekrarlanır,
       böylece aynı sayfadaki 2., 3... kimlik de bulunur.
    2) ADAY BÖLGELER: sayfada kart büyüklüğünde olup henüz kapsanmamış
       bloklar kırpılıp TEK BAŞINA aynı akıştan geçirilir. Küçük kartlar
       sayfanın tamamı küçüldüğünde eleniyordu; kırpılmış blokta aynı kart
       tam çözünürlükte tarandığı için yakalanır.
    3) ZOR MOD: hâlâ hiçbir kart yoksa tüm sayfa, çok daha geniş özellik
       bütçesiyle bir kez daha taranır (pahalı olduğu için sadece burada).

    Hiç kart bulunamazsa liste tek bir başarısız sonuç içerir (çağıran taraf
    her zaman en az bir satır üretebilsin diye)."""
    baslangic = time.perf_counter()

    resim, girdi_hatasi = _girdi_hazirla(resim)
    if girdi_hatasi:
        return [_bos_sonuc(girdi_hatasi)]

    sonuclar = []
    referans_hatalari = {}

    # 1) Tüm sayfa
    ilk_adaylar = _bolgede_kart_ara(
        resim, (0, 0), resim, sonuclar, azami_kart, debug_kart, referans_hatalari,
        eleme_hakki=AZAMI_BASARISIZ_ELEME,
    )

    # 2) Kapsanmamış aday bölgelere yakından bak.
    #    ek_tarama=False, "bu dosyada sayfa başına tek kimlik var" demektir:
    #    bir kart bulunduysa sayfanın geri kalanına bakmaya gerek yok (hızlı yol).
    if len(sonuclar) < azami_kart and (ek_tarama or not sonuclar):
        for x0, y0, x1, y1 in kart_adayi_bolgeler(resim, sonuclar):
            if len(sonuclar) >= azami_kart:
                break
            blok = np.ascontiguousarray(resim[y0:y1, x0:x1])
            if blok.shape[0] < 2 or blok.shape[1] < 2:
                continue
            _bolgede_kart_ara(
                blok, (x0, y0), resim, sonuclar, azami_kart, debug_kart, referans_hatalari
            )

    # 3) Hâlâ boş: zor mod
    if not sonuclar:
        zor_adaylar = _bolgede_kart_ara(
            resim, (0, 0), resim, sonuclar, azami_kart, debug_kart, referans_hatalari,
            mod="zor", eleme_hakki=AZAMI_BASARISIZ_ELEME,
        )
        if not ilk_adaylar:
            ilk_adaylar = zor_adaylar

    # Hicbir karta denk gelmeyen, ama kart buyuklugunde olan bloklar: tespit
    # bunlari cozemedi. Cagiran taraf isterse buralari dogrudan OCR'layip
    # (kurtarma) kimligin sessizce kaybolmasini onleyebilir.
    kurtarma_bolgeleri = kart_adayi_bolgeler(resim, sonuclar)

    sure = time.perf_counter() - baslangic

    if not sonuclar:
        if ilk_adaylar:
            mesaj = (f"Belge tipi eşleşmesi bulundu ({ilk_adaylar[0]['belge_tipi']}), "
                     "ancak güvenilir şekilde düzeltilemedi.")
        else:
            ayrinti = f" Referans hataları: {' | '.join(referans_hatalari.values())}" if referans_hatalari else ""
            mesaj = "Belge tipi bulunamadı." + ayrinti
        bos = _bos_sonuc(mesaj, referans_hatalari)
        bos["tespit_suresi"] = sure
        bos["kurtarma_bolgeleri"] = kurtarma_bolgeleri
        if ilk_adaylar:
            bos.update({
                "belge_tipi": ilk_adaylar[0]["belge_tipi"], "iyi_eslesme": ilk_adaylar[0]["iyi_eslesme"],
                "inlier": ilk_adaylar[0]["inlier"], "inlier_orani": ilk_adaylar[0]["inlier_orani"],
                "skor": ilk_adaylar[0]["skor"],
            })
        return [bos]

    sonuclar = sayfa_sirasina_diz(sonuclar)
    for sira, sonuc in enumerate(sonuclar, start=1):
        sonuc["referans_hatalari"] = referans_hatalari
        sonuc["kurtarma_bolgeleri"] = kurtarma_bolgeleri
        sonuc["kart_sirasi"] = sira
        sonuc["kart_sayisi"] = len(sonuclar)
        sonuc["tespit_suresi"] = sure
    return sonuclar


def kart_tespit_et_ve_duzelt(resim, debug_kart=False):
    """Geriye dönük uyumluluk: sayfadaki İLK (en üstteki) kartı döner."""
    return kartlari_tespit_et_ve_duzelt(resim, debug_kart=debug_kart)[0]
