import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from kimlikler.kimlikler_tc import tc_kimlik_gecerli_mi


# =========================================================
# NORMALIZE
# =========================================================

def normalize_text(text):
    text = str(text).upper().strip()
    text = text.translate(str.maketrans({
        "İ": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C",
    }))
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def isim_temizle(text):
    if not text:
        return ""
    text = str(text).upper().strip()
    # Göçmen belgelerinde Türkçe alfabesi dışındaki Latin harfleri ve
    # apostroflu/tireli adları kaybetme (ÉLODIE, GARCÍA, O'CONNOR gibi).
    text = "".join(
        karakter if karakter.isalpha() or karakter.isspace() or karakter in "-'’" else " "
        for karakter in text
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_ISIM_ICI_RAKAM_DUZELTMELERI = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}


def _isim_ocr_hatasini_duzelt(text):
    """Yalniz harflerin arasindaki tekil OCR rakamlarini duzeltir.

    EasyOCR net ve uzun bir adi bazen ``Y0UNUS`` veya ``MUST4FA`` olarak
    dondurebiliyor. Onceki genel ``rakam varsa reddet`` kurali bu nedenle tum
    ad kutusunu kaybediyordu. Duzeltme yalniz alfabetik iki komsu arasindaki en
    fazla iki rakama uygulanir; saf sayilar, tarih/seri ve ``ALI1`` gibi sonu
    rakamli tokenlar isim haline getirilmez.
    """
    ham = str(text or "").upper().strip()
    if not ham:
        return ""
    rakam_indeksleri = [i for i, karakter in enumerate(ham) if karakter.isdigit()]
    if not rakam_indeksleri:
        return isim_temizle(ham)

    harf_sayisi = sum(karakter.isalpha() for karakter in ham)
    if harf_sayisi < 3 or len(rakam_indeksleri) > min(2, max(1, harf_sayisi // 5)):
        return ""

    karakterler = list(ham)
    for indeks in rakam_indeksleri:
        karakter = ham[indeks]
        if karakter not in _ISIM_ICI_RAKAM_DUZELTMELERI:
            return ""
        if indeks == 0 or indeks == len(ham) - 1:
            return ""
        if not ham[indeks - 1].isalpha() or not ham[indeks + 1].isalpha():
            return ""
        karakterler[indeks] = _ISIM_ICI_RAKAM_DUZELTMELERI[karakter]

    return isim_temizle("".join(karakterler))


def benzerlik(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _guven_degeri(item, varsayilan=0.50):
    """OCR guvenini karsilastirmalarda kullanilabilir, sinirli bir sayiya cevirir."""
    if "conf" not in item:
        return varsayilan
    try:
        return max(0.0, min(1.0, float(item.get("conf", varsayilan))))
    except (TypeError, ValueError):
        return varsayilan


# =========================================================
# KUTU
# =========================================================

def merkez_y(item):
    return (item["y1"] + item["y2"]) / 2


def merkez_x(item):
    return (item["x1"] + item["x2"]) / 2


def yukseklik(item):
    return max(1, item["y2"] - item["y1"])


# =========================================================
# LABEL TİPİ
# =========================================================

_LABEL_HEDEFLERI = {
    "kimlik_no": "YABANCI KIMLIK NO",
    "soyad": "SOYADI",
    "ad": "ADI",
    "baba": "BABA ADI",
    "anne": "ANNE ADI",
    "gecerlilik": "BELGENIN GECERLILIK TARIHI",
}


def label_eslesme_skoru(text, tip, inline_izinli=True):
    """Metnin istenen etikete ne kadar acik bicimde benzediğini dondurur.

    Ozellikle uc harfli ``ADI`` etiketi icin genel fuzzy eslesme yapilmaz.
    Aksi halde ALI gibi yaygin adlar tek harf farki nedeniyle etiket sayiliyordu.
    """
    norm = normalize_text(text)
    if not norm or tip not in _LABEL_HEDEFLERI:
        return 0.0

    kelimeler = norm.split()
    kompakt = "".join(kelimeler)

    if tip == "ad":
        ilk = kelimeler[0] if kelimeler else ""
        if norm in {"AD", "ADI", "AD1", "A DI"}:
            return 1.0
        if inline_izinli and ilk in {"AD", "ADI", "AD1"} and len(kelimeler) > 1:
            return 0.94
        return 0.0

    if tip == "soyad":
        ilk = kelimeler[0] if kelimeler else ""
        if ilk in {"SOYAD", "SOYADI", "SOYAD1"}:
            return 1.0 if len(kelimeler) == 1 else (0.94 if inline_izinli else 0.0)

    if tip in {"baba", "anne"}:
        kok = "BABA" if tip == "baba" else "ANNE"
        if kok in kelimeler and any(x in kelimeler for x in {"AD", "ADI", "AD1"}):
            return 1.0

    if tip == "kimlik_no":
        yabanci_var = any(k.startswith("YABANCI") for k in kelimeler)
        kimlik_var = any(k.startswith("KIMLIK") for k in kelimeler)
        if yabanci_var and kimlik_var:
            return 1.0

    if tip == "gecerlilik":
        if "GECERLILIK" in kompakt and ("BELGE" in kompakt or "TARIH" in kompakt):
            return 1.0
        if "BASLANGIC" in kompakt and "BITIS" in kompakt:
            return 1.0

    hedef = _LABEL_HEDEFLERI[tip]
    hedef_norm = normalize_text(hedef)
    # Tek kelimelik degerlerin uzun etiketlere benzetilmesi, isimleri yutuyordu.
    min_kelime = 1 if tip == "soyad" else 2
    if len(kelimeler) < min_kelime:
        return 0.0
    if len(norm) < len(hedef_norm) * 0.62 or len(norm) > len(hedef_norm) * 1.45:
        return 0.0

    skor = benzerlik(norm, hedef_norm)
    esik = 0.72 if tip in {"soyad", "baba", "anne"} else 0.66
    return skor if skor >= esik else 0.0


def label_tipini_bul(text):
    norm = normalize_text(text)
    if not norm:
        return None

    skorlar = {tip: label_eslesme_skoru(norm, tip) for tip in _LABEL_HEDEFLERI}
    tip = max(skorlar, key=skorlar.get)
    return tip if skorlar[tip] > 0 else None


# =========================================================
# SATIR
# =========================================================

def satirlara_grupla(ocr_sonuclari):
    sirali = sorted(ocr_sonuclari, key=lambda x: (merkez_y(x), x["x1"]))
    satirlar = []

    for item in sirali:
        item_y = merkez_y(item)
        bulundu = False

        for satir in satirlar:
            satir_y = sum(merkez_y(x) for x in satir) / len(satir)
            ort_h = sum(yukseklik(x) for x in satir) / len(satir)
            tolerans = max(22, ort_h * 0.85)
            if abs(item_y - satir_y) <= tolerans:
                satir.append(item)
                bulundu = True
                break

        if not bulundu:
            satirlar.append([item])

    for satir in satirlar:
        satir.sort(key=lambda x: x["x1"])

    return satirlar


# =========================================================
# LABEL / VALUE
# =========================================================

def _birlesik_item(itemlar, text=None):
    return {
        "text": text if text is not None else " ".join(str(x.get("text", "")) for x in itemlar),
        "conf": sum(_guven_degeri(x) for x in itemlar) / len(itemlar),
        "x1": min(x["x1"] for x in itemlar),
        "y1": min(x["y1"] for x in itemlar),
        "x2": max(x["x2"] for x in itemlar),
        "y2": max(x["y2"] for x in itemlar),
        "_source_items": itemlar,
    }


def satir_label_bul(satir, tip):
    adaylar = []
    # Etiket EasyOCR tarafindan "YABANCI" / "KIMLIK" / "NO" gibi ayri
    # kutulara bolunebiliyor. En fazla dort komsu tokeni birlikte de deneriz.
    for baslangic in range(len(satir)):
        for uzunluk in range(1, min(4, len(satir) - baslangic) + 1):
            kaynaklar = satir[baslangic:baslangic + uzunluk]
            text = " ".join(str(x.get("text", "")) for x in kaynaklar)
            eslesme = label_eslesme_skoru(text, tip, inline_izinli=(uzunluk == 1))
            if eslesme <= 0:
                continue

            if tip == "ad" and uzunluk == 1 and baslangic > 0:
                onceki = normalize_text(satir[baslangic - 1].get("text", ""))
                if onceki in {"BABA", "ANNE", "SOYAD", "SOYADI"}:
                    continue

            item = kaynaklar[0] if uzunluk == 1 else _birlesik_item(kaynaklar, text)
            guven = sum(_guven_degeri(x) for x in kaynaklar) / len(kaynaklar)
            # Fuzzy bir etiket dusuk OCR guveniyle tek basina kabul edilmez.
            if eslesme < 0.90 and guven < 0.25:
                continue

            # Etiketler kartta degerlerinden soldadir. Esit kalitedeki iki
            # adaydan soldaki ve daha guvenli olan secilir.
            # Ayni etiketi tek kutu da acikca veriyorsa, yanindaki degerleri
            # etikete katmis daha genis pencereler tercih edilmemeli.
            skor = eslesme * 100 + guven * 12 - item["x1"] * 0.002 - (uzunluk - 1) * 2.0
            adaylar.append((skor, item))

    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


def isim_value_mi(item):
    text = str(item.get("text", "")).strip()
    if not text:
        return False
    # Yalnizca acik bir etiket eslesmesi degeri eler. ALI -> ADI gibi kisa
    # fuzzy eslesmeler label_tipini_bul tarafinda zaten yasaktir.
    if label_tipini_bul(text) is not None:
        return False
    if "conf" in item and _guven_degeri(item) < 0.08:
        return False

    temiz = _isim_ocr_hatasini_duzelt(text)
    return sum(karakter.isalpha() for karakter in temiz) >= 2


def _inline_isim_degeri(label_item, tip):
    """``ADI ALI`` gibi tek OCR kutusundaki etiketten degeri ayirir."""
    kaynaklar = label_item.get("_source_items", [label_item])
    if len(kaynaklar) != 1:
        return None

    kaynak = kaynaklar[0]
    tokenlar = str(kaynak.get("text", "")).strip().split()
    norm_tokenlar = [normalize_text(x) for x in tokenlar]
    if len(tokenlar) < 2:
        return None

    baslangic = None
    if tip == "ad" and norm_tokenlar[0] in {"AD", "ADI", "AD1"}:
        baslangic = 1
    elif tip == "soyad" and norm_tokenlar[0] in {"SOYAD", "SOYADI", "SOYAD1"}:
        baslangic = 1
    if baslangic is None or baslangic >= len(tokenlar):
        return None

    deger = _isim_ocr_hatasini_duzelt(" ".join(tokenlar[baslangic:]))
    if not deger:
        return None
    sanal = dict(kaynak)
    oran = baslangic / len(tokenlar)
    sanal["x1"] = int(kaynak["x1"] + (kaynak["x2"] - kaynak["x1"]) * oran)
    sanal["text"] = deger
    return sanal if isim_value_mi(sanal) else None


def satir_isim_bul(satir, label_item):
    if label_item is None:
        return None

    kaynaklar = label_item.get("_source_items", [label_item])
    label_h = yukseklik(label_item)
    inline = _inline_isim_degeri(label_item, label_tipini_bul(label_item.get("text", "")))
    adaylar = ([inline] if inline is not None else []) + [
        item for item in satir
        if not any(item is kaynak for kaynak in kaynaklar)
        and isim_value_mi(item)
        # ROI'nin alt kenarinda bir sonraki satirdan kalan ince metin parcasi,
        # genis satir gruplama toleransiyla mevcut label'a yapismasin.
        and yukseklik(item) >= label_h * 0.42
        and item["x1"] >= label_item["x2"] - 8
    ]
    if not adaylar:
        return None

    adaylar.sort(key=lambda x: x["x1"])
    # Satir gruplama toleransi nedeniyle cok uzaktaki baska bir alan ayni
    # satira girdiyse isim parcasina eklenmesini engelle.
    yakin_adaylar = []
    onceki_x2 = label_item["x2"]
    for item in adaylar:
        if yakin_adaylar:
            izinli_bosluk = max(90, yukseklik(item) * 7)
        else:
            izinli_bosluk = max(260, yukseklik(item) * 16, yukseklik(label_item) * 16)
        if item["x1"] - onceki_x2 > izinli_bosluk:
            break
        yakin_adaylar.append(item)
        onceki_x2 = item["x2"]
    adaylar = yakin_adaylar

    parcalar = [_isim_ocr_hatasini_duzelt(item["text"]) for item in adaylar]
    parcalar = [p for p in parcalar if p]
    if not parcalar:
        return None

    text = " ".join(parcalar)
    conf = sum(_guven_degeri(x, 0.0) for x in adaylar) / len(adaylar)

    return {
        "deger": text,
        "item": {
            "text": text, "conf": conf,
            "x1": min(x["x1"] for x in adaylar), "y1": min(x["y1"] for x in adaylar),
            "x2": max(x["x2"] for x in adaylar), "y2": max(x["y2"] for x in adaylar),
        },
        "items": adaylar,
    }


def _alan_label_kayitlari(satirlar):
    """Ad/soyad fallback'i icin karttaki alan satirlarini bir kez bulur."""
    kayitlar = []
    for satir in satirlar:
        for tip in ("kimlik_no", "ad", "soyad", "baba", "anne"):
            label = satir_label_bul(satir, tip)
            if label is not None:
                kayitlar.append({"tip": tip, "satir": satir, "label": label})
    return kayitlar


def _ayni_kaynaklari_kullaniyor(a, b):
    a_kaynaklari = a.get("_source_items", [a])
    b_kaynaklari = b.get("_source_items", [b])
    return any(x is y for x in a_kaynaklari for y in b_kaynaklari)


def _yakin_isim_fallback_bul(ocr_sonuclari, label_item, label_kayitlari):
    """Satir gruplama kacirdiginda label'in sagindaki isim degerini bulur.

    EasyOCR ayni basili satirdaki label ve deger icin farkli yukseklikte
    kutular uretebiliyor. Sabit piksel esigi yerine kutu yuksekligini olcek
    aliriz. Diger alan labellarinin orta noktalari da satir siniri olarak
    kullanilir; bu sayede soyad, baba veya anne adi hedef alana tasinmaz.
    """
    label_y = merkez_y(label_item)
    label_h = yukseklik(label_item)
    label_kaynaklari = label_item.get("_source_items", [label_item])

    # Diger taninmis alan labellari satirin ust/alt sinirini belirler. OCR
    # "BABA" / "ADI" parcalarini ayri gruplara dusururse, soldaki acik alan
    # koklerini de koruyucu satir isareti olarak hesaba katariz.
    koruyucu_merkezler = []
    for kayit in label_kayitlari:
        diger = kayit["label"]
        if _ayni_kaynaklari_kullaniyor(label_item, diger):
            continue
        if diger["x1"] <= label_item["x2"] + label_h * 4:
            koruyucu_merkezler.append(merkez_y(diger))

    for item in ocr_sonuclari:
        if any(item is kaynak for kaynak in label_kaynaklari):
            continue
        norm = normalize_text(item.get("text", ""))
        if norm in {"AD", "ADI", "AD1", "SOYAD", "SOYADI", "SOYAD1", "BABA", "ANNE"}:
            if item["x1"] <= label_item["x2"] + label_h * 4:
                koruyucu_merkezler.append(merkez_y(item))

    ust_sinir, alt_sinir = float("-inf"), float("inf")
    for diger_y in koruyucu_merkezler:
        if diger_y < label_y:
            ust_sinir = max(ust_sinir, (diger_y + label_y) / 2)
        elif diger_y > label_y:
            alt_sinir = min(alt_sinir, (diger_y + label_y) / 2)

    adaylar = []
    sinir_parcalari = []
    for item in ocr_sonuclari:
        if any(item is kaynak for kaynak in label_kaynaklari) or not isim_value_mi(item):
            continue

        item_h = yukseklik(item)
        olcek = max(label_h, item_h)
        item_y = merkez_y(item)
        dikey_bosluk = max(
            0,
            item["y1"] - label_item["y2"],
            label_item["y1"] - item["y2"],
        )

        if item["x1"] < label_item["x2"] - olcek * 0.45:
            continue
        if item["x1"] - label_item["x2"] > olcek * 24:
            continue
        # Dar OCR bandinin alt kenarinda bir sonraki satirin yalniz ust parcasi
        # gorunebilir. Bu kirpilmis 5-10 px kutuyu (ornegin Baba Adi degerini)
        # mevcut SOYADI label'inin degeri sanma. Ayni basili satirdaki gercek
        # deger, label yuksekliginin anlamli bir bolumunu kaplamalidir.
        if item_h < label_h * 0.42:
            continue
        dikey_yakin = (
            abs(item_y - label_y) <= olcek * 1.75
            and dikey_bosluk <= olcek * 0.85
        )
        merkez_hedefte = ust_sinir < item_y < alt_sinir
        if dikey_yakin and merkez_hedefte:
            adaylar.append(item)
            continue

        # Uzun ad birden fazla OCR kutusuna bolundugunde son parcanin merkezi
        # baseline farkiyla komsu satir sinirini az miktarda asabilir. Bu parcayi
        # tek basina kabul etmeyiz; asagida hedef satirdaki cekirdek parcaya yatay
        # ve dikey olarak bitisik oldugu kanitlanirsa birlestiririz.
        sinira_tasiyor = item["y2"] > ust_sinir and item["y1"] < alt_sinir
        if (
            sinira_tasiyor
            and abs(item_y - label_y) <= olcek * 2.25
            and dikey_bosluk <= olcek * 1.25
        ):
            sinir_parcalari.append(item)

    if adaylar and sinir_parcalari:
        for item in sorted(sinir_parcalari, key=lambda x: x["x1"]):
            for cekirdek in adaylar:
                olcek = max(label_h, yukseklik(item), yukseklik(cekirdek))
                yatay_bosluk = max(
                    0,
                    item["x1"] - cekirdek["x2"],
                    cekirdek["x1"] - item["x2"],
                )
                if (
                    abs(merkez_y(item) - merkez_y(cekirdek)) <= olcek * 0.90
                    and yatay_bosluk <= max(24, olcek * 2.5)
                ):
                    adaylar.append(item)
                    break

    if not adaylar:
        return None

    # Aday elemesi dikey olarak yapildi; mevcut yatay parca birlestirme ve
    # guven hesabi tek davranis kaynagi olarak korunur.
    return satir_isim_bul([label_item, *adaylar], label_item)


def _isim_alani_bul(ocr_sonuclari, tip, label_kayitlari):
    hedefler = [kayit for kayit in label_kayitlari if kayit["tip"] == tip]

    # Hizli ve en guvenilir yol: label ile deger ayni OCR satirindadir.
    for kayit in hedefler:
        sonuc = satir_isim_bul(kayit["satir"], kayit["label"])
        if sonuc is not None:
            return kayit["label"], sonuc

    # Yalnizca hizli yol sonuc vermediyse tum OCR kutularinda olcek-duyarli
    # yakinlik aramasi yapilir.
    for kayit in hedefler:
        sonuc = _yakin_isim_fallback_bul(
            ocr_sonuclari, kayit["label"], label_kayitlari,
        )
        if sonuc is not None:
            return kayit["label"], sonuc

    return None, None


_LABELSIZ_ISIM_YASAK_KELIMELERI = {
    "YABANCI", "YKN", "NO", "TC", "KIMLIK", "KIMLIGI", "BELGE", "BELGESI",
    "ULUSLARARASI", "KORUMA", "BASVURU", "SAHIBI", "TURKIYE",
    "CUMHURIYETI", "ICISLERI", "BAKANLIGI", "VALILIGI", "GECERLILIK",
    "TARIHI", "DOGUM", "UYRUGU", "CINSIYETI", "SERI", "BABA", "ANNE",
}


_GOCMEN_ALAN_SIRASI = {
    "kimlik_no": 0, "ad": 1, "soyad": 2, "baba": 3, "anne": 4,
}


def _tahmini_alan_satir_adimi(label_kayitlari):
    """Taninmis tablo labellarindan bir alan satirinin yuksekligini kestirir."""
    adimlar = []
    for indeks, ilk in enumerate(label_kayitlari):
        ilk_sira = _GOCMEN_ALAN_SIRASI.get(ilk["tip"])
        if ilk_sira is None:
            continue
        for ikinci in label_kayitlari[indeks + 1:]:
            ikinci_sira = _GOCMEN_ALAN_SIRASI.get(ikinci["tip"])
            sira_farki = ikinci_sira - ilk_sira if ikinci_sira is not None else 0
            if sira_farki == 0:
                continue
            y_farki = merkez_y(ikinci["label"]) - merkez_y(ilk["label"])
            # Tablo alanlarinin fiziksel ve mantiksal sirasi ayni olmali.
            if y_farki * sira_farki <= 0:
                continue
            adim = abs(y_farki / sira_farki)
            olcek = max(yukseklik(ilk["label"]), yukseklik(ikinci["label"]))
            if olcek * 0.55 <= adim <= olcek * 6.0:
                adimlar.append(adim)

    if not adimlar:
        return None
    adimlar.sort()
    orta = len(adimlar) // 2
    if len(adimlar) % 2:
        return adimlar[orta]
    return (adimlar[orta - 1] + adimlar[orta]) / 2


def _guvenilir_isim_dayanagi_mi(tip, label_item, sonuc):
    """Labelsiz komsu alan icin label+deger dayanak kalitesini denetler."""
    if label_item is None or sonuc is None:
        return False
    label_skoru = label_eslesme_skoru(label_item.get("text", ""), tip)
    if label_skoru <= 0:
        return False
    label_guveni = _guven_degeri(label_item, 0.0)
    # Fuzzy SOYADI okumasi ancak OCR guveni cok yuksekse dayanak olabilir.
    # Kesin label/alias (SOYADI, SOYAD1 vb.) icin daha normal esik yeterlidir.
    if label_guveni < (0.82 if label_skoru < 0.90 else 0.55):
        return False

    deger_item = sonuc["item"]
    min_deger_guveni = 0.72 if label_skoru < 0.90 else 0.55
    if _guven_degeri(deger_item, 0.0) < min_deger_guveni:
        return False

    olcek = max(yukseklik(label_item), yukseklik(deger_item))
    dikey_bosluk = max(
        0,
        deger_item["y1"] - label_item["y2"],
        label_item["y1"] - deger_item["y2"],
    )
    if deger_item["x1"] < label_item["x2"] - olcek * 0.45:
        return False
    if abs(merkez_y(deger_item) - merkez_y(label_item)) > olcek * 1.25:
        return False
    return dikey_bosluk <= olcek * 0.55


def _satirda_labelsiz_isim_yasak_mi(satir):
    metin = " ".join(str(item.get("text", "")) for item in satir)
    kelimeler = set(normalize_text(metin).split())
    if kelimeler & _LABELSIZ_ISIM_YASAK_KELIMELERI:
        return True

    # Hedef label OCR'da yoksa aday satir tamamen labelsiz olmalidir. Baska
    # bir alanin degerini (ozellikle baba/anne adini) komsu alan saymayiz.
    return any(
        satir_label_bul(satir, tip) is not None
        for tip in ("kimlik_no", "ad", "soyad", "baba", "anne", "gecerlilik")
    )


def _labelsiz_komsu_isim_bul(
    ocr_sonuclari,
    satirlar,
    eksik_tip,
    dayanak_label,
    dayanak_sonuc,
    label_kayitlari,
):
    """Eksik ADI/SOYADI label'ini yalniz guvenilir komsu satirdan tamamlar."""
    dayanak_tip = "soyad" if eksik_tip == "ad" else "ad"
    if not _guvenilir_isim_dayanagi_mi(dayanak_tip, dayanak_label, dayanak_sonuc):
        return None

    dayanak_item = dayanak_sonuc["item"]
    dayanak_y = merkez_y(dayanak_item)
    dayanak_x = dayanak_item["x1"]
    dayanak_olcegi = max(yukseklik(dayanak_label), yukseklik(dayanak_item))
    yon = -1 if eksik_tip == "ad" else 1
    satir_adimi = _tahmini_alan_satir_adimi(label_kayitlari)
    aday_sonuclari = []

    for satir in satirlar:
        if any(item is kaynak for item in satir for kaynak in dayanak_sonuc["items"]):
            continue
        if _satirda_labelsiz_isim_yasak_mi(satir):
            continue

        isim_itemlari = [
            item for item in satir
            if isim_value_mi(item) and _guven_degeri(item, 0.0) >= 0.35
        ]
        if not isim_itemlari:
            continue

        item_olcegi = max(yukseklik(item) for item in isim_itemlari)
        if not dayanak_olcegi * 0.55 <= item_olcegi <= dayanak_olcegi * 1.80:
            continue
        olcek = max(dayanak_olcegi, item_olcegi)
        hizalama_toleransi = olcek * 2.75
        sirali = sorted(
            (
                item for item in isim_itemlari
                if item["x1"] >= dayanak_x - hizalama_toleransi
            ),
            key=lambda item: item["x1"],
        )
        if not sirali or abs(sirali[0]["x1"] - dayanak_x) > hizalama_toleransi:
            continue
        if sirali[0]["x1"] < dayanak_label["x2"] - olcek * 0.45:
            continue

        # Ayni deger satirindaki bolunmus ad parcalarini birlestiririz; ikinci
        # ayri bir alfabetik kume varsa satiri belirsiz kabul ederiz.
        secilenler = [sirali[0]]
        onceki_x2 = sirali[0]["x2"]
        satir_belirsiz = False
        for item in sirali[1:]:
            if item["x1"] - onceki_x2 > olcek * 7:
                satir_belirsiz = True
                break
            secilenler.append(item)
            onceki_x2 = item["x2"]
        if satir_belirsiz:
            continue

        aday_y = sum(merkez_y(item) for item in secilenler) / len(secilenler)
        yonlu_mesafe = (aday_y - dayanak_y) * yon
        if not olcek * 0.90 <= yonlu_mesafe <= olcek * 3.40:
            continue
        if satir_adimi is not None:
            # SOYADI label'i tamamen kayboldugunda bir alt satir beklenir.
            # Iki satir alttaki Baba Adi degeri, yalniz alfabetik ve net olsa
            # bile soyad olarak kabul edilmemelidir.
            beklenen_y = dayanak_y + yon * satir_adimi
            tolerans = max(satir_adimi * 0.45, olcek * 0.70)
            if abs(aday_y - beklenen_y) > tolerans:
                continue

        # Aday, taninmis veya parcali baska bir alan label'inin satirindaysa
        # YKN/baba/anne degerine tasma riski vardir.
        engellendi = False
        for kayit in label_kayitlari:
            diger_label = kayit["label"]
            if _ayni_kaynaklari_kullaniyor(dayanak_label, diger_label):
                continue
            diger_y = merkez_y(diger_label)
            if abs(aday_y - diger_y) <= olcek * 1.75:
                engellendi = True
                break
            if min(aday_y, dayanak_y) < diger_y < max(aday_y, dayanak_y):
                engellendi = True
                break
        if not engellendi:
            for item in ocr_sonuclari:
                kelimeler = set(normalize_text(item.get("text", "")).split())
                if not kelimeler & _LABELSIZ_ISIM_YASAK_KELIMELERI:
                    continue
                if abs(aday_y - merkez_y(item)) <= olcek * 1.75:
                    engellendi = True
                    break
        if engellendi:
            continue

        text = " ".join(
            _isim_ocr_hatasini_duzelt(item["text"]) for item in secilenler
        ).strip()
        conf = sum(_guven_degeri(item, 0.0) for item in secilenler) / len(secilenler)
        if not text or conf < 0.55:
            continue

        aday_sonuclari.append({
            "deger": text,
            "item": {
                "text": text,
                "conf": conf,
                "x1": min(item["x1"] for item in secilenler),
                "y1": min(item["y1"] for item in secilenler),
                "x2": max(item["x2"] for item in secilenler),
                "y2": max(item["y2"] for item in secilenler),
            },
            "items": secilenler,
        })

    # Birden fazla makul labelsiz satir varsa sira varsayimiyla secim yapmayiz.
    return aday_sonuclari[0] if len(aday_sonuclari) == 1 else None


# =========================================================
# YABANCI KİMLİK NO
# =========================================================

def yabanci_no_metni_duzelt(text):
    donusum = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "İ": "1", "L": "1",
        "Z": "2", "S": "5", "G": "6", "B": "8",
    }
    sonuc = ""
    for karakter in str(text).upper():
        if karakter.isdigit():
            sonuc += karakter
        elif karakter in donusum:
            sonuc += donusum[karakter]
    return sonuc


def yabanci_no_gecerli_mi(no):
    # Yabanci kimlik numaralari 99 ile baslar ve MERNIS'in 11 haneli kimlik
    # numarasi kontrol basamaklarini kullanir. Yalniz uzunluk/prefix kontrolu,
    # ekrandaki ...446 -> ...448 OCR hatasini gercek numara gibi kabul ediyordu.
    return (
        no is not None
        and no.isdigit()
        and len(no) == 11
        and no.startswith("99")
        and tc_kimlik_gecerli_mi(no)
    )


_NO_OCR_KARAKTERLERI = "0-9OQDILİZSGB"


def _no_parcasini_coz(parca):
    kompakt = re.sub(r"[\s./-]", "", str(parca).upper())
    if not kompakt or re.fullmatch(rf"[{_NO_OCR_KARAKTERLERI}]+", kompakt) is None:
        return None

    aday = yabanci_no_metni_duzelt(kompakt)
    if not yabanci_no_gecerli_mi(aday):
        return None

    duzeltme_sayisi = sum(not c.isdigit() for c in kompakt)
    # Checksum dogrulansa da cok sayida harften uretilen rastgele bir tokeni
    # kimlik numarasina cevirmiyoruz.
    if duzeltme_sayisi > 2:
        return None
    return aday, duzeltme_sayisi


def _item_no_adaylari(item):
    raw = str(item.get("text", "")).upper().strip()
    adaylar = []

    # En guvenilir yapi: kutunun tamami yalnizca 11 haneli sayi (ayiracli da olabilir).
    tam = _no_parcasini_coz(raw)
    if tam is not None:
        no, duzeltme = tam
        sadece_rakam = raw.isdigit()
        adaylar.append({
            "deger": no,
            "item": item,
            "duzeltme": duzeltme,
            "acik_yapi": duzeltme == 0,
            "yapi_skoru": 64 if sadece_rakam else (54 if duzeltme == 0 else 34),
        })

    # Dinamik tablo geometrisiyle dogrudan YKN hucresinden gelen metinde kase
    # veya hucre cizgisi tek bir ek karakter uretebilir ("|990..." gibi).
    # Tum kutuyu bu nedenle kaybetme: yalniz bu acikca isaretli fiziksel hucrede,
    # tekil 99... parcayi ayikla ve ayni MERNIS checksum kontrolunden gecir.
    if (
        tam is None
        and item.get("_dogrudan_hucre_ocr")
        and item.get("_fallback_alan") == "kimlik_no"
    ):
        desen = rf"(?<![{_NO_OCR_KARAKTERLERI}])(9[9G](?:[\s./-]?[{_NO_OCR_KARAKTERLERI}]){{9}})(?![{_NO_OCR_KARAKTERLERI}])"
        bulunanlar = []
        for eslesme in re.finditer(desen, raw):
            cozulmus = _no_parcasini_coz(eslesme.group(1).rstrip())
            if cozulmus is not None:
                bulunanlar.append(cozulmus)
        tekil = {no: duzeltme for no, duzeltme in bulunanlar}
        if len(tekil) == 1:
            no, duzeltme = next(iter(tekil.items()))
            adaylar.append({
                "deger": no,
                "item": item,
                "duzeltme": duzeltme,
                "acik_yapi": False,
                "yapi_skoru": 58 if duzeltme == 0 else 38,
            })

    # Label ve numara ayni OCR kutusundaysa, yalnizca 99/9G ile baslayan acik
    # sayisal parcayi ayikla. Metindeki tum rakamlari korlemesine birlestirmeyiz.
    if label_eslesme_skoru(raw, "kimlik_no") > 0:
        desen = rf"(?<![A-Z0-9])(9[9G](?:[\s./-]?[{_NO_OCR_KARAKTERLERI}]){{9}})(?![{_NO_OCR_KARAKTERLERI}])"
        for eslesme in re.finditer(desen, raw):
            cozulmus = _no_parcasini_coz(eslesme.group(1).rstrip())
            if cozulmus is None:
                continue
            no, duzeltme = cozulmus
            adaylar.append({
                "deger": no, "item": item, "duzeltme": duzeltme,
                "acik_yapi": False, "yapi_skoru": 42 if duzeltme == 0 else 30,
            })

    return adaylar


def _label_yakinlik_skoru(aday_item, label_item):
    if aday_item is label_item:
        return 110

    dy = abs(merkez_y(aday_item) - merkez_y(label_item))
    satir_tol = max(24, yukseklik(aday_item) * 0.9, yukseklik(label_item) * 0.9)
    if dy <= satir_tol and aday_item["x1"] >= label_item["x1"]:
        yatay_bosluk = max(0, aday_item["x1"] - label_item["x2"])
        return max(0, 100 - yatay_bosluk * 0.08)
    if dy <= 95:
        dx = abs(merkez_x(aday_item) - merkez_x(label_item))
        return max(0, 58 - dy * 0.35 - dx * 0.015)
    return 0


def yabanci_no_bul(ocr_sonuclari):
    adaylar = []
    for item in ocr_sonuclari:
        adaylar.extend(_item_no_adaylari(item))
    if not adaylar:
        return None

    satirlar = satirlara_grupla(ocr_sonuclari)
    labellar = [label for satir in satirlar if (label := satir_label_bul(satir, "kimlik_no"))]

    for aday in adaylar:
        yakinlik = max(
            (_label_yakinlik_skoru(aday["item"], label) for label in labellar),
            default=0,
        )
        aday["label_skoru"] = yakinlik
        aday["skor"] = aday["yapi_skoru"] + _guven_degeri(aday["item"]) * 22 + yakinlik

    # Ayni numaranin birden fazla OCR kutusundaki tekrarini belirsizlik sayma.
    numaraya_gore = {}
    for aday in adaylar:
        onceki = numaraya_gore.get(aday["deger"])
        if onceki is None or aday["skor"] > onceki["skor"]:
            numaraya_gore[aday["deger"]] = aday
    sirali = sorted(numaraya_gore.values(), key=lambda x: x["skor"], reverse=True)
    en_iyi = sirali[0]

    if len(sirali) > 1:
        # Birden cok farkli 99... sayisi varsa yalnizca etikete acikca yakin ve
        # rakiplerinden belirgin guclu olan aday kabul edilir.
        if en_iyi["label_skoru"] < 62:
            return None
        ikinci = sirali[1]
        if ikinci["label_skoru"] >= 62 and en_iyi["skor"] - ikinci["skor"] < 18:
            return None
    elif en_iyi["label_skoru"] < 62:
        # Labelsiz geriye uyumlu fallback: tek, temiz, butun kutuyu kaplayan ve
        # guveni yeterli bir 11 haneli token olmali.
        guven_bilinmiyor = "conf" not in en_iyi["item"]
        if not en_iyi["acik_yapi"] or (
            not guven_bilinmiyor and _guven_degeri(en_iyi["item"]) < 0.55
        ):
            return None

    return {"deger": en_iyi["deger"], "item": en_iyi["item"]}


# =========================================================
# TARİH
# =========================================================

_TARIH_DESENI = re.compile(
    r"(?<!\d)(\d{1,2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{4})(?!\d)"
)


def _tarih_eslesmeleri(text):
    sonuc = []
    for eslesme in _TARIH_DESENI.finditer(str(text or "")):
        gun, ay, yil = eslesme.groups()
        try:
            tarih = datetime(int(yil), int(ay), int(gun)).date()
        except ValueError:
            continue
        sonuc.append((tarih, eslesme.start(), eslesme.end()))
    return sonuc


def tarihleri_bul(text):
    if not text:
        return []
    return [tarih for tarih, _, _ in _tarih_eslesmeleri(text)]


def tarih_metne_cevir(tarih):
    return tarih.strftime("%d.%m.%Y") if tarih is not None else "Bulunamadi"


def turkiye_bugun():
    # Belgenin hukuki gün sınırı Türkiye saatine göre değerlendirilir. UTC
    # kullanımı gece 00:00-03:00 arasında bir günlük sapma oluşturabiliyordu.
    return datetime.now(timezone(timedelta(hours=3))).date()


def _satir_tarih_olaylari(satir):
    olaylar = []
    for item in satir:
        text = str(item.get("text", ""))
        uzunluk = max(1, len(text))
        for tarih, bas, son in _tarih_eslesmeleri(text):
            # Ayni kutudaki iki tarihin de soldan saga sirasi korunur.
            x = item["x1"] + (item["x2"] - item["x1"]) * ((bas + son) / 2) / uzunluk
            olaylar.append({"tarih": tarih, "item": item, "x": x, "bas": bas, "son": son})
    return sorted(olaylar, key=lambda x: x["x"])


def tarih_debug_itemlari_olustur(satir, tarihler):
    olaylar = _satir_tarih_olaylari(satir)
    secilenler = []
    kullanilan = set()
    for tarih in tarihler[:2]:
        for index, olay in enumerate(olaylar):
            if index not in kullanilan and olay["tarih"] == tarih:
                secilenler.append(olay)
                kullanilan.add(index)
                break
    if len(secilenler) < 2:
        return []

    if secilenler[0]["item"] is not secilenler[1]["item"]:
        return [secilenler[0]["item"], secilenler[1]["item"]]

    # Tek OCR kutusundaki iki tarih icin kutuyu metindeki konumlarina gore bol.
    item = secilenler[0]["item"]
    text_uzunlugu = max(1, len(str(item.get("text", ""))))
    sinir_orani = (secilenler[0]["son"] + secilenler[1]["bas"]) / (2 * text_uzunlugu)
    sinir = int(item["x1"] + (item["x2"] - item["x1"]) * sinir_orani)
    return [{**item, "x2": sinir}, {**item, "x1": sinir}]


# =========================================================
# GEÇERLİLİK
# =========================================================

def gecerlilik_tarihi_bul(ocr_sonuclari):
    satirlar = satirlara_grupla(ocr_sonuclari)
    adaylar = []

    label_bilgileri = []
    for satir in satirlar:
        label = satir_label_bul(satir, "gecerlilik")
        if label is not None:
            label_bilgileri.append((label, sum(merkez_y(x) for x in satir) / len(satir)))

    for satir in satirlar:
        olaylar = _satir_tarih_olaylari(satir)
        if len(olaylar) < 2:
            continue

        satir_y = sum(merkez_y(x) for x in satir) / len(satir)

        # Uc veya daha fazla tarih ayni satirdaysa her komsu cifti ayri adaydir.
        # Kart uzerindeki dogal soldan-saga sirayi bozmayiz ve baslangic > bitis
        # olan cifti kesinlikle kabul etmeyiz.
        for index in range(len(olaylar) - 1):
            ilk, ikinci = olaylar[index], olaylar[index + 1]
            if ilk["tarih"] > ikinci["tarih"]:
                continue

            label_skor = 0.0
            for label, label_y in label_bilgileri:
                dy = abs(label_y - satir_y)
                if dy <= 30:
                    # Ayni satirda tarihlerin etiketten sonra gelmesi en guclu
                    # isarettir. Etiketten onceki tarih cifti (orn. dogum tarihi)
                    # daha dusuk oncelik alir.
                    if ilk["x"] >= label["x1"]:
                        bosluk = max(0, ilk["x"] - label["x2"])
                        yakinlik = max(205, 285 - bosluk * 0.05)
                    else:
                        yakinlik = 115
                    label_skor = max(label_skor, yakinlik)
                elif dy <= 105:
                    label_skor = max(label_skor, 210 - dy * 0.8)

            guven = (_guven_degeri(ilk["item"]) + _guven_degeri(ikinci["item"])) / 2
            tam_cift_bonusu = 28 if len(olaylar) == 2 else 0
            skor = label_skor + guven * 24 + tam_cift_bonusu
            adaylar.append({
                "skor": skor, "label_skoru": label_skor, "satir": satir,
                "tarihler": [ilk["tarih"], ikinci["tarih"]],
            })

    if not adaylar:
        # Tek tarih genel OCR'da baslangic, dogum veya basvuru tarihi olabilir;
        # bu nedenle label yakinligi bile tek basina yeterli degildir. Yalnizca
        # geometriyle dogrulanmis gecerlilik satirinin sag (bitis) crop'undan
        # gelen acik isaretli ve guvenli tek tarih kabul edilir.
        isaretli_bitisler = []
        for satir in satirlar:
            for item in satir:
                if not (
                    item.get("_bitis_hucresi")
                    and item.get("_geometri_dogrulandi")
                    and item.get("_dogrudan_hucre_ocr")
                    and _guven_degeri(item, 0.0) >= 0.55
                ):
                    continue
                olaylar = _satir_tarih_olaylari([item])
                if len(olaylar) == 1:
                    isaretli_bitisler.append(olaylar[0])

        if len(isaretli_bitisler) == 1:
            olay = isaretli_bitisler[0]
            bitis = olay["tarih"]
            if turkiye_bugun() > bitis:
                gecerli, durum = False, "suresi_gecmis"
            else:
                gecerli, durum = True, "gecerli"
            return {
                "baslangic": None,
                "bitis": bitis,
                "gecerli": gecerli,
                "durum": durum,
                "items": [olay["item"]],
            }

        return {"baslangic": None, "bitis": None, "gecerli": None, "durum": "kontrol_edilemedi", "items": []}

    etiketli_adaylar = [x for x in adaylar if x["label_skoru"] > 0]
    if etiketli_adaylar:
        secilen = max(etiketli_adaylar, key=lambda x: x["skor"])
    else:
        # Birden fazla labelsiz tarih satirindan keyfi olarak en alttakini
        # secmek yerine belirsiz sonucu raporla. Tek satir eski davranisi korur.
        aday_satirlari = {id(x["satir"]) for x in adaylar}
        if len(aday_satirlari) > 1:
            return {
                "baslangic": None, "bitis": None, "gecerli": None,
                "durum": "kontrol_edilemedi", "items": [],
            }
        secilen = max(adaylar, key=lambda x: x["skor"])

    satir, tarihler = secilen["satir"], secilen["tarihler"]
    baslangic, bitis = tarihler[0], tarihler[1]  # soldan sağa
    bugun = turkiye_bugun()

    if bugun < baslangic:
        gecerli, durum = False, "henuz_baslamadi"
    elif bugun > bitis:
        gecerli, durum = False, "suresi_gecmis"
    else:
        gecerli, durum = True, "gecerli"

    debug_items = tarih_debug_itemlari_olustur(satir, tarihler)
    return {"baslangic": baslangic, "bitis": bitis, "gecerli": gecerli, "durum": durum, "items": debug_items}


# =========================================================
# ANA
# =========================================================

def gocmen_bilgilerini_bul(ocr_sonuclari):
    satirlar = satirlara_grupla(ocr_sonuclari)
    no_sonuc = yabanci_no_bul(ocr_sonuclari)
    label_kayitlari = _alan_label_kayitlari(satirlar)
    ad_label, ad_sonuc = _isim_alani_bul(ocr_sonuclari, "ad", label_kayitlari)
    soyad_label, soyad_sonuc = _isim_alani_bul(ocr_sonuclari, "soyad", label_kayitlari)

    if ad_sonuc is None:
        ad_sonuc = _labelsiz_komsu_isim_bul(
            ocr_sonuclari,
            satirlar,
            "ad",
            soyad_label,
            soyad_sonuc,
            label_kayitlari,
        )
    if soyad_sonuc is None:
        soyad_sonuc = _labelsiz_komsu_isim_bul(
            ocr_sonuclari,
            satirlar,
            "soyad",
            ad_label,
            ad_sonuc,
            label_kayitlari,
        )

    gecerlilik = gecerlilik_tarihi_bul(ocr_sonuclari)

    kimlik_no = no_sonuc["deger"] if no_sonuc else "Bulunamadi"
    ad = ad_sonuc["deger"] if ad_sonuc else "Bulunamadi"
    soyad = soyad_sonuc["deger"] if soyad_sonuc else "Bulunamadi"

    kimlik_no_conf = _guven_degeri(no_sonuc["item"], 0.0) if no_sonuc else 0.0
    ad_conf = _guven_degeri(ad_sonuc["item"], 0.0) if ad_sonuc else 0.0
    soyad_conf = _guven_degeri(soyad_sonuc["item"], 0.0) if soyad_sonuc else 0.0

    bulunan = sum([kimlik_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if min(kimlik_no_conf, ad_conf, soyad_conf) >= 0.50 else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    return {
        "kimlik_no": kimlik_no, "ad": ad, "soyad": soyad, "guven": guven,
        "kimlik_no_conf": kimlik_no_conf, "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "baslangic_tarihi": tarih_metne_cevir(gecerlilik["baslangic"]),
        "bitis_tarihi": tarih_metne_cevir(gecerlilik["bitis"]),
        "belge_gecerli": gecerlilik["gecerli"], "gecerlilik_durumu": gecerlilik["durum"],
        "kimlik_no_item": no_sonuc["item"] if no_sonuc else None,
        "ad_item": ad_sonuc["item"] if ad_sonuc else None,
        "soyad_item": soyad_sonuc["item"] if soyad_sonuc else None,
        "ad_label": ad_label, "soyad_label": soyad_label,
        "gecerlilik_items": gecerlilik["items"],
    }
