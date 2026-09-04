# -*- coding: utf-8 -*-
"""Dışarıdan gelen kimlik listesi (ör. muhtarın gönderdiği Excel) ile
uygulamanın çıkardığı sonuçları karşılaştırır.

Amaç: "kaç kişi tuttu, kimler tutmadı, hangi kimlikler eksik" sorusuna
belgeye dayalı cevap vermek. Eşleştirme önce kimlik numarasından, numara
yoksa/ tutmazsa ad-soyaddan yapılır.

Gelen dosyanın sütun düzeni sabit değil; başlıklardan, olmazsa hücre
içeriğinden (11 haneli geçerli numara / harf ağırlıklı metin) bulunuyor.
Hangi sütunun ne sayıldığı sonuçta bildiriliyor ki yanlış okunduğunda
kullanıcı görebilsin.
"""

import os
import re

try:
    import openpyxl
except ImportError:
    openpyxl = None

from metin_ayiklama import normalize_text, benzerlik, tc_kimlik_gecerli_mi


OKUNABILIR_UZANTILAR = (".xlsx", ".xlsm")

# Ad karşılaştırmasında bu orandan düşük benzerlik "farklı ad" sayılır.
AD_ESLESME_ESIGI = 0.90

# Numara yokken ada göre eşleştirme bu orandan yüksek benzerlik ister.
ISIMLE_ESLESME_ESIGI = 0.93

# Başlık satırdan itibaren başlamak zorunda değil (üstte doküman başlığı,
# logo, tarih gibi satırlar olabilir); ilk bu kadar satır taranıyor.
BASLIK_ARAMA_SATIRI = 40
ICERIK_ARAMA_SATIRI = 250


def _metin(deger):
    if deger is None:
        return ""
    if isinstance(deger, float) and deger.is_integer():
        return str(int(deger))
    return str(deger).strip()


def tc_temizle(deger):
    """Hücre değerinden 11 haneli kimlik numarasını çıkarır."""
    rakam = re.sub(r"\D", "", _metin(deger))
    return rakam if len(rakam) == 11 else ""


# Başlık kelimeleri normalize_text'ten (büyük harf + Türkçe harfler sadeleşmiş)
# geçtiği için "ad", "Ad", "AD", "İsim", "isim" gibi her yazımı aynı kelimeye
# indirgiyor — büyük/küçük harf ayrımı burada zaten devre dışı.
_AD_KELIMELERI = {"AD", "ADI", "ISIM", "ISMI"}
_SOYAD_KELIMELERI = {"SOYAD", "SOYADI", "SOYISIM", "SOYISMI"}


def _basliktan_sutunlar(satir_degerleri):
    """Başlık satırındaki metinlerden sütun eşlemesi kurar.

    Kelime bazlı eşleştirme kullanılıyor (ör. "Mahalle Adı" içindeki "ADI"
    kelimesi "AD" sütunuyla karışmasın diye tam kelime aranıyor, alt dize
    değil). TC için ise noktalama/boşluk farklılıklarını (T.C., TC No, Tc
    Numarası...) tolere etmek üzere boşluksuz metinde aranıyor.

    Excel'in en üstünde genelde bir logo/açıklama/uyarı bloğu olur (ör.
    "MUHTAR İSİM SOYİSİM - TELEFON -TC NO" gibi birleştirilmiş bir hücre);
    bu tür metinler TEK hücrede hem TC hem AD/SOYAD sinyali taşıdığı için
    gerçek bir sütun başlığıyla karıştırılabiliyordu. Bir hücrede TC pattern'i
    AD/SOYAD kelimesiyle birlikte geçiyorsa, bu "karışık" hücre hiçbir alana
    atanmıyor — gerçek başlıklar tek bir kavramı adlandırır, birden fazlasını
    aynı anda değil."""
    sutunlar = {}
    for indeks, deger in enumerate(satir_degerleri):
        norm = normalize_text(_metin(deger))
        if not norm:
            continue
        norm_bosluksuz = norm.replace(" ", "")
        kelimeler = set(norm.split())

        ad_var = bool(kelimeler & _AD_KELIMELERI)
        soyad_var = bool(kelimeler & _SOYAD_KELIMELERI)
        # Yalnız "No" (sıra numarası, fiş no, kayıt no...) TC sinyali SAYILMAZ —
        # gerçek dünyada bu neredeyse hep bir sıra numarası sütunudur ve TC
        # kelimesi başka bir sütunda ayrıca yazar.
        tc_var = (
            "TC" in norm_bosluksuz or "KIMLIKNO" in norm_bosluksuz
            or "KIMLIKNUMARA" in norm_bosluksuz
        )

        if tc_var and (ad_var or soyad_var):
            continue   # karışık/açıklama hücresi — atla

        if "tc" not in sutunlar and tc_var:
            sutunlar["tc"] = indeks
        elif "tam_ad" not in sutunlar and ad_var and soyad_var:
            # Aynı hücrede hem "ad" hem "soyad" kelimesi geçiyor: "Ad Soyad",
            # "Adı Soyadı", "İsim Soyisim" gibi birleşik sütun.
            sutunlar["tam_ad"] = indeks
        elif "soyad" not in sutunlar and soyad_var:
            sutunlar["soyad"] = indeks
        elif "ad" not in sutunlar and ad_var:
            sutunlar["ad"] = indeks
    return sutunlar


def _icerikten_sutunlar(satirlar):
    """Başlık yoksa: hangi sütun numara, hangileri isim tutuyor, içeriğe bak."""
    if not satirlar:
        return {}

    sutun_sayisi = max(len(s) for s in satirlar)
    tc_puan = [0] * sutun_sayisi
    isim_puan = [0] * sutun_sayisi

    for satir in satirlar:
        for i, deger in enumerate(satir):
            metin = _metin(deger)
            if not metin:
                continue
            if tc_temizle(metin) and tc_kimlik_gecerli_mi(tc_temizle(metin)):
                tc_puan[i] += 1
            elif len(normalize_text(metin).replace(" ", "")) >= 3 and not metin.isdigit():
                isim_puan[i] += 1

    sutunlar = {}
    if tc_puan and max(tc_puan) >= 2:
        sutunlar["tc"] = tc_puan.index(max(tc_puan))

    isim_sutunlari = [i for i, p in enumerate(isim_puan) if p >= 2]
    isim_sutunlari.sort(key=lambda i: isim_puan[i], reverse=True)
    if len(isim_sutunlari) >= 2:
        # İki güçlü isim sütunu: soldaki ad, sağdaki soyad kabul edilir.
        ilk, ikinci = sorted(isim_sutunlari[:2])
        sutunlar["ad"], sutunlar["soyad"] = ilk, ikinci
    elif isim_sutunlari:
        sutunlar["tam_ad"] = isim_sutunlari[0]
    return sutunlar


def _baslik_puani(aday, satir_degerleri):
    """Bir satırın gerçek sütun başlığı olma ihtimalini puanlar.

    Çok alan bulunması iyi (tc+ad+soyad üçü birden en güçlü sinyal), hücre
    metinlerinin KISA olması iyi — gerçek başlıklar ("Ad", "TC Kimlik No")
    kısadır; üstteki açıklama/uyarı satırları uzun cümlelerdir. Tuple
    karşılaştırması: önce alan sayısı, eşitlikte kısa hücre kazanır."""
    uzunluklar = [
        len(_metin(satir_degerleri[idx]))
        for idx in aday.values() if idx < len(satir_degerleri)
    ]
    ort_uzunluk = sum(uzunluklar) / len(uzunluklar) if uzunluklar else 999.0
    return (len(aday), -ort_uzunluk)


def _sayfadan_kayitlar(sayfa):
    satirlar = [list(s) for s in sayfa.iter_rows(values_only=True)]
    if not satirlar:
        return [], {}

    sutunlar = {}
    baslik_satiri = None
    en_iyi_puan = None
    # İlk eşleşen değil, EN İYİ puanlı satır seçiliyor: sayfanın üstünde
    # açıklama/uyarı metinleri sahte biçimde "başlık gibi" görünebiliyor
    # (bkz. _basliktan_sutunlar docstring'i). Eşit puanda daha AŞAĞIDAKİ
    # satır tercih edilir — gerçek başlık genelde veriye en yakın olandır.
    for i, satir in enumerate(satirlar[:BASLIK_ARAMA_SATIRI]):
        aday = _basliktan_sutunlar(satir)
        if not ("tc" in aday or "tam_ad" in aday or ("ad" in aday and "soyad" in aday)):
            continue
        puan = _baslik_puani(aday, satir)
        if en_iyi_puan is None or puan >= en_iyi_puan:
            sutunlar, baslik_satiri, en_iyi_puan = aday, i, puan

    if not sutunlar:
        sutunlar = _icerikten_sutunlar(satirlar[:ICERIK_ARAMA_SATIRI])

    if not sutunlar:
        return [], {}

    kayitlar = []
    basla = (baslik_satiri + 1) if baslik_satiri is not None else 0
    for i, satir in enumerate(satirlar[basla:], start=basla + 1):
        def hucre(anahtar):
            indeks = sutunlar.get(anahtar)
            return _metin(satir[indeks]) if indeks is not None and indeks < len(satir) else ""

        tc = tc_temizle(hucre("tc"))
        ad = hucre("ad")
        soyad = hucre("soyad")
        tam_ad = hucre("tam_ad") or (f"{ad} {soyad}".strip())

        if not tc and not normalize_text(tam_ad):
            continue
        kayitlar.append({
            "tc": tc,
            "ad": ad,
            "soyad": soyad,
            "tam_ad": tam_ad.strip(),
            "satir": i,
        })

    return kayitlar, {
        "baslik_satiri": (baslik_satiri + 1) if baslik_satiri is not None else None,
        "sutunlar": {k: v + 1 for k, v in sutunlar.items()},   # 1 tabanlı
    }


def listeyi_oku(yol):
    """Dış listeyi okur. Doner: (kayitlar, bilgi). Okunamazsa ValueError."""
    if openpyxl is None:
        raise ValueError("openpyxl kurulu değil; Excel okunamıyor.")
    if not str(yol).lower().endswith(OKUNABILIR_UZANTILAR):
        raise ValueError("Yalnızca .xlsx / .xlsm dosyaları okunabiliyor.")

    kitap = openpyxl.load_workbook(yol, data_only=True, read_only=True)
    try:
        en_iyi, en_iyi_bilgi, en_iyi_sayfa = [], {}, ""
        for sayfa in kitap.worksheets:
            kayitlar, bilgi = _sayfadan_kayitlar(sayfa)
            if len(kayitlar) > len(en_iyi):
                en_iyi, en_iyi_bilgi, en_iyi_sayfa = kayitlar, bilgi, sayfa.title
    finally:
        kitap.close()

    if not en_iyi:
        raise ValueError(
            "Dosyada kimlik numarası veya ad-soyad sütunu bulunamadı. "
            "Sütun başlıklarında 'TC Kimlik No', 'Ad', 'Soyad' geçmesi yeterli."
        )

    en_iyi_bilgi["sayfa"] = en_iyi_sayfa
    en_iyi_bilgi["toplam"] = len(en_iyi)
    return en_iyi, en_iyi_bilgi


# =========================================================
# KARŞILAŞTIRMA
# =========================================================

def _bizim_kayit(satir):
    ad = _metin(satir.get("Ad"))
    soyad = _metin(satir.get("Soyad"))
    ad = "" if ad == "Bulunamadi" else ad
    soyad = "" if soyad == "Bulunamadi" else soyad
    tc = tc_temizle(satir.get("Kimlik No"))
    return {
        "tc": tc,
        "ad": ad,
        "soyad": soyad,
        "tam_ad": f"{ad} {soyad}".strip(),
        "satir": satir,
    }


def _adlar_uyuyor(a, b):
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return None            # karşılaştırılamıyor
    return benzerlik(a, b) >= AD_ESLESME_ESIGI


def karsilastir(bizim_satirlar, dis_kayitlar):
    """Dış liste ile bizim sonuçları eşleştirir.

    Döner: {"eslesen", "duzeltilmeli", "eksik", "fazla", "ozet"} — her biri
    satır listesi.
      - "duzeltilmeli": kişi eşleşti ama bir alan tutmuyor — ya numara AYNI
        isim farklı (isim OCR hatası olabilir) ya da isim örtüşüyor numara
        FARKLI (kimlik numarası muhtemelen yanlış okundu). Kayıtta hangisi
        olduğu "tip" alanında ("ad" | "tc") tutuluyor.
      - "eksik": dış listede olup bizde bulunmayanlar (asıl veri kaybı).
      - "fazla": bizde olup listede olmayanlar."""
    bizimkiler = [_bizim_kayit(s) for s in bizim_satirlar]
    bizimkiler = [b for b in bizimkiler if b["tc"] or normalize_text(b["tam_ad"])]

    tc_indeks = {}
    for kayit in bizimkiler:
        if kayit["tc"]:
            tc_indeks.setdefault(kayit["tc"], []).append(kayit)

    eslesen, duzeltilmeli, eksik = [], [], []
    kullanilan = set()

    # 1) Kimlik numarasıyla eşleştir (güçlü anahtar)
    numarasiz = []
    for dis in dis_kayitlar:
        adaylar = [k for k in tc_indeks.get(dis["tc"], []) if id(k) not in kullanilan] if dis["tc"] else []
        if not adaylar:
            numarasiz.append(dis)
            continue

        bizim = adaylar[0]
        kullanilan.add(id(bizim))
        uyum = _adlar_uyuyor(dis["tam_ad"], bizim["tam_ad"])
        kayit = {"dis": dis, "bizim": bizim, "anahtar": "kimlik no"}
        if uyum is False:
            # Numara aynı, isim tutmuyor.
            duzeltilmeli.append({**kayit, "tip": "ad"})
        else:
            eslesen.append(kayit)

    # 2) Numarayla bulunamayanları ad-soyada göre dene
    for dis in numarasiz:
        dis_ad = normalize_text(dis["tam_ad"])
        en_iyi, en_iyi_puan = None, 0.0
        if dis_ad:
            for bizim in bizimkiler:
                if id(bizim) in kullanilan or not normalize_text(bizim["tam_ad"]):
                    continue
                puan = benzerlik(dis_ad, normalize_text(bizim["tam_ad"]))
                if puan > en_iyi_puan:
                    en_iyi, en_iyi_puan = bizim, puan

        if en_iyi is None or en_iyi_puan < ISIMLE_ESLESME_ESIGI:
            eksik.append({"dis": dis})
            continue

        kullanilan.add(id(en_iyi))
        # İsim örtüşüyor ama her iki tarafta da FARKLI bir numara varsa, bu
        # "eşleşti" değil — kimlik numarası muhtemelen yanlış okunmuş ve
        # düzeltilmesi gerekiyor. Eskiden bu durum sessizce "eşleşti" olarak
        # gösteriliyordu.
        if dis["tc"] and en_iyi["tc"] and dis["tc"] != en_iyi["tc"]:
            duzeltilmeli.append({"dis": dis, "bizim": en_iyi, "tip": "tc"})
        else:
            not_metni = "bizde numara okunamadı" if dis["tc"] and not en_iyi["tc"] else ""
            eslesen.append({"dis": dis, "bizim": en_iyi, "anahtar": "ad soyad",
                            "not": not_metni})

    fazla = [{"bizim": b} for b in bizimkiler if id(b) not in kullanilan]

    return {
        "eslesen": eslesen,
        "duzeltilmeli": duzeltilmeli,
        "eksik": eksik,
        "fazla": fazla,
        "ozet": {
            "dis_toplam": len(dis_kayitlar),
            "bizim_toplam": len(bizimkiler),
            "eslesen": len(eslesen),
            "duzeltilmeli": len(duzeltilmeli),
            "eksik": len(eksik),
            "fazla": len(fazla),
        },
    }


def ozet_metni(sonuc, dosya_adi=""):
    o = sonuc["ozet"]
    parcalar = [
        f"{dosya_adi}: {o['dis_toplam']} kayıt" if dosya_adi else f"Listede {o['dis_toplam']} kayıt",
        f"bizde {o['bizim_toplam']} kimlik",
        f"{o['eslesen']} eşleşti",
    ]
    if o["duzeltilmeli"]:
        parcalar.append(f"{o['duzeltilmeli']} kişide veri düzeltilmesi gerekiyor")
    if o["eksik"]:
        parcalar.append(f"{o['eksik']} kayıt bizde YOK")
    if o["fazla"]:
        parcalar.append(f"{o['fazla']} kimlik listede yok")
    return "  •  ".join(parcalar)


def satirlara_don(sonuc):
    """Sonucu tek bir tabloya (dışa aktarım / gösterim için) çevirir.

    Her satırda "_bizim_satir" alanı da bulunur: bizim tabloya karşılık
    gelen ORİJİNAL satır sözlüğüne referans (yalnız arayüz içi kullanım
    için — Excel'e aktarılırken bu alan yazılmaz, satirlara_don çağıranı
    yalnızca kendi başlık listesindeki sütunları dışa aktarmalı)."""
    satirlar = []
    for kayit in sonuc["eslesen"]:
        satirlar.append({
            "Durum": "Eşleşti",
            "Kimlik No": kayit["dis"]["tc"] or kayit["bizim"]["tc"],
            "Listedeki Ad Soyad": kayit["dis"]["tam_ad"],
            "Bizdeki Ad Soyad": kayit["bizim"]["tam_ad"],
            "Not": kayit.get("not", "") or f"eşleştirme: {kayit['anahtar']}",
            "_bizim_satir": kayit["bizim"]["satir"],
        })
    for kayit in sonuc["duzeltilmeli"]:
        if kayit["tip"] == "tc":
            kimlik_no = f"{kayit['bizim']['tc']} → liste: {kayit['dis']['tc']}"
            not_metni = "isim örtüşüyor, numara farklı — muhtemelen yanlış okundu"
        else:
            kimlik_no = kayit["dis"]["tc"]
            not_metni = "numara aynı, ad tutmuyor"
        satirlar.append({
            "Durum": "Eşleşti ama veri düzeltilmesi gerek",
            "Kimlik No": kimlik_no,
            "Listedeki Ad Soyad": kayit["dis"]["tam_ad"],
            "Bizdeki Ad Soyad": kayit["bizim"]["tam_ad"],
            "Not": not_metni,
            "_bizim_satir": kayit["bizim"]["satir"],
        })
    for kayit in sonuc["eksik"]:
        satirlar.append({
            "Durum": "Bizde yok",
            "Kimlik No": kayit["dis"]["tc"],
            "Listedeki Ad Soyad": kayit["dis"]["tam_ad"],
            "Bizdeki Ad Soyad": "",
            "Not": f"listenin {kayit['dis']['satir']}. satırı",
            "_bizim_satir": None,
        })
    for kayit in sonuc["fazla"]:
        satirlar.append({
            "Durum": "Listede yok",
            "Kimlik No": kayit["bizim"]["tc"],
            "Listedeki Ad Soyad": "",
            "Bizdeki Ad Soyad": kayit["bizim"]["tam_ad"],
            "Not": "bizde okundu, listede karşılığı yok",
            "_bizim_satir": kayit["bizim"]["satir"],
        })
    return satirlar
