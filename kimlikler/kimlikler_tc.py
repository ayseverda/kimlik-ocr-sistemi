import re


# =========================================================
# TC KİMLİK NUMARASI DOĞRULAMA
# =========================================================

def tc_kimlik_gecerli_mi(no):

    if not no or len(no) != 11 or not no.isdigit() or no[0] == "0":
        return False

    d = [int(x) for x in no]

    d10 = (
        sum(d[0:9:2]) * 7
        -
        sum(d[1:8:2])
    ) % 10

    d11 = sum(d[:10]) % 10

    return (
        d[9] == d10
        and
        d[10] == d11
    )


# =========================================================
# OCR HARF / RAKAM HATALARINI DÜZELT
# =========================================================

def tc_metni_duzelt(text):

    donusum = {
        "O": "0",
        "Q": "0",
        "D": "0",

        "I": "1",
        "İ": "1",
        "L": "1",

        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }

    sonuc = ""

    for karakter in str(text).upper():

        if karakter.isdigit():
            sonuc += karakter

        elif karakter in donusum:
            sonuc += donusum[karakter]

    return sonuc


# =========================================================
# OCR SONUÇLARINDAN TC BUL
# =========================================================

def tc_bul(ocr_sonuclari):

    # =====================================================
    # 1. DİREKT TC
    # =====================================================

    for item in ocr_sonuclari:

        rakamlar = re.sub(
            r"\D",
            "",
            item["text"]
        )

        if (
            len(rakamlar) == 11
            and
            tc_kimlik_gecerli_mi(rakamlar)
        ):
            return rakamlar, item


    # =====================================================
    # 2. OCR HATALARINI DÜZELTİP TEKRAR DENE
    # =====================================================

    for item in ocr_sonuclari:

        aday = tc_metni_duzelt(
            item["text"]
        )

        if (
            len(aday) == 11
            and
            tc_kimlik_gecerli_mi(aday)
        ):
            return aday, item


    return "Bulunamadi", None