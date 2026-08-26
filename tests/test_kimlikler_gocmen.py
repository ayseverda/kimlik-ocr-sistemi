import unittest
from datetime import date

from kimlikler.kimlikler_gocmen import (
    gecerlilik_tarihi_bul,
    gocmen_bilgilerini_bul,
    label_tipini_bul,
    satir_label_bul,
    yabanci_no_bul,
)


def kutu(text, x1, y1, x2=None, y2=None, conf=0.90):
    return {
        "text": text,
        "conf": conf,
        "x1": x1,
        "y1": y1,
        "x2": x2 if x2 is not None else x1 + max(35, len(text) * 9),
        "y2": y2 if y2 is not None else y1 + 20,
    }


class LabelVeIsimTestleri(unittest.TestCase):
    def test_ali_ad_etiketi_sayilmaz_ve_deger_olarak_okunur(self):
        self.assertIsNone(label_tipini_bul("ALI"))

        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 20, 100, 60),
            kutu("ALI", 90, 100, 145),
            kutu("SOYADI", 20, 150, 90),
            kutu("YILMAZ", 120, 150, 205),
        ])

        self.assertEqual("ALI", sonuc["ad"])
        self.assertEqual("YILMAZ", sonuc["soyad"])

    def test_tek_kutudaki_label_ve_deger_ayrilir(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI ALI VELI", 20, 100, 190),
            kutu("SOYADI YILMAZ", 20, 150, 200),
        ])

        self.assertEqual("ALI VELI", sonuc["ad"])
        self.assertEqual("YILMAZ", sonuc["soyad"])

    def test_labelin_solundaki_metin_deger_sayilmaz(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ALI", 15, 100, 60),
            kutu("ADI", 100, 100, 140),
        ])
        self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_labelden_cok_uzak_metin_deger_sayilmaz(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 20, 100, 60),
            kutu("RASGELE", 700, 100, 790),
        ])
        self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_dusuk_guvenli_fuzzy_label_reddedilir(self):
        satir = [kutu("SOYADL", 20, 100, 95, conf=0.10), kutu("YILMAZ", 120, 100)]
        self.assertIsNone(satir_label_bul(satir, "soyad"))

    def test_baba_adi_satirindaki_adi_ad_alani_sayilmaz(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("BABA", 20, 100, 70),
            kutu("ADI", 80, 100, 120),
            kutu("MEHMET", 150, 100, 230),
        ])
        self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_turkce_disi_latin_harfleri_ve_apostrof_korunur(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 20, 100, 60),
            kutu("Élodie", 90, 100, 160),
            kutu("SOYADI", 20, 150, 90),
            kutu("O'Connor-García", 120, 150, 280),
        ])

        self.assertEqual("ÉLODIE", sonuc["ad"])
        self.assertEqual("O'CONNOR-GARCÍA", sonuc["soyad"])

    def test_farkli_ocr_satirina_dusen_uzun_ad_yakinlik_fallbackiyle_okunur(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 30, 100, 90, 120),
            # Merkez farki 26 px: 20 px yukseklikteki kutular mevcut 22 px
            # satir gruplama toleransini asar, fakat ayni basili alandadir.
            kutu("ESRA SABAH MUHAMMED ALI", 390, 126, 690, 146),
            kutu("SOYADI", 30, 160, 110, 180),
            kutu("KRUT", 390, 162, 455, 182),
        ])

        self.assertEqual("ESRA SABAH MUHAMMED ALI", sonuc["ad"])
        self.assertEqual("KRUT", sonuc["soyad"])

    def test_uzun_addaki_tekil_ic_rakam_ocr_hatasi_tum_alani_kaybettirmez(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 25, 100, 65, 120),
            # Net baskidaki YOUNUS, OCR tarafindan Y0UNUS okunmus olabilir.
            kutu("ILHAM MUSTAFA Y0UNUS", 380, 126, 650, 146, conf=0.91),
            kutu("SOYADI", 25, 160, 105, 180),
            kutu("AL WIIB", 380, 162, 480, 182, conf=0.94),
        ])

        self.assertEqual("ILHAM MUSTAFA YOUNUS", sonuc["ad"])
        self.assertEqual("AL WIIB", sonuc["soyad"])

    def test_sayi_ve_sonda_rakam_iceren_metin_isim_yapilmaz(self):
        for sahte in ("2025", "KAYIT 2025", "ILHAM1"):
            with self.subTest(sahte=sahte):
                sonuc = gocmen_bilgilerini_bul([
                    kutu("ADI", 25, 100, 65, 120),
                    kutu(sahte, 380, 100, 520, 120),
                ])
                self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_dikey_oynayan_uzun_ad_parcalari_komsu_satira_tasinmadan_birlesir(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 25, 100, 65, 120),
            kutu("ILHAM MUSTAFA", 380, 124, 535, 144, conf=0.93),
            # Merkez, iki label orta noktasini az miktarda asiyor; kutunun ust
            # kismi hala ad satirinda ve yatay olarak ilk parcaya bitisik.
            kutu("YOUNUS", 540, 136, 620, 156, conf=0.90),
            kutu("SOYADI", 25, 160, 105, 180),
            kutu("AL WIIB", 380, 162, 480, 182, conf=0.94),
        ])

        self.assertEqual("ILHAM MUSTAFA YOUNUS", sonuc["ad"])
        self.assertEqual("AL WIIB", sonuc["soyad"])

    def test_yakinlik_fallbacki_farkli_goruntu_olceginde_de_calisir(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 60, 200, 180, 240),
            # Tum olculer iki katina yakinken sabit piksel esigi kullanilmamali.
            kutu("ESRA SABAH", 780, 250, 1080, 290),
            kutu("SOYADI", 60, 330, 220, 370),
            kutu("KRUT", 780, 334, 910, 374),
        ])

        self.assertEqual("ESRA SABAH", sonuc["ad"])
        self.assertEqual("KRUT", sonuc["soyad"])

    def test_ad_fallbacki_soyad_baba_ve_anne_satirina_tasinmaz(self):
        for diger_label, diger_deger in (
            ("SOYADI", "KRUT"),
            ("BABA ADI", "MEHMET"),
            ("ANNE ADI", "MERYEM"),
        ):
            with self.subTest(diger_label=diger_label):
                sonuc = gocmen_bilgilerini_bul([
                    kutu("ADI", 20, 100, 60, 120),
                    kutu(diger_label, 20, 140, 115, 160),
                    # Ad label'ina olcek olarak yakin gorunur; ancak diger alan
                    # merkezine daha yakindir ve ad degeri olarak alinmamalidir.
                    kutu(diger_deger, 120, 124, 210, 144),
                ])

                self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_kirpilmis_baba_degeri_soyad_olarak_alinmaz(self):
        # Dar ROI'nin alt kenarinda Baba Adi label'i disarida kalirken degerin
        # yalniz ust parcasi gorunebilir. Onceki yakinlik fallback'i bu 10 px
        # parcayi net SOYADI label'inin degeri sayiyordu.
        sonuc = gocmen_bilgilerini_bul([
            kutu("SOYADI", 25, 536, 120, 578, conf=0.96),
            kutu("TAREQ KHUDHUR", 390, 578, 560, 588, conf=0.94),
        ])

        self.assertEqual("Bulunamadi", sonuc["soyad"])

    def test_adi_labeli_yokken_soyad_satirinin_ustundeki_ad_okunur(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("YABANCI KIMLIK NO", 25, 80, 210, 100),
            kutu("99251771180", 390, 80, 530, 100),
            # ADI label'i EasyOCR sonucunda hic yok.
            kutu("ESRA SABAH MUHAMMED ALI", 390, 130, 690, 150, conf=0.93),
            # Net kartta label bazen SOYIA gibi hatali ama yuksek guvenli okunur.
            kutu("SOYIA", 25, 180, 105, 200, conf=0.96),
            kutu("KRUT", 390, 180, 455, 200, conf=0.94),
            kutu("BABA ADI", 25, 230, 120, 250),
            kutu("ABDULKAREEM", 390, 230, 535, 250),
        ])

        self.assertEqual("ESRA SABAH MUHAMMED ALI", sonuc["ad"])
        self.assertEqual("KRUT", sonuc["soyad"])
        self.assertIsNone(sonuc["ad_label"])

    def test_soyadi_labeli_yokken_ad_satirinin_altindaki_soyad_okunur(self):
        sonuc = gocmen_bilgilerini_bul([
            kutu("ADI", 25, 100, 65, 120, conf=0.96),
            kutu("ESRA SABAH", 380, 100, 515, 120, conf=0.94),
            # SOYADI label'i EasyOCR sonucunda hic yok.
            kutu("KRUT", 380, 150, 445, 170, conf=0.92),
            kutu("BABA ADI", 25, 200, 120, 220),
            kutu("MEHMET", 380, 200, 465, 220),
        ])

        self.assertEqual("ESRA SABAH", sonuc["ad"])
        self.assertEqual("KRUT", sonuc["soyad"])
        self.assertIsNone(sonuc["soyad_label"])

    def test_labelsiz_ad_fallbacki_ykn_ve_header_metinlerini_ad_saymaz(self):
        for metin in ("YABANCI", "KIMLIK BELGESI", "BASVURU SAHIBI"):
            with self.subTest(metin=metin):
                sonuc = gocmen_bilgilerini_bul([
                    kutu(metin, 390, 130, 560, 150, conf=0.95),
                    kutu("SOYADI", 25, 180, 105, 200, conf=0.96),
                    kutu("KRUT", 390, 180, 455, 200, conf=0.94),
                ])

                self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_labelsiz_soyad_fallbacki_baba_ve_anne_degerini_almaz(self):
        for label, deger in (("BABA ADI", "MEHMET"), ("ANNE ADI", "MERYEM")):
            with self.subTest(label=label):
                sonuc = gocmen_bilgilerini_bul([
                    kutu("ADI", 25, 100, 65, 120, conf=0.96),
                    kutu("ESRA", 390, 100, 455, 120, conf=0.94),
                    kutu(label, 25, 150, 120, 170),
                    kutu(deger, 390, 150, 480, 170, conf=0.95),
                ])

                self.assertEqual("Bulunamadi", sonuc["soyad"])

    def test_soyad_satiri_tamamen_kayipsa_iki_satir_alttaki_baba_adi_alinmaz(self):
        sonuc = gocmen_bilgilerini_bul([
            # YKN ve ADI label araligi, tablonun bir satir adimini verir.
            kutu("YABANCI KIMLIK NO", 25, 75, 210, 95),
            kutu("ADI", 25, 100, 65, 120, conf=0.96),
            kutu("ILHAM MUSTAFA YOUNUS", 380, 100, 650, 120, conf=0.94),
            # SOYADI label/degeri OCR'da yok; Baba Adi label'i de dar ROI'nin
            # disinda, ama degeri iki satir asagidan sizmis durumda.
            kutu("MUSTAFA YOUNUS", 380, 150, 550, 170, conf=0.95),
        ])

        self.assertEqual("ILHAM MUSTAFA YOUNUS", sonuc["ad"])
        self.assertEqual("Bulunamadi", sonuc["soyad"])

    def test_labelsiz_alan_belirsiz_veya_dusuk_guvenliyse_secilmez(self):
        vakalar = (
            [
                kutu("BIRINCI", 390, 120, 475, 140),
                kutu("IKINCI", 390, 150, 475, 170),
            ],
            [kutu("ESRA", 390, 130, 455, 150, conf=0.35)],
            [kutu("ESRA", 250, 130, 315, 150, conf=0.95)],
        )
        for ek_kutular in vakalar:
            with self.subTest(ek_kutular=ek_kutular):
                sonuc = gocmen_bilgilerini_bul([
                    *ek_kutular,
                    kutu("SOYADI", 25, 180, 105, 200, conf=0.96),
                    kutu("KRUT", 390, 180, 455, 200, conf=0.94),
                ])

                self.assertEqual("Bulunamadi", sonuc["ad"])

        dusuk_label_guvenli = gocmen_bilgilerini_bul([
            kutu("ESRA", 390, 130, 455, 150, conf=0.95),
            kutu("SOYIA", 25, 180, 105, 200, conf=0.40),
            kutu("KRUT", 390, 180, 455, 200, conf=0.94),
        ])
        self.assertEqual("Bulunamadi", dusuk_label_guvenli["ad"])


class YabanciKimlikNoTestleri(unittest.TestCase):
    def test_etiketsiz_temiz_ve_benzersiz_numara_korunur(self):
        sonuc = yabanci_no_bul([kutu("99086835446", 100, 100, conf=0.92)])
        self.assertEqual("99086835446", sonuc["deger"])

    def test_metin_icine_gomulu_rastgele_numara_reddedilir(self):
        sonuc = yabanci_no_bul([kutu("KAYIT 99086835446 TEST", 20, 100, 260)])
        self.assertIsNone(sonuc)

    def test_dogrulanmis_ykn_hucresindeki_tek_gurultulu_numara_ayiklanir(self):
        item = kutu("| 99011944816 ;", 350, 100, 610, conf=0.39)
        item.update({
            "_dogrudan_hucre_ocr": True,
            "_fallback_alan": "kimlik_no",
        })

        sonuc = yabanci_no_bul([
            kutu("YABANCI KIMLIK NO", 20, 100, 300, conf=1.0),
            item,
        ])

        self.assertEqual("99011944816", sonuc["deger"])

    def test_birden_fazla_labelsiz_farkli_numara_belirsizdir(self):
        sonuc = yabanci_no_bul([
            kutu("99086835446", 20, 100),
            kutu("99488452166", 20, 180),
        ])
        self.assertIsNone(sonuc)

    def test_tokenlara_bolunmus_labele_yakin_numara_secilir(self):
        sonuc = yabanci_no_bul([
            kutu("YABANCI", 20, 100, 95),
            kutu("KIMLIK", 105, 100, 165),
            kutu("NO", 175, 100, 200),
            kutu("99O86835446", 225, 100, 350),
            kutu("99488452166", 20, 210, 145),
        ])
        self.assertEqual("99086835446", sonuc["deger"])

    def test_labelsiz_dusuk_guvenli_numara_reddedilir(self):
        sonuc = yabanci_no_bul([kutu("99086835446", 20, 100, conf=0.20)])
        self.assertIsNone(sonuc)

    def test_tek_rakam_ocr_hatasi_checksum_gecmiyorsa_reddedilir(self):
        # Gorseldeki dogru ...446 yerine OCR'in urettigi ...448 degeri.
        sonuc = yabanci_no_bul([
            kutu("YABANCI KIMLIK NO", 20, 100, 200),
            kutu("99086835448", 225, 100, 350, conf=0.96),
        ])
        self.assertIsNone(sonuc)


class GecerlilikTarihiTestleri(unittest.TestCase):
    def test_yalniz_geometrisi_dogrulanmis_sag_hucre_tek_bitis_kabul_edilir(self):
        bitis = kutu("31.12.2028", 420, 100, 540, conf=0.91)
        bitis.update({
            "_bitis_hucresi": True,
            "_geometri_dogrulandi": True,
            "_dogrudan_hucre_ocr": True,
        })

        sonuc = gecerlilik_tarihi_bul([bitis])

        self.assertIsNone(sonuc["baslangic"])
        self.assertEqual(date(2028, 12, 31), sonuc["bitis"])
        self.assertEqual([bitis], sonuc["items"])

    def test_labelsiz_ve_isaretsiz_tek_tarih_bitis_sayilmaz(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("31.12.2028", 420, 100, 540, conf=0.99),
        ])

        self.assertIsNone(sonuc["bitis"])
        self.assertEqual("kontrol_edilemedi", sonuc["durum"])

    def test_label_yanindaki_tek_tarih_bile_sag_hucre_kaniti_yoksa_reddedilir(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("BELGENIN GECERLILIK TARIHI", 20, 100, 300),
            kutu("31.12.2028", 420, 100, 540, conf=0.99),
        ])

        self.assertIsNone(sonuc["bitis"])

    def test_labele_yakin_tarih_cifti_diger_satira_onceliklidir(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("01.01.1990", 20, 50, 120),
            kutu("02.02.1990", 150, 50, 250),
            kutu("BELGENIN GECERLILIK TARIHI", 20, 150, 285),
            kutu("01.01.2024", 80, 190, 180),
            kutu("31.12.2028", 220, 190, 320),
        ])
        self.assertEqual(date(2024, 1, 1), sonuc["baslangic"])
        self.assertEqual(date(2028, 12, 31), sonuc["bitis"])

    def test_baslangic_bitisten_sonraysa_cift_reddedilir(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("BELGENIN GECERLILIK TARIHI", 20, 100, 285),
            kutu("31.12.2030", 80, 140, 180),
            kutu("01.01.2020", 220, 140, 320),
        ])
        self.assertEqual("kontrol_edilemedi", sonuc["durum"])
        self.assertIsNone(sonuc["baslangic"])

    def test_ayni_satirda_etiketten_sonraki_cift_secilir(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("01.01.1990", 10, 100, 110),
            kutu("BELGENIN GECERLILIK TARIHI", 130, 100, 390),
            kutu("01.01.2024", 420, 100, 520),
            kutu("31.12.2028", 550, 100, 650),
        ])
        self.assertEqual(date(2024, 1, 1), sonuc["baslangic"])
        self.assertEqual(date(2028, 12, 31), sonuc["bitis"])

    def test_birden_fazla_labelsiz_tarih_satiri_belirsizdir(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("01.01.2020", 20, 50, 120),
            kutu("01.01.2021", 150, 50, 250),
            kutu("01.01.2022", 20, 150, 120),
            kutu("01.01.2023", 150, 150, 250),
        ])
        self.assertEqual("kontrol_edilemedi", sonuc["durum"])
        self.assertIsNone(sonuc["baslangic"])

    def test_tek_labelsiz_tarih_cifti_geriye_uyumlu_olarak_okunur(self):
        sonuc = gecerlilik_tarihi_bul([
            kutu("01.01.2024", 20, 100, 120),
            kutu("31.12.2028", 150, 100, 250),
        ])
        self.assertEqual(date(2024, 1, 1), sonuc["baslangic"])
        self.assertEqual(date(2028, 12, 31), sonuc["bitis"])


if __name__ == "__main__":
    unittest.main()
