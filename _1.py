import cv2

wan = cv2.imread("wan.jpeg")  # görseli bir değişkene atadık
cv2.imshow("avatar wan", wan)   # göster dedik
cv2.waitKey(0)   # görselin ekranda kalmasını sağlar
cv2.destroyAllWindows()  # tuşa basınca kapanır

print(str(wan.item(100, 100, 2)))  # görselin 100x100 pikselinin KIRMIZI değeri -- 2
print(str(wan.item(100, 100, 1)))  # görselin 100x100 pikselinin YEŞİL değeri -- 1
print(str(wan.item(100, 100, 0)))  # görselin 100x100 pikselinin MAVİ değeri -- 0

wan.itemset((100, 100, 2), 0)  # görselin 100x100 pikselindeki kırmızı değerini 0 yap.

print(str(wan.shape))  # görselin boyutunu öğreniriz.
print(str(len(wan.shape)))  # görselin kaç parametreye sahip olduğunu görürüz
# 3 parametre RGB renkli, 2 parametre siyah beyaz
print(str(wan.size))   # görselin kaç piksel boyutunda olduğunu söyler. RGB renkli ise 3 katını gösterir.
print(wan.dtype)  # çok önemli bu. 2 görüntüyü toplayabilmek için 2 görselin de data type'ları aynı olmalıdır. uint8 gibi

#roi = wan[310:390.172:230] # region of image
# virgülün solu Y ekseninde alınacak alan
# virgülün sağı X ekseninde alınacak alan

#wan[310:390.172:230] = roi

b, g, r = cv2.split(wan)
# resmi kanallarına ayırır. b, g, r bizim verdiğimiz değişkenler

wan[:, :, 0] = 0 # köşeli parantez içindeki 0 değerinin 1 veya 2 olmasına göre kırmızı veya yeşille de oynayabiliriz
# mavi değerini sıfırlar
# resim maviden uzak görünür

wan[:, :, 0] = 255 # köşeli parantez içindeki 0 değerinin 1 veya 2 olmasına göre kırmızı veya yeşille de oynayabiliriz
# mavi değerini 255 yapar
# resim mavimsi görünür

