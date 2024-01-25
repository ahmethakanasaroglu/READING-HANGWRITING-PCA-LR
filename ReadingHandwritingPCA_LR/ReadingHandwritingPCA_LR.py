#%% Kullanacağımı MNIST elyazısı rakamları veritabanında(sklearn içinde gelmektedir) 784 feature sütunu mevcut ve training set olarak 60k örnek veri ve a 10k örneklik test seti bulunmaktadır.

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA           # feature sayısını azaltmak için
from sklearn.datasets import fetch_openml         # mnist datasetini yüklemek için gerekli
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')     # 28x28   # dataframe'i tanımladık
mnist.data.shape


#%% Mnist veriseti içindeki rakam fotolarını görmek için fonksiyon tanımlayalım

# Parametre olarak dataframe ve ilgili veri fotografının index numarasını alsın
def showimage(dframe, index):
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)
    
    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()

# örnek kullanımı
showimage(mnist.data, 0)

#%% Split data -->  training ve test set

train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)  # test ve train oranı = 1/7 ve 6/7 # random olarak ayır diye 0 verdik
type(train_img)    # DataFrame

test_img_copy = test_img.copy() # rakam tahminlerimiiz check etmek için train_img dataframe'ini kopyalıyoruz, çünkü az sonra değişecek
showimage(test_img_copy,0)

#%% Datayı Scale etmemiz lazım -- çünkü PCA scale edilmemiş verilerde hatalı sonuclar verebiliyor bu nedenle mutlaka scaling işleminden geçiriyoruz. Bu amaçla da StandardScaler kullanıyoruz.    

scaler = StandardScaler()

scaler.fit(train_img)   # scaler'i sadece training set üzerinde fit yapmamız yeterli

train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)    # transform işlemini hem training sete hem test sete yapmamız gerekiyor

#%% PCA İŞLEMİ   

pca = PCA(.95) # variance'ın 95% oranında korunmasını istedigimizi belirtiyoruz.

pca.fit(train_img)  # PCA'i sadece training sete yapmamız yeterli

print(pca.n_components_)    # 784 boyutu yüzde 95 koruyarak kaça düşürdü bakıyoruz -- 154'e düşmüş 

train_img = pca.transform(train_img)    # yukarıda modeli olusturup fit etmistik. şimdi datasetlere uyarlıyoruz.
test_img = pca.transform(test_img)     # şimdi transform işlemiyle hem train hem test veri setimizin boyutlarını 784'den 154'e düşürelim


#%% Logistic Regression modelimizi(machine learning) PCA işleminden geçirilmiş veri setimiz üzerinde uygulayacazğız

logisticRegr = LogisticRegression(solver= 'lbfgs', max_iter= 10000) # default solver cok yavas calıstıgı icin daha hızlı olan 'lbfgs' seçerek logisticregression nesnemizi olusturuyoruz

logisticRegr.fit(train_img, train_lbl)   # LR Modelini eğitme -- train datası kullanarak 

logisticRegr.predict(test_img[0].reshape(1,-1))   # model eğitildi üstte. şimdi el yazısı rakamları makine öğrenmesi ile tanıma işlemini gerceklestiriyoruz
showimage(test_img_copy, 0)
logisticRegr.predict(test_img[2].reshape(1,-1))
showimage(test_img_copy, 2)

## ACCURACY ORANI 
logisticRegr.score(test_img, test_lbl)  # acc degerine baktık = 0.9196

































