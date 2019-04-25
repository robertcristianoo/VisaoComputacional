import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import Canny as alg #class Canny

def rgb2gray(rgb): #função básica pra converter para tons de cinza
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def load_data(dir_name = 'images'): #carrega todas as imagens dentro do diretório 'images'
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs


def visualize(imgs, format=None, gray=False): #plota imagens
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        plt_idx = i + 1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()

#INICIA AQUI
imgs = load_data() #carrega as imagens

detector = alg.Canny(imgs, sigma=1.4, tam_gauss=5, fraca_intensidade=0.10, forte_intensidade=0.20, pixel_fraco=100) #seta os params para a classe
imgs_final = detector.detect() #recebe imagem resultante
visualize(imgs_final, 'gray') #exibe