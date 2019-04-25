from scipy import ndimage
from scipy.ndimage.filters import convolve
import numpy as np

class Canny:
    def __init__(self, imgs, sigma=1, tam_gauss=5, pixel_fraco=75, pixel_forte=255, fraca_intensidade=0.05,
                 forte_intensidade=0.15):
        self.imgs = imgs #conjunto de imgs de entrada
        self.imgs_final = [] #conjunto de imgs resultante
        self.img_smoothed = None #imagem pós filtro gauss
        self.gradientMat = None #imagem pós calc_gradiente
        self.thetaMat = None #ângulo theta do gradiente
        self.nonMaxImg = None #imagem pós max_supression
        self.thresholdImg = None #imagem pós identificado pixeis fortes e fracos
        self.pixel_fraco = pixel_fraco #param para identificar pixeis fracos
        self.pixel_forte = pixel_forte #param para identificar pixeis fortes
        self.sigma = sigma
        self.tam_gauss = tam_gauss #tamanho da matriz [AxA] para filtro gauss
        self.fraca_intensidade = fraca_intensidade
        self.forte_intensidade = forte_intensidade
        return

    def filtro_gauss(self, size, sigma=1): #redução do ruído da imagem
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal #Gaussian filter kernel equation
        return g

    def calc_gradiente(self, img): #calcula o gradiente da img
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) #sobel filter direção x (horizontal)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32) #sobel filter direção y (vertical)

        Ix = ndimage.filters.convolve(img, Kx) #derivada x
        Iy = ndimage.filters.convolve(img, Ky) #derivada y

        G = np.hypot(Ix, Iy) #sqrt(Ix**2 + Iy**2)
        G = G / G.max() * 255 #normaliza
        theta = np.arctan2(Iy, Ix) #angulo theta
        return (G, theta)

    def non_max_suppression(self, img, theta_grad): #afina as arestas encontradas no gradiente da img
        M, N = img.shape #M = linhas, N = colunas
        Z = np.zeros((M, N), dtype=np.int32) #matriz de zeros do tamanho da img
        angle = theta_grad * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angulo de 0 graus
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angulo de 45 graus
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angulo de 90 graus
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angulo de 135 graus
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]
                    # verifica se o pixel possui intensidade maior que o atual
                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def threshold(self, img): #identifica pixeis com intensidade forte e fraca da img

        forte_intensidade = img.max() * self.forte_intensidade;
        fraca_intensidade = forte_intensidade * self.fraca_intensidade;

        M, N = img.shape #M = linhas, N = colunas
        res = np.zeros((M, N), dtype=np.int32) #matriz de zeros do tamanho da img

        fraco = np.int32(self.pixel_fraco) #pega o param da classe como ref
        forte = np.int32(self.pixel_forte)

        pixeis_forte_i, pixeis_forte_j = np.where(img >= forte_intensidade) #acha todos os pixeis fortes da img
        pixeis_fraco_i, pixeis_fraco_j = np.where((img <= forte_intensidade) & (img >= fraca_intensidade)) #acha todos os pixeis fracos da img

        res[pixeis_forte_i, pixeis_forte_j] = forte
        res[pixeis_fraco_i, pixeis_fraco_j] = fraco

        return (res)

    def hysteresis(self, img): #transforma pixeis em intensidade alta olhando pela sua vizinhança

        M, N = img.shape
        fraco = self.pixel_fraco
        forte = self.pixel_forte

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i, j] == fraco):
                    try:
                        #se algum pixel ao redor de img[i,j] for forte
                        if ((img[i + 1, j - 1] == forte) or (img[i + 1, j] == forte) or (img[i + 1, j + 1] == forte)
                                or (img[i, j - 1] == forte) or (img[i, j + 1] == forte)
                                or (img[i - 1, j - 1] == forte) or (img[i - 1, j] == forte) 
                                or (img[i - 1, j + 1] == forte)):
                            img[i, j] = forte #transforma o pixel img[i,j] em forte
                        else:
                            img[i, j] = 0 #se não ignora ele
                    except IndexError as e:
                        pass

        return img

    def detect(self):
        imgs_final = []
        for i, img in enumerate(self.imgs):
            self.img_smoothed = convolve(img, self.filtro_gauss(self.tam_gauss, self.sigma))
            self.gradientMat, self.thetaMat = self.calc_gradiente(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return self.imgs_final

