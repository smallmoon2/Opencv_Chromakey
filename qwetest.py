import numpy as np, cv2, math

import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D             # 3차원 그래프 라이브러리

def zeropadding(img):
    h,w = img.shape[:2]
    m = 1 << int(np.ceil(np.log2(h)))
    n = 1 << int(np.ceil(np.log2(w)))

    dst = np.zeros((m,n), img.dtype)
    dst[0:h, 0:w] = img[:]
    return dst

def calc_spectrum(complex):
    if complex.ndim == 2:
        dst = abs(complex)
    else:
        dst = cv2.magnitude(complex[:,:,0], complex[:,:,1])
    dst = cv2.log(dst + 1)
    cv2.normalize(dst, dst, 0 , 255 , cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(dst)

def fftshift(img):
    dst = np.zeros(img.shape, img.dtype)
    h,w = dst.shape[:2]
    cy, cx = h//2 , w//2

    dst[h-cy:, w-cx:] = np.copy(img[0:cy, 0:cx])
    dst[0:cy, 0:cx ] = np.copy(img[h-cy:, w-cx:])
    dst[0:cy, w-cx:] = np.copy(img[h-cy:, 0:cx])
    dst[h-cy:, 0:cx] = np.copy(img[0:cy, w - cx:])

    return dst
def exp(knN):
    th = -2 * math.pi * knN
    return complex(math.cos(th), math.sin(th))

def dft(g):
    N = len(g)
    dst = [sum(g[n] * exp(k*n/N ) for n in range(N)) for k in range(N) ]
    return np.array(dst)

def idft(g):
    N = len(g)
    dst = [sum(g[n] * exp(-k*n/N) for n in range(N)) for k in range(N) ]
    return np.array(dst) / N

def dft2(image):
    tmp = [dft(row) for row in image]
    dst = [dft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                   # 전치 환원 후 반환

def idft2(image):
    tmp = [idft(row) for row in image]
    dst = [idft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                   # 전치 환원 후 반환

# def ck_time(mode = 0):
#     global stime
#     if (mode ==0 ):
#        stime = time.perf_counter()
#     elif (mode==1):
#        etime = time.perf_counter()


def butterfly(pair, L, N, dir):
    for k in range(L):                                       # 버터플라이 수행
        Geven, Godd = pair[k], pair[k + L]
        pair[k]     = Geven + Godd * exp(dir * k / N)       # 짝수부
        pair[k + L] = Geven - Godd * exp(dir * k / N)       # 홀수부

def pairing(g, N, dir, start=0, stride=1):


    if N == 1: return [g[start]]
    L = N // 2
    sd = stride * 2

    part1 = pairing(g, L, dir, start, sd)
    part2 = pairing(g, L, dir, start + stride, sd)

    pair = part1 + part2                                     # 결과 병합



    butterfly(pair, L, N, dir)
    return pair

def fft(g):
    return pairing(g, len(g), 1)

def ifft(g):
    fft = pairing(g, len(g), -1)
    return [v / len(g) for v in fft]

def fft2(image):
    pad_img = zeropadding(image)  # 영삽입
    tmp = [fft(row) for row in pad_img]
    dst = [fft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                        # 전치 환원 후 반환

def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                        # 전치 환원 후 반환

def FFT(image, mode = 2):
    if mode == 1: dft = fft2(image)
    elif mode==2: dft = np.fft.fft2(image)
    elif mode==3: dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = fftshift(dft)                              # 셔플링
    spectrum = calc_spectrum(dft)               # 주파수 스펙트럼 영상
    return dft, spectrum

def IFFT(dft, shape, mode=2):
    dft = fftshift(dft)                                 # 역 셔플링
    if mode == 1: img = ifft2(dft).real
    if mode == 2: img = np.fft.ifft2(dft).real
    if mode ==3:  img = cv2.idft(dft, flags= cv2.DFT_SCALE)[:,:,0]
    img = img[:shape[0], :shape[1]]                 # 영삽입 부분 제거
    return cv2.convertScaleAbs(img)



def get_gaussianFilter(shape, R):
    u = np.array(shape)//2
    y = np.arange(-u[0], u[0], 1)
    x = np.arange(-u[1], u[1], 1)
    x, y = np.meshgrid(x, y)
    filter = np.exp(-(x**2 + y**2)/ (2 * R**2))
    return x, y, filter if len(shape) < 3 else cv2.merge([filter, filter])

def get_gaussianFilter1(shape, sigmaX, sigmaY):           # OpenCV 함수로 가우시안 커널 생성
    r = cv2.getGaussianKernel(shape[0], sigmaY)
    c = cv2.getGaussianKernel(shape[1], sigmaX)
    filter = np.outer(r,c)
    cv2.normalize(filter, filter, 0, 1, cv2.NORM_MINMAX)
    return filter if len(shape) < 3 else cv2.merge([filter, filter])

def get_butterworthFilter(shape, R, n):
    u = np.array(shape)//2
    y = np.arange(-u[0], u[0], 1)
    x = np.arange(-u[1], u[1], 1)
    x, y = np.meshgrid(x, y)
    dist = np.sqrt(x** 2 + y** 2)
    filter = 1 / (1 + np.power(dist / R, 2 * n))
    return x, y, filter if len(shape) < 3 else cv2.merge([filter, filter])

def mouse_event(event, x, y, flags, param):
    global radius

    if event == cv2.EVENT_FLAG_LBUTTON:
        dft, spectrum = FFT(image, mode)  # FFT 수행 및 셔플링
        x1, y1, gauss_filter = get_gaussianFilter(dft.shape, 50)  # 필터 생성
        x2, y2, butter_filter = get_butterworthFilter(dft.shape, raiseS, raiseN)

        filtered_dft1 = dft * gauss_filter  # 주파수 공간 필터링 - 원소 간 곱셈
        filtered_dft2 = dft * butter_filter
        gauss_img = IFFT(filtered_dft1, image.shape, mode)
        butter_img = IFFT(filtered_dft2, image.shape, mode)

        spectrum1 = calc_spectrum(filtered_dft1)
        spectrum2 = calc_spectrum(filtered_dft2)

        if mode == 3:
            gauss_filter, butter_filter = gauss_filter[:, :, 0], butter_filter[:, :, 0]

        fig = plt.figure("HOOAI Python Example Codes",figsize=(10,10))
        ax1 = plt.subplot(332, projection='3d')
        ax1.plot_surface(x1, y1, gauss_filter, cmap='RdPu'), plt.title('gauss_filter')
        ax2 = plt.subplot(333, projection='3d')
        ax2.plot_surface(x2, y2, butter_filter, cmap='RdPu'), plt.title('butter_filter')

        titles = ['input image', 'gauss_lowpassed_image', 'butter_lowpassed_image',
                  'input spectrum', 'gauss_lowpassed_spectrum', 'butter_lowpassed_spectrum']
        images = [image, gauss_img, butter_img, spectrum, spectrum1, spectrum2]

        cv2.imshow("change", butter_img)

        # cv2.createTrackbar("굴곡", "change", 0, 10, raiseN)

        plt.gray()  # 명암도 영상으로 표시
        for i, t in enumerate(titles):
            plt.subplot(3, 3, i + 4), plt.imshow(images[i]), plt.title(t)
        plt.tight_layout(), plt.show()



raiseS = 10
raiseN = 3
mode = 3

def raiseSc(pos):
    global raiseS
    print("pos")
    raiseS = pos
def raiseNc(pos):
    global raiseN
    print("pos")
    raiseN = pos

def mouse_callback3(event, x, y, flags, param):

    global  result , dft , imagefull, spectrum_img,mask

    if event == cv2.EVENT_MOUSEMOVE :
        change = [-3,-2,-1, 0 , 1 , 2, 3]

        if flags & cv2.EVENT_FLAG_MBUTTON:
            if draw == 0:
                for i in change:
                    for j in change:
                        spectrum_img[y + i , x + j] = 255
                        mask[y + i , x + j] = 255



    onRemoveMoire(1)

            # if draw == 1:
            #     for i in change:
            #         for j in change:
            #             image [y + i , x + j] = imagefull [y + i , x + j]
def asdf():
    print("실행")
def onRemoveMoire(val):
    global mask , a
    radius = cv2.getTrackbarPos("radius", title)
    th = cv2.getTrackbarPos("threshold", title)


    mask = cv2.threshold(spectrum_img, th, 255, cv2.THRESH_BINARY_INV)[1]


    y, x = np.divmod(mask.shape, 2)[0]

    print(x,y)

    cv2.circle(mask, (x, y), radius, 255, -1)

    #cv2.imshow("resulqwet", spectrum_img)
    #cv2.imshow("mask", mask)

    if dfta.ndim < 3:
        remv_dfta = np.zeros(dfta.shape, np.complex)
        remv_dfta.imag = cv2.copyTo(dfta.imag, mask=mask)
        remv_dfta.real = cv2.copyTo(dfta.real, mask=mask)
    else:
        remv_dfta = cv2.copyTo(dfta, mask=mask)

    result[:, finalimage.shape[1]:] = IFFT(remv_dfta, finalimage.shape, mode)
    cv2.imshow(title, calc_spectrum(remv_dfta))
    cv2.imshow("result", result)

# title = "removed moire"
#
# finalimage = cv2.imread('images/sea2.jpg', cv2.IMREAD_GRAYSCALE)
# if finalimage is None: raise Exception("영상 파일 읽기 에러")
#
# draw = 0
#
# a = 1
# mode = 3
# result = cv2.repeat(finalimage, 1, 2)  # 원본 영상 + 결과 영상
# dfta, spectrum_img = FFT(finalimage, mode)  # OpenCV dft() 함수 수행
#
#
# print(result)
#
# title = "removed moire"
# cv2.imshow("result", result)
# cv2.imshow(title, spectrum_img)
# cv2.createTrackbar("radius", title, 10, 255, onRemoveMoire)
# cv2.createTrackbar("threshold", title, 120, 255, onRemoveMoire)
# cv2.setMouseCallback("removed moire", mouse_callback3)
#
# cv2.waitKey(0)

# image = cv2.imread('images/iu2.jpg', cv2.IMREAD_GRAYSCALE)
# if image is None: raise Exception("영상 파일 읽기 에러")
#
# dft, spectrum = FFT(image, mode)                    # FFT 수행 및 셔플링
# x1, y1, gauss_filter = get_gaussianFilter(dft.shape, 50)       # 필터 생성
# x2, y2, butter_filter = get_butterworthFilter(dft.shape, raiseS, raiseN)
#
# x2, y2, butter_filter = get_butterworthFilter(dft.shape, raiseS, raiseN)
#
# filtered_dft1 = dft * gauss_filter                  # 주파수 공간 필터링 - 원소 간 곱셈
# filtered_dft2 = dft * butter_filter
# gauss_img = IFFT(filtered_dft1, image.shape, mode)
# butter_img= IFFT(filtered_dft2, image.shape, mode)
#
# spectrum1 = calc_spectrum(filtered_dft1)
# spectrum2 = calc_spectrum(filtered_dft2)
#
# if mode==3:
#     gauss_filter, butter_filter = gauss_filter[:,:,0], butter_filter[:,:,0]
#
# fig = plt.figure("HOOAI ????????",figsize=(10,10))                   # 그래프3 생성
# ax1 = plt.subplot(332, projection='3d')
# ax1.plot_surface(x1, y1, gauss_filter, cmap='RdPu'), plt.title('gauss_filter')
# ax2 = plt.subplot(333, projection='3d')
# ax2.plot_surface(x2, y2, butter_filter,cmap='RdPu'), plt.title('butter_filter')
#
# titles = ['input image','gauss_lowpassed_image', 'butter_lowpassed_image',
#           'input spectrum','gauss_lowpassed_spectrum','butter_lowpassed_spectrum']
# images = [image, gauss_img, butter_img, spectrum, spectrum1, spectrum2]
#
# cv2.imshow("change", butter_img)
#
# cv2.createTrackbar("raiseS", "change", 0, 200, raiseSc)
#
# cv2.createTrackbar("raiseN", "change", 0, 10, raiseNc)
#
# #cv2.createTrackbar("굴곡", "change", 0, 10, raiseN)
#
# cv2.setMouseCallback("change", mouse_event)
#
# plt.gray()  # 명암도 영상으로 표시
# for i, t in enumerate(titles):
#     plt.subplot(3, 3, i + 4), plt.imshow(images[i]), plt.title(t)
#
# plt.tight_layout()
# plt.show()





