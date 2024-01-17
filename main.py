import numpy as np, math
import cv2
from qwetest import FFT, IFFT, calc_spectrum, get_gaussianFilter, get_butterworthFilter ,onRemoveMoire, dft

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

controller = cv2.imread("images/controller.jpg", cv2.IMREAD_COLOR)


import matplotlib.pyplot as plt


imageG = cv2.imread("images/sea2.jpg", cv2.IMREAD_GRAYSCALE)
insertimageG = cv2.imread("images/iu2.jpg", cv2.IMREAD_GRAYSCALE)
if imageG is None: raise Exception("영상 파일 읽기 오류 발생")

image = cv2.imread("images/sea2.jpg", cv2.IMREAD_COLOR)
insertimage = cv2.imread("images/iu2.jpg", cv2.IMREAD_COLOR)

if image is None: raise Exception("영상 파일 읽기 오류 발생") # 예외 처리
if insertimage is None: raise Exception("영상 파일 읽기 오류 발생") # 예외 처리


imageG1 = cv2.mean(imageG)
imageG2 = cv2.mean(insertimageG)

onestart = 0

cv2.putText(insertimage, " ", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

sell = (imageG1[0] - imageG2[0])/2

H, W, channel = insertimage.shape
h, w, channel = insertimage.shape
delcolor = [250,250,250]

delval = 0
draw = 0
imagefull2 = image.copy()
imagefull = insertimage.copy()
imagebackup = insertimage.copy()

imagebackup = 0

#힌색 255 255 255

#검은색 0  0  0

def scaling_nearest(img, size):                                # 크기 변경 함수
    dst = np.zeros(size[::-1], img.dtype)               # 행렬과 크기는 원소가 역순
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])
    i = np.arange(0, size[1], 1)
    j = np.arange(0, size[0], 1)
    i, j = np.meshgrid(i, j)
    y, x = np.int32(i / ratioY), np.int32(j / ratioX)
    dst[i,j] = img[y,x]

    return dst

def bilinear_value(img, pt):
    x, y = np.int32(pt)
    if x >= img.shape[1]-1: x = x -1
    if y >= img.shape[0]-1: y = y - 1

    P1, P2, P3, P4 = np.float32(img[y:y+2,x:x+2].flatten())
   ## 4개의 화소 가져옴 – 화소 직접 접근
    # P1 = float(img[y, x] )                         # 상단 왼쪽 화소
    # P2 = float(img[y + 0, x + 1])                  # 상단 오른쪽 화소
    # P3 = float(img[y + 1, x + 0])                  # 하단 왼쪽 화소
    # P4 = float(img[y + 1, x + 1])                  # 하단 오른쪽 화소

    alpha, beta = pt[1] - y,  pt[0] - x                   # 거리 비율
    M1 = P1 + alpha * (P3 - P1)                      # 1차 보간
    M2 = P2 + alpha * (P4 - P2)
    P  = M1 + beta  * (M2 - M1)                     # 2차 보간
    return  np.clip(P, 0, 255)                       # 화소값 saturation후 반환

def scaling_bilinear(img, size):                   	# 양선형 보간
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])  # 변경 크기 비율

    dst = [[ bilinear_value(img, (j/ratioX, i/ratioY))  # for문 이용한 리스트 생성
             for j in range(size[0])]
           for i in range(size[1])]
    return np.array(dst, img.dtype)

def erode(img, mask=None):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None: mask = np.ones((3, 3), np.uint8)
    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0]

    mcnt = cv2.countNonZero(mask)
    for i in range(ycenter, img.shape[0] - ycenter):           # 입력 행렬 반복 순회
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1              # 마스크 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1              # 마스크 너비 범위
            roi = img[y1:y2, x1:x2]                            # 마스크 영역
            temp = cv2.bitwise_and(roi, mask)
            cnt  =  cv2.countNonZero(temp)                     # 일치한 화소수 계산
            dst[i, j] = 255 if (cnt == mcnt) else 0            # 출력 화소에 저장
    return dst

def dilate(img, mask):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None: mask = np.ones((3, 3), np.uint8)
    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0]


    for i in range(ycenter, img.shape[0] - ycenter):           # 입력 행렬 반복 순회
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1              # 마스크 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1              # 마스크 너비 범위
            roi = img[y1:y2, x1:x2]                            # 마스크 영역
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp)                     # 일치한 화소수 계산
            dst[i, j] = 0 if (cnt == 0) else 255            # 출력 화소에 저장
    return dst

def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)                 # 회선 결과 저장 행렬
    xcenter, ycenter = mask.shape[1]//2, mask.shape[0]//2  # 마스크 중심 좌표

    for i in range(ycenter, rows - ycenter):                  # 입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1               # 관심영역 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1               # 관심영역 너비 범위
            roi = image[y1:y2, x1:x2].astype("float32")         # 관심영역 형변환

            tmp = cv2.multiply(roi, mask)                       # 회선 적용 - OpenCV 곱셈
            dst[i, j] = cv2.sumElems(tmp)[0]                    # 출력화소 저장
    return dst

def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)

    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)
    dst = cv2.magnitude(dst1, dst2)                # 회선 결과인 두 행렬의 크기 계산
    dst1, dst2 = np.abs(dst1), np.abs(dst2)  # 회선 결과 행렬 양수 변경

    dst = np.clip(dst, 0, 255).astype("uint8")
    dst1 = np.clip(dst1, 0, 255).astype("uint8")
    dst2 = np.clip(dst2, 0, 255).astype("uint8")
    return dst, dst1, dst2

def mouse_callback3(event, x, y, flags, param):

    global  result , dft , imagefull, spectrum_img,mask

    if event == cv2.EVENT_MOUSEMOVE :
        change = [-3,-2,-1, 0 , 1 , 2, 3]

        if flags & cv2.EVENT_FLAG_MBUTTON:
            if draw == 0:
                for i in change:
                    for j in change:
                        spectrum_img[y + i , x + j] = 0
                        mask[y + i , x + j] = 0

            if draw == 1:
                for i in change:
                    for j in change:
                        spectrum_img[y + i , x + j] = 255
                        mask[y + i , x + j] = 255



    onRemoveMoire(1)

def mouse_callback(event, x, y, flags, param):
    global insertimagem, delcolor,h ,w , delval , draw , imagebackup
    if event == cv2.EVENT_LBUTTONDOWN :

        imagebackup = insertimage.copy()

        delcolor = insertimage[y][x]

        print(delcolor)

        delcolormin = delcolor - delval
        delcolormix = delcolor + delval

        for i in range(3):
            if delcolormin[i] > 255-delval:
                delcolormin[i] = 0

        for i in range(3):
            if delcolormix[i] < 0+delval:
                delcolormix[i] = 255

        for i in range(h):
            for j in range(w):
                if all(insertimage[i, j] > delcolormin) and all(insertimage[i, j] < delcolormix):
                    insertimage[i, j] = [255,255,255]
        for title in titles:
            cv2.imshow(title, eval(title))


    if event == cv2.EVENT_MOUSEMOVE :
        change = [-3,-2,-1, 0 , 1 , 2, 3]

        if flags & cv2.EVENT_FLAG_MBUTTON:
            if draw == 0:
                for i in change:
                    for j in change:
                        insertimage [y + i , x + j] = [255,255,255]

            if draw == 1:
                for i in change:
                    for j in change:
                        insertimage [y + i , x + j] = imagefull [y + i , x + j]



        if flags & cv2.EVENT_FLAG_RBUTTON:
            print("s")
            r = 0
            g = 0
            b = 0
            change = [-1, 1, 0]
            for i in change:
                for j in change:
                    r += insertimage[y + i , x + j][0]
                    g += insertimage[y + i, x + j][1]
                    b += insertimage[y + i, x + j][2]
            r = int(r / 9)
            g = int(g / 9)
            b = int(b / 9)
            for i in change:
                for j in change:
                    insertimage [y + i , x + j] = [(insertimage[y + i, x + j][0] + r)/2,
                    (insertimage[y + i, x + j][1] + g)/2, (insertimage[y + i, x + j][2] + b)/2]

            delcolor = insertimage[y][x]

            # circlen = insertimage.copy()

            # cv2.circle(insertimage, (x, y), 4, (int(delcolor[0]),int(delcolor[1]),int(delcolor[2])), 2)

        for title in titles:
            cv2.imshow(title, eval(title))

def mouse_callback2(event, x, y, flags, param):

    if event == cv2.EVENT_MOUSEMOVE :
        change = [-3,-2,-1, 0 , 1 , 2, 3]


        if flags & cv2.EVENT_FLAG_MBUTTON:
            if draw == 0:
                for i in change:
                    for j in change:
                        image [y + i , x + j] = [255,255,255]

            if draw == 1:
                for i in change:
                    for j in change:
                        image [y + i , x + j] = imagefull2 [y + i , x + j]



        if flags & cv2.EVENT_FLAG_RBUTTON:
            print("s")
            r = 0
            g = 0
            b = 0
            change = [-1, 1, 0]

            for i in change:
                for j in change:
                    r += image[y + i , x + j][0]
                    g += image[y + i, x + j][1]
                    b += image[y + i, x + j][2]

            r = int(r / 9)
            g = int(g / 9)
            b = int(b / 9)

            for i in change:
                for j in change:
                    image [y + i , x + j] = [(image[y + i, x + j][0] + r)/2,
                    (image[y + i, x + j][1] + g)/2, (image[y + i, x + j][2] + b)/2]

            delcolor = image[y][x]

            # circlen = insertimage.copy()

            # cv2.circle(insertimage, (x, y), 4, (int(delcolor[0]),int(delcolor[1]),int(delcolor[2])), 2)

        for title in titles:
            cv2.imshow(title, eval(title))

def onChange(pos):
    global delval
    delval = pos

#test_up1 = cv2.pyrUp(insertimage, dstsize=(w * 2, h * 2), borderType=cv2.BORDER_DEFAULT)
#image = cv2.pyrDown(image)

## 각 행렬을 영상으로 표시
title = 'Trackbar Event'
titles = ['image', 'insertimage']


for title in titles:
    cv2.imshow(title, eval(title))
######################################################


blurring_mask1 = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
blurring_mask2 = np.array([[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                           [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                           [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]])

blurring_out1 = cv2.filter2D(image, -1, blurring_mask2)

#cv2.imshow("blurring_mask2", blurring_out1)


sharpening_mask1 = np.array([[-0, -1, -0], [-1, 5, -1], [-0,-1, -0]])

a = 2.0

sharpening_out1 = cv2.filter2D(image, -1, sharpening_mask1)

#cv2.imshow("sharpen_1", sharpening_out1)

data1 = [-1, 0, 1,
         -2, 0, 2,
         -1, 0, 1]
data2 = [-1, -2, -1,
         0, 0, 0,
         1, 2, 1]

dst, dst1, dst2 = differential(insertimageG, data1, data2)




def opening(img, mask):                     # 열림 연산
    tmp = erode(img, mask)                  # 침식
    dst = dilate(tmp, mask)                 # 팽창
    return dst

def closing(img, mask):                     # 닫힘 연산
    tmp = dilate(img, mask)
    dst = erode(tmp, mask)
    return dst


mask = np.array([[0, 1, 0],                 # 마스크 초기화
                 [1, 1, 1],
                 [0, 1, 0]]).astype("uint8")

th_img = cv2.threshold(dst2, 40, 255, cv2.THRESH_BINARY)[1]   # 영상 이진화

#cv2.imshow("2진화 ", th_img)

dst1 = opening(th_img, mask)                            # 사용자 정의 열림 함수 호출
dst2 = closing(th_img, mask)                            # 사용자 정의 닫힘 함수 호출
dst3 = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask)   # OpenCV의 열림 함수
dst4 = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, mask, 1)  # OpenCV의 닫힘 함수

kernel = np.ones((50, 50), np.uint8)

result = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel)

#cv2.imshow("2진화 ", result)

#cv2.imshow("User closing", dst2)
#cv2.imshow("OpenCV closing", dst4)

size = (200, 200)

dst1 = scaling_bilinear(insertimageG, size)                # 크기 변경 - 양선형 보간
dst2 = scaling_nearest(imageG, size)                 # 크기 변경 - 최근접 이웃 보간
dst3 = cv2.resize(insertimage, size, 0, 0, cv2.INTER_LINEAR)  # OpenCV 함수 적용
dst4 = cv2.resize(insertimage, size, 0, 0, cv2.INTER_NEAREST)

#cv2.imshow("User_bilinear", dst1)
#cv2.imshow("User_Nearest", dst2)
#cv2.imshow("OpenCV_bilinear", dst3)
#cv2.imshow("OpenCV_Nearest", dst4)
#####################################################

cv2.imshow("controller", controller)

cv2.createTrackbar("colorval", "insertimage", 0, 25, onChange)
cv2.setMouseCallback('insertimage', mouse_callback)
cv2.setMouseCallback('image', mouse_callback2)

def imagechangeB(asd):
    asd = cv2.cvtColor(asd, cv2.COLOR_BGR2GRAY)
    dft, spectrum_img = FFT(asd, mode)  # OpenCV dft() 함수 수행

    #cv2.imshow("qwe", spectrum_img)
    return spectrum_img
def imagechangeB2(asd):
    dft, spectrum_img = FFT(asd, mode)  # OpenCV dft() 함수 수행

    cv2.imshow("qwe", spectrum_img)
    return spectrum_img
def mouse_event(event, x, y, flags, param):
    global radius,image

    if event == cv2.EVENT_FLAG_LBUTTON:
        imagebutterworthFilter(image)

raiseS = 10
raiseN = 3
mode = 3
Si = 1


def raiseSc(pos):
    global raiseS
    print("pos")
    raiseS = pos
def raiseNc(pos):
    global raiseN
    print("pos")
    raiseN = pos

def imagebutterworthFilter(Mimage):
    global onestart , selfimagemask , selfimage, finalimage

    image = cv2.cvtColor(Mimage, cv2.COLOR_BGR2GRAY)

    finalimage = image

    dft, spectrum = FFT(image, mode)                    # FFT 수행 및 셔플링
    x1, y1, gauss_filter = get_gaussianFilter(dft.shape, 50)       # 필터 생성

    x2, y2, butter_filter = get_butterworthFilter(dft.shape, raiseS, raiseN)

    filtered_dft1 = dft * gauss_filter                  # 주파수 공간 필터링 - 원소 간 곱셈
    filtered_dft2 = dft * butter_filter


    gauss_img = IFFT(filtered_dft1, image.shape, mode)
    butter_img= IFFT(filtered_dft2, image.shape, mode)

    spectrum1 = calc_spectrum(filtered_dft1)
    spectrum2 = calc_spectrum(filtered_dft2)

    if mode==3:
        gauss_filter, butter_filter = gauss_filter[:,:,0], butter_filter[:,:,0]

    fig = plt.figure("HOOAI asgb",figsize=(10,10))                   # 그래프3 생성

    ax2 = plt.subplot(321, projection='3d')
    ax2.plot_surface(x2, y2, butter_filter,cmap='RdPu'), plt.title('butter_filter')

    titles = ['image', 'input image',
              'spectrum','input image spectrum']

    images = [image, butter_img, spectrum, spectrum2]

    cv2.imshow("change2", spectrum2)

    cv2.imshow("change", butter_img)

    #cv2.createTrackbar("굴곡", "change", 0, 10, raiseN)

    cv2.setMouseCallback("change", mouse_event)

    if (onestart == 0):

        cv2.createTrackbar("raiseS", "change", 0, 200, raiseSc)

        cv2.createTrackbar("raiseN", "change", 0, 10, raiseNc)

        onestart = 1

    plt.gray()  # 명암도 영상으로 표시
    for i, t in enumerate(titles):
        plt.subplot(3, 2, i + 3), plt.imshow(images[i]), plt.title(t)
    plt.tight_layout()
    plt.show()

    #finalimage = butter_img

def onRemoveMoire(val):
    global mask , spectrum_img, dfta,finalimage, result

    title = "removed moire"

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

def endstart():
    global spectrum, finalimage, spectrum_img, dfta, result
    draw = 0

    a = 1
    mode = 3
    result = cv2.repeat(finalimage, 1, 2)  # 원본 영상 + 결과 영상
    dfta, spectrum_img = FFT(finalimage, mode)  # OpenCV dft() 함수 수행

    title = "removed moire"
    cv2.imshow("result", result)
    cv2.imshow(title, spectrum_img)

    cv2.createTrackbar("radius", "removed moire", 10, 255, onRemoveMoire)
    cv2.createTrackbar("threshold", "removed moire", 120, 255, onRemoveMoire)
    cv2.setMouseCallback("removed moire", mouse_callback3)




while True:
    key = cv2.waitKey()
    if key == ord('f'):
        print("최종")
        endstart()
    if key == ord('s'):
        insertimage = cv2.filter2D(insertimage, -1, sharpening_mask1)
    if key == ord('b'):

        insertimage = cv2.filter2D(insertimage, -1, blurring_mask1)
    if key == ord('m'):
        selfimage = image
        selfimagemask = imagechangeB(image)
        imagebutterworthFilter(image)

    if key == ord('x'):
        print("합치기")

        masks = cv2.threshold(insertimage, 254, 255, cv2.THRESH_BINARY)[1]


        masks = cv2.split(masks)



        aimage = masks[0]
        dimage = cv2.bitwise_not(aimage)


        (H, W), (h, w) = image.shape[:2], insertimage.shape[:2]
        x, y = (W - w) // 2, (H - h)
        roi = image[y:y + h, x:x + w]

        foreground = cv2.bitwise_and(insertimage, insertimage, mask=dimage)
        background = cv2.bitwise_and(roi, roi, mask=aimage)



        dst = cv2.add(background, foreground)
        image[y:y + h, x:x + w] = dst

        #cv2.imshow('image', image)

        cv2.waitKey()
    elif key == ord('['):
        Si = 0.95
        insertimage = cv2.resize(insertimage, (0,0), fx= Si, fy=Si, interpolation = cv2.INTER_NEAREST)
        insertimageG = cv2.resize(insertimageG, (0, 0), fx=Si, fy=Si, interpolation=cv2.INTER_NEAREST)
        for title in titles:
            cv2.imshow(title, eval(title))
    elif key == ord(']'):
        Si = 1.05
        insertimage = cv2.resize(insertimage, (0,0), fx= Si, fy=Si, interpolation = cv2.INTER_NEAREST)
        insertimageG = cv2.resize(insertimageG, (0, 0), fx=Si, fy=Si, interpolation=cv2.INTER_NEAREST)
        for title in titles:
            cv2.imshow(title, eval(title))
    elif key == ord('n'):
        print("합치기")

        masks = cv2.threshold(insertimage, 254, 255, cv2.THRESH_BINARY)[1]

        masks = cv2.split(masks)

        aimage = masks[0]
        dimage = cv2.bitwise_not(aimage)

        #cv2.imshow("dimage", dimage)

        dst, dst1, dst2 = differential(insertimageG, data1, data2)

        #cv2.imshow("마스크 ", data2)

        cv2.imshow("Roberts mask", dst2)


        mask = np.array([[0, 1, 0],  # 마스크 초기화
                         [1, 1, 1],
                         [0, 1, 0]]).astype("uint8")

        th_img = cv2.threshold(dst2, 40, 255, cv2.THRESH_BINARY)[1]  # 영상 이진화

        cv2.imshow("2 ", th_img)

        dst1 = opening(th_img, mask)  # 사용자 정의 열림 함수 호출
        dst2 = closing(th_img, mask)  # 사용자 정의 닫힘 함수 호출
        dst3 = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask)  # OpenCV의 열림 함수
        dst4 = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, mask, 1)  # OpenCV의 닫힘 함수

        kernel = np.ones((50, 50), np.uint8)

        result = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel)


        dimage = result  # 이거임


        cv2.imshow("newdimage", dimage)

        aimage = cv2.bitwise_not(dimage)

        (H, W), (h, w) = image.shape[:2], insertimage.shape[:2]
        x, y = (W - w) // 2, (H - h)
        roi = image[y:y + h, x:x + w]

        foreground = cv2.bitwise_and(insertimage, insertimage, mask=dimage)
        background = cv2.bitwise_and(roi, roi, mask=aimage)


        dst = cv2.add(background, foreground)
        image[y:y + h, x:x + w] = dst

        # cv2.imshow('image', image)

        cv2.waitKey()


    elif key == ord('a'):
        if draw == 1:
            print("채우기")
            draw = 0
        elif draw == 0:
            print("채우기")
            draw = 1

        print("없애기")
    elif key == ord('z'):
        print("뒤로가기")
        insertimage = imagebackup.copy()

    elif key == ord('c'):
        if sell < 0:
            sell = sell * -1
            array = np.full(insertimage.shape, (sell, sell, sell), dtype=np.uint8)
            cgimage = cv2.subtract(insertimage, array)

        elif sell > 0:
            print("?")
            array = np.full(insertimage.shape, (sell, sell, sell), dtype=np.uint8)
            cgimage = cv2.add(insertimage, array)
        print(cgimage[10][10])

        insertimage = cgimage.copy()

        print("화면보정")

    elif key == 27:
        break


cv2.waitKey(0)