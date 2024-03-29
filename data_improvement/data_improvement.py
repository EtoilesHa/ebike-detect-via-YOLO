import os
import cv2
import numpy as np
import random


# 缩放
def scale_pic(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale)


# 翻转
def horizontal_reverse(image):
    return cv2.flip(image, 1, dst=None)


def vertical_reverse(image):
    return cv2.flip(image, 0, dst=None)


# 旋转+缩放
def rotate_and_scale(img, angele=15, scale=1.1):
    w = img.shape[0]
    h = img.shape[1]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angele, scale)
    ret = cv2.warpAffine(img, matrix, (w,h))
    return ret


def brighter_or_darker(img, prec=1.2):
    img_copy = img.copy()
    w = img.shape[0]
    h = img.shape[1]
    for i in range(0, w):
        for j in range(0, h):
            for k in range(3):
                img_copy[i,j,k] =np.clip(int(img[i,j,k])*prec, a_max=255, a_min=0)
    return img_copy


# 平移
def move_img(img, x, y):
    img_size = img.shape
    h = img_size[0]
    w = img_size[1]
    mat_trans = np.float32([[1,0,x],[0,1,y]])
    ret = cv2.warpAffine(img, mat_trans, (w,h))
    return ret


# 噪声 - 椒盐噪声
def AddSaltPepperNoise(src, rate=0.05):
    srcCopy = src.copy()
    height, width = srcCopy.shape[0:2]
    noiseCount = int(rate*height*width/2)
    # add salt noise
    X = np.random.randint(width,size=(noiseCount,))
    Y = np.random.randint(height,size=(noiseCount,))
    srcCopy[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width,size=(noiseCount,))
    Y = np.random.randint(height,size=(noiseCount,))
    srcCopy[Y, X] = 0
    return srcCopy


# 噪声 - 高斯噪声
def AddGaussNoise(src,sigma=5):
    mean = 0
    # 获取图片的高度和宽度
    height, width, channels = src.shape[0:3]
    gauss = np.random.normal(mean,sigma,(height,width,channels))
    noisy_img = np.uint8(src + gauss)
    return noisy_img


# 高斯模糊
def blur(img):
    return cv2.GaussianBlur(img, (5,5), 1)


# --------------------------------------
def test_one_sample():
    test_pic = r"C:\Users\13917\Desktop\2023_deeplearning\dataset_food\Snipaste_2023-12-28_17-24-48.jpg"
    test_pic = cv2.imread(test_pic, 4)
    if test_pic is None:
        print("打开失败")
        exit()
    cv2.imshow("img_show", test_pic)
    cv2.waitKey(0)
    #
    # img_resize = scale_pic(test_pic, 0.6)
    # cv2.imshow("img_resize", img_resize)
    # cv2.waitKey(0)
    #
    # img_horizon = horizontal_reverse(test_pic)
    # cv2.imshow("img_horizon", img_horizon)
    # cv2.waitKey(0)
    #
    # img_vertical = vertical_reverse(test_pic)
    # cv2.imshow("img_vertical", img_vertical)
    # cv2.waitKey(0)

    img_rotate = rotate_and_scale(test_pic, 30, 1.2)
    cv2.imshow("img_rotate", img_rotate)
    cv2.waitKey(0)

    # img_brighter = brighter_or_darker(test_pic, 0.6)
    # cv2.imshow("img_brighter", img_brighter)
    # cv2.waitKey(0)
    #
    # img_move = move_img(test_pic, 50, 50)
    # cv2.imshow("img_move", img_move)
    # cv2.waitKey(0)

    # img_pepper = AddSaltPepperNoise(test_pic)
    # cv2.imshow("img_pepper", img_pepper)
    # cv2.waitKey(0)
    #
    # img_Gauss = AddGaussNoise(test_pic, 3)
    # cv2.imshow("img_Gauss", img_Gauss)
    # cv2.waitKey(0)

    img_blur = blur(test_pic)
    cv2.imshow("img_blur", img_blur)
    cv2.waitKey(0)
# --------------------------------------
def opt_on_dir():
    root_path = r"C:\Users\13917\Desktop\2023_deeplearning\data_improvement\ebike\images"
    save_path = r"C:\Users\13917\Desktop\2023_deeplearning\data_improvement\ebike\results"
    for a,b,c in os.walk(root_path):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            print(file_i_path)
            img_i = cv2.imread(file_i_path)

            img_blur = blur(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4]+"blur.jpg"), img_blur)

            img_AddGaussNoise = AddGaussNoise(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "GaussNoise.jpg"), img_AddGaussNoise)

            img_Pepper = AddSaltPepperNoise(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "img_Pepper.jpg"), img_Pepper)

            img_light = brighter_or_darker(img_i,random.uniform(0.6, 1.5))
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "img_light.jpg"), img_light)

            img_basic = rotate_and_scale(img_i, random.randint(0, 15), random.uniform(0.8, 1.2))
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "img_basic.jpg"), img_basic)

            img_move = move_img(img_i, random.randint(0, int(img_i.shape[0]*0.5)), random.randint(0, int(img_i.shape[1]*0.5)))
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "img_move.jpg"), img_move)

            if random.randint(0, 1):
                img_reverse = horizontal_reverse(img_i)
            else:
                img_reverse = vertical_reverse(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "img_reverse.jpg"), img_reverse)


if __name__ == '__main__':
    # test_one_sample()
    opt_on_dir()
