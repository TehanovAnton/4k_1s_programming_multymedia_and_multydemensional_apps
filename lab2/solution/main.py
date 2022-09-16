import cv2
import numpy as np

img_path = "mops_image.jpg"
figures_path = "image.png"
ball_white_background_path = "ball_on_white_background.jpg"


def read_img(img_src):
    return cv2.imread(img_src)


def show_and_wait_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def show_two_and_wait_img(title, img1, img2):
    cv2.imshow(title, img1)
    cv2.imshow(title, img2)
    cv2.waitKey(0)


def task1():
    img = read_img(img_path)
    kernel = np.array([
        [0, -2, 0],
        [-2, 10, -2],
        [0, -2, 0]
    ])
    result_img = cv2.filter2D(img, -1, kernel)

    show_and_wait_img('original', img)
    show_and_wait_img('sharped', result_img)


def task2():
    img = read_img(img_path)

    # blur
    result_img = cv2.blur(img, (3, 3))
    show_and_wait_img('original', img)
    show_and_wait_img('blured', result_img)

    # GaussianBlur
    result_img = cv2.GaussianBlur(img, (-3, -3), cv2.BORDER_DEFAULT)
    show_and_wait_img('original', img)
    show_and_wait_img('gaussian blured', result_img)

    # medianBlur
    result_img = cv2.medianBlur(img, 7)
    show_and_wait_img('original', img)
    show_and_wait_img('blured', result_img)


def task3():
    img = read_img(ball_white_background_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binnarized_img = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)

    # dilate
    result_img = cv2.dilate(binnarized_img, (5, 5), iterations=5)
    show_and_wait_img('original', binnarized_img)
    show_and_wait_img('dilate', result_img)

    # erode
    result_img = cv2.erode(binnarized_img, (5, 5), iterations=5)

    show_and_wait_img('original', binnarized_img)
    show_and_wait_img('erode', result_img)


def task4():
    img = read_img(ball_white_background_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binnarized_img = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)

    # dilate difference
    dilate_img = cv2.dilate(binnarized_img, (5, 5), iterations=5)
    result_img = dilate_img - binnarized_img

    show_and_wait_img('original', binnarized_img)
    show_and_wait_img('difference with dilate', result_img)

    # erode difference
    erode_img = cv2.erode(binnarized_img, (5, 5), iterations=5)
    result_img = binnarized_img - erode_img

    show_and_wait_img('original', binnarized_img)
    show_and_wait_img('difference with erode', result_img)


def task5():
    img = read_img(ball_white_background_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binnarized_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    # dilate difference
    dilate_img = cv2.dilate(binnarized_img, (5, 5), iterations=5)
    result_img = dilate_img - binnarized_img

    show_and_wait_img('original', binnarized_img)
    show_and_wait_img('difference with dilate', result_img)

    # erode difference
    erode_img = cv2.erode(binnarized_img, (5, 5), iterations=5)
    result_img = binnarized_img - erode_img

    show_and_wait_img('original', binnarized_img)
    show_and_wait_img('difference with erode', result_img)

# task1()
# task2()
# task3()
# task4()
# task5()