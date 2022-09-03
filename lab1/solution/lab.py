import cv2
import matplotlib.pyplot as plt

img_src = 'C:/Users/Anton/source/4k-1c/labs/programming_multymedia_and_multydemensional_apps/lab1/solution/Test.jpg'
gray_img_src = 'C:/Users/Anton/source/4k-1c/labs/programming_multymedia_and_multydemensional_apps/lab1/solution/test_gray.jpg'
binclr_ing_scr = 'C:/Users/Anton/source/4k-1c/labs/programming_multymedia_and_multydemensional_apps/lab1/solution/test_binclr.jpg'
over_lited_src = 'C:/Users/Anton/source/4k-1c/labs/programming_multymedia_and_multydemensional_apps/lab1/solution/over_lite.jpg'

def show_img(title, img):
    cv2.imshow(title, img)

def write_file(path_to, img):
    cv2.imwrite(path_to, img)

def wait_key_destroy_wins():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_hist(hist):
    plt.plot(hist, color='k')
    plt.show()

image = cv2.imread(img_src)

# shades of gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

show_img('Original image', image)
show_img('Gray image', gray)
write_file(gray_img_src, gray)

# black and white
th, bin_colors = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

show_img('binclr image', bin_colors)
write_file(binclr_ing_scr, bin_colors)
wait_key_destroy_wins()

# overlited with histogram
over_lited = cv2.imread(over_lited_src)
over_lited = cv2.cvtColor(over_lited, cv2.COLOR_BGR2GRAY)

show_img('over lite', over_lited)
show_hist(cv2.calcHist([over_lited], [0], None, [256], [0, 256]))
wait_key_destroy_wins()

over_lited_equalized = cv2.equalizeHist(over_lited)

show_img('over_lite image equalized', over_lited_equalized)
show_hist(cv2.calcHist([over_lited_equalized], [0], None, [256], [0, 256]))
wait_key_destroy_wins()



