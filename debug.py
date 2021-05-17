import cv2 as cv

def display(title, image, timeout=1, trueSize=False):
    '''
    OpenCV machinery for showing an image until the user presses a key.
    :param title: Window title in string form
    :param image: ndArray containing image to show
    :return:
    '''
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    if trueSize:
        cv.resizeWindow(title, image.shape[1], image.shape[0])
    cv.imshow(title, image)
    cv.waitKey(timeout)
