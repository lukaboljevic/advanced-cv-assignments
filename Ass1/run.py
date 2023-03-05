import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from time import perf_counter

from lucas import lucas_kanade
from horn import horn_schunck
from utils import rotate_image, show_flow


def draw(img1, img2, normalize_values=False, **kwargs):
    """
    Draw the results of Lucas-Kanade and Horn-Schunck optical flow estimation.
    """

    # Set entire figure title size, so it stands out
    plt.rc("figure", titlesize=18)

    # Optionally, normalize pixel values
    if normalize_values:
        img1 = img1 / 255.0
        img2 = img2 / 255.0


    # Draw the two images first
    fig, ((ax_11, ax_12), (ax_21, ax_22)) = plt.subplots(2, 2)
    ax_11.imshow(img1)
    ax_11.set_title("Frame t")
    ax_12.imshow(img2)
    ax_12.set_title("Frame t+1")


    # Perform Lucas-Kanade
    N = kwargs.get("N", 3)  # default value N = 3 if not provided
    u_lk, v_lk = lucas_kanade(img1, img2, N)
    title_lk = f"Lucas-Kanade OF, N = {N}"


    # Perform Horn-Schunck
    max_iters = kwargs.get("max_iters", 500)
    lmbd = kwargs.get("lmbd", 1)
    init_with_lk = kwargs.get("init_with_lk", False)
    add_to_title = " (init. with LK)" if init_with_lk else ""
    u_hs, v_hs = horn_schunck(img1, img2, max_iters, lmbd, N=N if init_with_lk else None)
    title_hs = f"Horn-Schunck OF{add_to_title}, # iters = {max_iters}, $\lambda$ = {lmbd}"


    # Draw the results
    show_flow(u_lk, v_lk, ax_21, type="field", set_aspect=True)
    ax_21.set_title(title_lk)
    show_flow(u_hs, v_hs, ax_22, type="field", set_aspect=True)
    ax_22.set_title(title_hs)

    fig.suptitle(f"Optical flow estimation, normalize pixel values = {normalize_values}")
    fig.set_size_inches(10, 7)
    plt.subplots_adjust(bottom=0.07, top=0.9)
    plt.show()


def test_time(img1, img2, normalize_values=False, **kwargs):
    """
    Test execution time for Lucas-Kanade and Horn-Schunck algorithms
    """
    if normalize_values:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    # Get parameters
    N = kwargs.get("N", 3)
    max_iters = kwargs.get("max_iters", 500)
    lmbd = kwargs.get("lmbd", 1)


    # Lucas-Kanade
    start_lk = perf_counter()
    lucas_kanade(img1, img2, N)
    end_lk = perf_counter() - start_lk


    # Horn-Schunck
    start_hs = perf_counter()
    horn_schunck(img1, img2, max_iters, lmbd)
    end_hs = perf_counter() - start_hs


    # Horn-Schunck initialized with output of Lucas-Kanade
    start_hs_with_lk = perf_counter()
    horn_schunck(img1, img2, max_iters, lmbd, N=N)
    end_hs_with_lk = perf_counter() - start_hs_with_lk


    print("========================================================")
    print(f"Parameters: N = {N}, max_iters = {max_iters}, lambda = {lmbd}")
    print(f"Lucas-Kanade: {round(end_lk, 3)} s")
    print(f"Horn-Schunck: {round(end_hs, 3)} s")
    print(f"Horn-Schunck + Lucas-Kanade: {round(end_hs_with_lk, 3)} s")
    print()



if __name__ == "__main__":
    # img1 = np.random.rand(200, 200).astype(np.float32)
    # img2 = img1.copy()
    # img2 = rotate_image(img2, 1)
    # normalize = False

    # img1 = cv.imread("./lab2/024.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./lab2/025.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True

    # img1 = cv.imread("./collision/00000120.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./collision/00000121.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True
    
    # img1 = cv.imread("./disparity/office_left.png", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./disparity/office_right.png", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True

    img1 = cv.imread("./waffles/waffles1.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv.imread("./waffles/waffles2.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    normalize = True


    draw(img1, 
         img2, 
         normalize_values=normalize,
         N=20,
         max_iters=1000,
         lmbd=1,
         init_with_lk=False)
    
    # test_time(img1,
    #           img2,
    #           normalize_values=normalize,
    #           N=50,
    #           max_iters=1000,
    #           lmbd=1)
