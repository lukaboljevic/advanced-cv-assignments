import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from time import perf_counter

from lucas import lucas_kanade
from horn import horn_schunck
from utils import rotate_image, show_flow


def draw(img1, img2, normalize_values=True, overlay=True, **kwargs):
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
    N = kwargs.get("N")
    u_lk, v_lk = lucas_kanade(img1, img2, N)
    title_lk = f"Lucas-Kanade OF, N = {N}"


    # Perform Horn-Schunck
    max_iters = kwargs.get("max_iters")
    lmbd = kwargs.get("lmbd")
    init_with_lk = kwargs.get("init_with_lk")
    eps = kwargs.get("eps")
    add_to_title = f" (+ LK, N = {N})" if init_with_lk else ""
    u_hs, v_hs = horn_schunck(img1,
                              img2,
                              max_iters,
                              lmbd,
                              N=N if init_with_lk else None,
                              eps=eps)
    title_hs = f"Horn-Schunck OF{add_to_title}, #iters = {max_iters}, $\lambda$ = {lmbd}"


    # Draw the results
    alpha = 0.7  # opacity of image shown beneath the optical flow field
    extent = (0, img1.shape[1], -img1.shape[0], 0)

    if overlay:
        ax_21.imshow(img1, alpha=alpha, extent=extent)
    show_flow(u_lk, v_lk, ax_21, type="field", set_aspect=True)
    ax_21.set_title(title_lk)

    if overlay:
        ax_22.imshow(img1, alpha=alpha, extent=extent)
    show_flow(u_hs, v_hs, ax_22, type="field", set_aspect=True)
    ax_22.set_title(title_hs)

    fig.suptitle(f"Optical flow estimation, normalize pixel values = {normalize_values}")
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(bottom=0.07, top=0.9)
    plt.show()


if __name__ == "__main__":
    # img1 = np.random.rand(200, 200).astype(np.float32)
    # img2 = img1.copy()
    # img2 = rotate_image(img2, 1)
    # N, max_iters, lmbd = 20, 500, 1
    # normalize = False
    # overlay = False
    # eps=1e-5

    # img1 = cv.imread("./lab2/024.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./lab2/025.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # N, max_iters, lmbd = 20, 1000, 1
    # normalize = True
    # overlay = True
    # eps=3e-5

    # img1 = cv.imread("./collision/00000120.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./collision/00000121.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # N, max_iters, lmbd = 50, 1000, 5  # or try lmbd=1
    # normalize = True
    # overlay = True
    # eps=5e-6

    img1 = cv.imread("./waffles/waffles1.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv.imread("./waffles/waffles2.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    N, max_iters, lmbd = 50, 1000, 5
    normalize = True
    overlay = True
    eps=1e-5

    # img1 = cv.imread("./waffles/waffles1_fast.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./waffles/waffles2_fast.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # N, max_iters, lmbd = 50, 1000, 5
    # normalize = True
    # overlay = True
    # eps=1e-5


    draw(img1, 
         img2, 
         normalize_values=normalize,
         overlay=overlay,
         N=N,
         max_iters=max_iters,
         lmbd=lmbd,
         init_with_lk=False,
         eps=eps)
    
