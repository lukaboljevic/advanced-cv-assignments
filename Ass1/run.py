import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

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
    num_iters = kwargs.get("num_iters", 500)
    lmbd = kwargs.get("lmbd", 5)
    u_hs, v_hs = horn_schunck(img1, img2, num_iters, lmbd)
    title_hs = f"Horn-Schunck OF, # iters = {num_iters}, $\lambda$ = {lmbd}"


    # Draw the results
    show_flow(u_lk, v_lk, ax_21, type="field", set_aspect=True)
    ax_21.set_title(title_lk)
    show_flow(u_hs, v_hs, ax_22, type="field", set_aspect=True)
    ax_22.set_title(title_hs)

    fig.suptitle(f"Optical flow estimation, normalize pixel values = {normalize_values}")
    fig.set_size_inches(10, 7)
    plt.subplots_adjust(bottom=0.07, top=0.9)
    plt.show()


if __name__ == "__main__":
    # img1 = np.random.rand(200, 200).astype(np.float32)
    # img2 = img1.copy()
    # img2 = rotate_image(img2, 1)
    # normalize = False

    # img1 = cv.imread("./lab2/024.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./lab2/025.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = False

    # img1 = cv.imread("./collision/00000120.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./collision/00000121.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True
    
    img1 = cv.imread("./disparity/cporta_left.png", cv.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv.imread("./disparity/cporta_right.png", cv.IMREAD_GRAYSCALE).astype(np.float32)
    normalize = True


    draw(img1, 
         img2, 
         normalize_values=normalize,
         N=3,
         num_iters=700,
         lmbd=10)
