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
    fig.set_size_inches(9, 8)
    plt.subplots_adjust(bottom=0.07, top=0.9)
    plt.show()


def draw_comparison_lk(img1, img2, Ns, normalize_values=True):
    # Just so we don't draw too many things, 3 should be enough
    num = len(Ns)
    assert num <= 3

    # Set entire figure title size, so it stands out
    plt.rc("figure", titlesize=16)

    # Optionally, normalize pixel values
    if normalize_values:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    # Initialize
    fig, axes = plt.subplots(1, num)
    alpha = 0.7  # opacity of image shown beneath the optical flow field
    extent = (0, img1.shape[1], -img1.shape[0], 0)

    # Perform Lucas-Kanade `num` times, and draw
    for i in range(num):
        N = Ns[i]
        u, v = lucas_kanade(img1, img2, N, verbose=False)
        title = f"N = {N}"

        axes[i].imshow(img1, alpha=alpha, extent=extent)
        show_flow(u, v, axes[i], type="field", set_aspect=True)
        axes[i].set_title(title)
    
    fig.suptitle(f"Lucas-Kanade, results for different N\nnormalize pixel values = {normalize_values}")
    fig.set_size_inches(14, 4)


def draw_comparison_hs(img1, img2, iter_limits, lambdas, normalize_values=True):
    # Just so we don't draw too many things, 3 should be enough
    assert len(iter_limits) <= 3
    assert len(lambdas) <= 3

    # Set entire figure title size, so it stands out
    plt.rc("figure", titlesize=16)

    # Optionally, normalize pixel values
    if normalize_values:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    alpha = 0.7  # opacity of image shown beneath the optical flow field
    extent = (0, img1.shape[1], -img1.shape[0], 0)


    ###############################################################
    ### First compare effects of different number of iterations ###
    ###############################################################

    fig1, axes1 = plt.subplots(1, len(iter_limits))

    # Perform Horn-Schunck `len(iter_limits)` times, and draw
    for i in range(len(iter_limits)):
        max_iters = iter_limits[i]
        u, v = horn_schunck(img1, img2, max_iters, 1, verbose=False)
        title = f"#iters = {max_iters}, $\lambda$ = 1"

        axes1[i].imshow(img1, alpha=alpha, extent=extent)
        show_flow(u, v, axes1[i], type="field", set_aspect=True)
        axes1[i].set_title(title)
    
    fig1.suptitle(f"Horn-Schunck, results for different # iterations\nnormalize pixel values = {normalize_values}")
    fig1.set_size_inches(14, 4)


    ##################################################
    ### Then compare effects for different lambdas ###
    ##################################################

    fig2, axes2 = plt.subplots(1, len(lambdas))

    # Perform Horn-Schunck `len(lambdas)` times, and draw
    for i in range(len(lambdas)):
        lmbd = lambdas[i]
        u, v = horn_schunck(img1, img2, 500, lmbd, verbose=False)
        title = f"#iters = 500, $\lambda$ = {lmbd}"

        axes2[i].imshow(img1, alpha=alpha, extent=extent)
        show_flow(u, v, axes2[i], type="field", set_aspect=True)
        axes2[i].set_title(title)
    
    fig2.suptitle(f"Horn-Schunck, results for different $\lambda$\nnormalize pixel values = {normalize_values}")
    fig2.set_size_inches(14, 4)



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


    # draw(img1, 
    #      img2, 
    #      normalize_values=normalize,
    #      overlay=overlay,
    #      N=N,
    #      max_iters=max_iters,
    #      lmbd=lmbd,
    #      init_with_lk=False,  # if True, max_iters should be smaller!
    #      eps=eps)
    
    draw_comparison_lk(img1, img2, [10, 30, 60])
    draw_comparison_hs(img1, img2, [100, 300, 500], [0.1, 1, 5])
    plt.show()
    
