import numpy as np
import cv2 as cv
from time import perf_counter

from lucas import lucas_kanade
from horn import horn_schunck
from utils import rotate_image


def test_time_lk(img1, img2, N, normalize_values=True):
    """
    Test execution time for Lucas-Kanade algorithm
    """
    if normalize_values:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    print("=" * 70)
    start_lk = perf_counter()
    lucas_kanade(img1, img2, N, verbose=False)
    end_lk = perf_counter() - start_lk

    print(f"Lucas-Kanade, N = {N}: {round(end_lk, 3)} s")
    print()


def test_time_hs(img1, img2, N, max_iters, lmbd, eps, normalize_values=True):
    """
    Test execution time for Horn-Schunck algorithm
    """
    if normalize_values:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    verbose = False

    # Horn-Schunck
    print("=" * 70)
    print(">>>>>> HORN-SCHUNCK <<<<<<")
    start_hs = perf_counter()
    horn_schunck(img1, img2, max_iters, lmbd, eps=eps, verbose=verbose)
    end_hs = perf_counter() - start_hs


    # Horn-Schunck initialized with output of Lucas-Kanade, without allowing "convergence"
    print("=" * 70)
    print(">>>>>> HORN-SCHUNCK + LUCAS-KANADE <<<<<<")
    start_hs_with_lk = perf_counter()
    horn_schunck(img1, img2, max_iters, lmbd, N=N, verbose=verbose)
    end_hs_with_lk = perf_counter() - start_hs_with_lk


    # Same as above, just allowing for convergence
    print("=" * 70)
    print(">>>>>> HORN-SCHUNCK + LUCAS-KANADE, ALLOW CONVERGENCE <<<<<<")
    start_hs_with_lk_conv = perf_counter()
    horn_schunck(img1, img2, max_iters, lmbd, N=N, eps=eps, verbose=verbose)
    end_hs_with_lk_conv = perf_counter() - start_hs_with_lk_conv


    print("=" * 70)
    print(f"Parameters: max_iters = {max_iters}, lambda = {lmbd}, N (for LK) = {N}")
    print(f"Horn-Schunck: \t\t\t\t\t{round(end_hs, 3)} s")
    print(f"Horn-Schunck + Lucas-Kanade: \t\t\t{round(end_hs_with_lk, 3)} s")
    print(f"Horn-Schunck + Lucas-Kanade, allow converging: \t{round(end_hs_with_lk_conv, 3)} s")
    print()


if __name__ == "__main__":
    # img1 = np.random.rand(200, 200).astype(np.float32)
    # img2 = img1.copy()
    # img2 = rotate_image(img2, 1)
    # normalize = False
    # eps=1e-5

    img1 = cv.imread("./lab2/024.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv.imread("./lab2/025.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    normalize = True
    eps=3e-5

    # img1 = cv.imread("./collision/00000120.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./collision/00000121.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True
    # eps=5e-6

    # img1 = cv.imread("./waffles/waffles1.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./waffles/waffles2.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True
    # eps=1e-5

    # img1 = cv.imread("./waffles/waffles1_fast.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # img2 = cv.imread("./waffles/waffles2_fast.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)
    # normalize = True
    # eps=1e-5

    
    Ns = [3, 10, 20, 50]
    iter_limits = [200, 500, 1000]

    for N in Ns:
        test_time_lk(img1, img2, N, normalize_values=normalize)

    print()
    print("*" * 80)
    print("*" * 80)
    print()

    for max_iters in iter_limits:
        test_time_hs(img1,
                     img2,
                     normalize_values=normalize,
                     N=20,  # shouldn't vary as we're testing whether HS can be sped up by using LK
                     max_iters=max_iters,
                     lmbd=5,  # doesn't affect execution time much
                     eps=eps)
        print()
        print("*" * 80)
        print("*" * 80)
        print()