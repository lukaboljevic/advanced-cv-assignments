import cv2
import numpy as np
import utils as ut


"""
To get to the mode of the PDF, we iteratively calculate the formula from slides:
    x = [ sum_{i=1 to N} x_i * w_i * kernel_derivative ] / [ sum_{i=1 to N} w_i * kernel_derivative ]

    N is the number of (data) points "within" the kernel, x_i represents coordinates within the region
    defined by the kernel (check assignment 2 slides, slide 4), w_i is the value of the PDF at
    coordinate x_i, kernel_derivative is the derivative of the kernel we want to use. If we want to
    use Epanechnikov kernel, then kernel_derivative = 1 (for all inputs), so we just ignore it.

    The kernels we are working with are 1 dimensional, so:
    To get a change in x and y, we need to apply the equation twice - the only thing that changes
    is the kernel. In other words, we have a kernel for x direction, and a kernel for y direction.
    Again check slide 4 of assignment 2 slides, and use those.

    the || (x - x_i) / h ||^2 part is explained here: https://drive.google.com/file/d/18r7qsTiAqpRguAlDBI4JArFjEMwO8Uaf/view
    around 2:25:00
    
    For a bit more detailed explanation on the formula, go to 2:34:00 on the same video
"""


def mean_shift(image, start_pos, kernel_size, eps):
    """
    Perform mean shift on an image ("PDF"), given a starting position and kernel (neighborhood) size.

    Note that start position is taken as (column, row), not (row, column) - in other words,
    start_pos is given as a set of coordinates (x, y).
    """
    kernel_x, kernel_y = ut.coordinates_kernels(kernel_size)
    # TODO For now, kernel derivative is constant (i.e. kernel is Epanechnikov)

    curr_pos = start_pos
    print(f"Starting position: row {start_pos[1]}, col {start_pos[0]}\n")
    num_iters = 0

    while True:
        patch, _ = ut.get_patch(image, curr_pos, kernel_size)
        change_x = np.divide(np.sum(np.multiply(kernel_x, patch)), np.sum(patch))
        change_y = np.divide(np.sum(np.multiply(kernel_y, patch)), np.sum(patch))

        if abs(change_x) < eps and abs(change_y) < eps:
            row = int(curr_pos[1])
            column = int(curr_pos[0])
            print(f"Mode found, row {row}, column {column}")
            print(f"Value: {image[row, column]}")
            print(f"Converged in {num_iters} iterations")
            break

        curr_pos = (curr_pos[0] + change_x, curr_pos[1] + change_y)  # get_patch takes care of the rounding
        num_iters += 1


def main(draw=True):
    # image = ut.generate_responses_1()  # max is at start_pos (50, 70) (again, [column, row])
    image = ut.generate_responses_2()
    print("Max val:", np.max(image))
    col = np.argmax(np.max(image, axis=0))
    row = np.argmax(np.max(image, axis=1))
    print(f"Max at row {row}, col {col}: {image[row, col]}\n")

    if draw:
        temp = cv2.resize(image*255.0, (400, 400))
        cv2.imshow("Artificial PDF", temp)
        cv2.waitKey(0)
    
    mean_shift(image, (60, 40), (7, 7), 0.01)


if __name__ == "__main__":
    main()