from typing import Dict, Any

import utils
import numpy as np
import PIL
from PIL import Image

# from IPython.display import display
NDArray = Any


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    # first we turn the picture to graysacle and get the gradient

    img = utils.to_grayscale(image)

    height = img.shape[0]
    width = img.shape[1]

    # we need to calculate how many horizontal and vertical seams we will have to find
    k_height = abs(height - out_height)
    k_width = abs(width - out_width)

    if (k_width == 0):
        # no need for work on vertical seams :)
        outimg = np.copy(image)
        redimg = outimg

    else:
        # lot's of work to be done!

        # finding the horizontal seams
        h_seams = np.array(find_k_seams(img, k_width, forward_implementation)).astype(int)
        outimg = np.zeros((height, out_width, 3), dtype=np.uint8)
        redimg = np.copy(image)

        # start by dealing with vertical seams
        k = k_width

        # create the red coloured seams on a copy of the OG picture
        for i in range(height):
            for j in range(k):
                redimg[i][h_seams[i][width - k + j]] = [255, 0, 0]

        if (width > out_width):
            # we need to remove the seams we found
            # create the resized copy:
            for i in range(height):
                for j in range(out_width):
                    outimg[i][j] = image[i][h_seams[i][j]]


        elif (width < out_width):
            # we need to duplicate the seams we found
            for i in range(height):
                outimg[i][:width] = image[i]
            # we copied the original image to the out matrix, leaving free space (k columns) for the seams we want to duplicate

            seams_indices = np.rot90(np.rot90(h_seams, k=1, axes=(0, 1))[:k], k=-1, axes=(0, 1))
            seams_indices.sort()

            # copy the k seam color values to the right end of the outimg
            for j in range(k_width):
                for i in range(height):
                    outimg[i][width + j] = image[i][seams_indices[i][j]]

            # "pushing" the seams into their respective place with roll operations
            for i in range(height):
                for j in range(k_width):
                    x_to_replicate = seams_indices[i][(-1 - j)]
                    outimg[i][x_to_replicate:] = np.roll(outimg[i][x_to_replicate:], shift=1, axis=0)

    if (k_height == 0):
        # no need for work on vertical seams :)
        outimg2 = outimg
        blackimg = outimg

    else:
        # lot's of work to be done!

        # rotating the updated working image and finding the vertical seams
        outimg = np.rot90(outimg, k=1, axes=(0, 1))
        img = utils.to_grayscale(outimg)  # already rotated
        # img = np.rot90(img, k=1, axes=(0, 1))

        # finding the horizontal seams
        k = k_height
        w_seams = np.array(find_k_seams(img, k, forward_implementation)).astype(int)

        blackimg = np.copy(outimg)  # already rotated
        outimg2 = np.zeros((out_width, out_height, 3), dtype=np.uint8)  # created "rotated" already

        # create the black coloured seams on a rotated copy of the outimg (with the removed/added seams)
        for i in range(out_width):
            for j in range(k):
                blackimg[i][w_seams[i][(-1 - j)]] = [0, 0, 0]

        # rotating the black lined image back to it's original state
        blackimg = np.rot90(blackimg, k=-1, axes=(0, 1))

        # altering the outimg
        if (height > out_height):
            # we need to remove the seams we found
            # create the resized copy:
            for i in range(out_width):
                for j in range(out_height):
                    outimg2[i][j] = outimg[i][w_seams[i][j]]


        elif (height < out_height):
            # we need to duplicate the seams we found
            for i in range(out_width):
                outimg2[i][:height] = outimg[i]
            # we copied the original image to the out matrix, leaving free space (k columns) for the seams we want to duplicate

            seams_indices = np.rot90(np.rot90(w_seams, k=1, axes=(0, 1))[:k], k=-1, axes=(0, 1))
            seams_indices.sort()

            # copy the k seam color values
            for j in range(k_height):
                for i in range(out_width):
                    outimg2[i][height + j] = outimg[i][seams_indices[i][j]]

            # "pushing" the seams into their respective place with roll operations
            for i in range(out_width):
                for j in range(k_height):
                    x_to_replicate = seams_indices[i][(-1 - j)]
                    outimg2[i][x_to_replicate:] = np.roll(outimg2[i][x_to_replicate:], shift=1, axis=0)

                    # x_to_replicate = seams_indices[i][(-1 - j)]
                    # outimg[i][x_to_replicate:] = np.roll(outimg[i][x_to_replicate:], shift=1, axis=0)

        # rotating the outimg2 back to it's original state
        outimg2 = np.rot90(outimg2, k=-1, axes=(0, 1))

    return {'resized': outimg2, 'vertical_seams': redimg, 'horizontal_seams': blackimg}

    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def find_k_seams(img: NDArray, k: int, forward_implementation: bool) -> Dict[str, NDArray]:
    height = img.shape[0]
    width = img.shape[1]

    # creating the 'index' matrix containing [[0,1,2,...,w-1],
    #                                         .
    #                                         .
    #                                         .
    #                                        [0,1,2,...,w-1]]

    F = np.indices(img.shape, dtype='int', sparse=False)
    # FY = F[0]   # the Y indices saver matrix as explained in the HW
    FX = F[1]  # the X indices saver matrix as explained in the HW
    for iteration in range(k):
        # init zero matrix for M and gradient matrix E
        E = utils.get_gradients(img)
        M = np.copy(E)

        CL = np.zeros(img.shape)
        CR = np.zeros(img.shape)
        CV = np.zeros(img.shape)
        # calculate CL, CR, CV
        if forward_implementation:
            Dict1 = C(img)
            CL = Dict1['L']
            CR = Dict1['R']
            CV = Dict1['V']

        for i in range(1, height):
            zero_column = [0]
            L_row = np.concatenate([zero_column, M[i - 1][0:-1]]) + CL[i]
            M_row = M[i - 1] + CV[i]
            R_row = np.concatenate([M[i - 1][1:], zero_column]) + CR[i]
            M[i] += np.minimum(np.minimum(R_row, M_row), L_row)
            M[i][0] = E[i][0] + np.minimum(M_row[0], R_row[0])
            M[i][width - 1] = E[i][width - 1] + np.minimum(M_row[width - 1], L_row[width - 1])

        # now we have M, let's calculate the minimum seam :)
        seam = np.zeros(height)
        # we will start from the bottom, in the minimal value of the bottom row of M
        seam[height - 1] = int(np.argmin(M[height - 1][0:width]))

        for i in range(height - 1, 0, -1):
            index = int(seam[i])
            if index == 0:  # we're in the first col
                if M[i][index] == E[i][index] + M[i - 1][index] + CV[i][index]:
                    seam[i - 1] = index
                else:
                    seam[i - 1] = index + 1
            if index == width - 1:  # we're in the last col
                if M[i][index] == E[i][index] + M[i - 1][index] + CV[i][index]:
                    seam[i - 1] = index
                else:
                    seam[i - 1] = index - 1
            else:
                if M[i][index] == E[i][index] + M[i - 1][index] + CV[i][index]:
                    seam[i - 1] = index
                elif M[i][index] == E[i][index] + M[i - 1][index - 1] + CL[i][index]:
                    seam[i - 1] = index - 1
                else:
                    seam[i - 1] = index + 1

        # now we have the x values of our seam saved in the seam array, the y values correspond to the array's

        seam = seam.astype(int)

        for s in range(len(seam)):
            # slice the left sub-array of each row ant roll it to the left by one index
            FX[s][seam[s]:] = np.roll(FX[s][seam[s]:], -1)
            # do the same for the picture we're processing
            img[s][seam[s]:] = np.roll(img[s][seam[s]:], -1)
        # now that we "removed" the seam, width is smaller
        width -= 1
        img = np.delete(img, -1, axis=1)

    return FX


def C(img: NDArray) -> Dict[str, NDArray]:
    # calculate |I(i,j+1)-I(i,j-1)|
    zero_column = np.broadcast_to(np.NAN, [img.shape[0],
                                           1])  # we changed this line from the presentation, img.shape[1]->img.shape[0]
    left = np.concatenate([zero_column, img[:, 0:-1]], axis=1)
    right = np.concatenate([img[:, 1:], zero_column], axis=1)
    CV = np.abs(left - right)
    CV[np.isnan(CV)] = 255.0

    Dict1 = {'V': CV}

    # calculate the 'up' matrix
    zero_row = np.broadcast_to(np.NAN, [1, img.shape[1]])
    up = np.concatenate([zero_row, img[0:-1, :]], axis=0)

    # up[np.isnan(up)] = 255.0

    CL = np.abs(left - up) + CV
    CL[np.isnan(CL)] = 255.0
    Dict1['L'] = CL

    CR = np.abs(right - up) + CV
    CR[np.isnan(CR)] = 255.0
    Dict1['R'] = CR

    CV[np.isnan(CV)] = 255.0

    Dict1['V'] = CV

    return Dict1
