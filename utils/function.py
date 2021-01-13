import numpy as np

def check_threshold(img_BGR, sat_threshold, pixcount_th):
    if img_BGR.sum() < pixcount_th:
        return False

    hsv = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if (s > sat_threshold).sum() < pixcount_th:
        return False

    return True


def reverse_padsize(img, reduce, sz):

    shape = img.shape

    pad0 = (reduce*sz + shape[0] // (reduce*sz)) % (reduce*sz)
    pad1 = (reduce*sz + shape[1] // (reduce*sz)) % (reduce*sz)
    pad_x = (pad0*2, pad0+pad0*2)
    pad_y = (pad1*2, pad1+pad1*2)

    return pad_x, pad_y


def img_reshape(img):
    '''return the shape of an image in the format [x_shape, y_shape, color_channels]
    raw image (dx,dy,3)
    tiling of padded image (K,512,512)'''
    shape = img.shape
    if len(img.shape) == 3:
        img = img.squeeze()
        img_reshaped = img.reshape(img.shape[0], img.shape[1], 3)
        shape = img_reshaped.shape

    return shape, img


def reconstructed_img(img_raw, mask_2d):
    '''inputs: raw image (dx,dy,3), tiling of padded image (K,512,512)
    output: original image reconstructed (dx,dy,3)'''
    tiled_mask = np.ndarray([100, 512, 512])
    # loop over tiles (n)
            # check thresholding
            # if thresholding passes
            # perform inference
                # pad image(split_image_mask_into_tiles(self, reduce=1, sz=512)
                # pad mask(split_image_mask_into_tiles(self, reduce=1, sz=512)
            # else
                # mask = 0 (512 x 512)
    n = tiled_mask.shape[0]  # numpy arrays of size (i, 512 x 512)
    reconstruct_img_count = 0
    reconstruct_idx = []
    num_original_images = 0
    original_images = []

    for i in range(n):
        tiled_mask = tiled_mask[i, :, :]
        if check_threshold(tiled_mask, sat_threshold, pixcount_th):
            tiled_img, tiled_mask = split_image_mask_into_tiles(img_raw, mask_2d, reduce=1, sz=512)
        else:
            mask_2d = 0 * (512, 512)
            reconstruct_img_count += 1
            reconstruct_idx.append(i)
            for t in reconstruct_idx:
                # stitch prediction tiles to make a big 2D array
                original_image_reconstructed = reverse_padsize(tiled_mask, reduce=1, sz=512)
                # original_image_reconstructed[t] = tiled_mask[pad_x: -pad_x, pad_y: -pad_y]
                # reconstruct_img[t] = reconstruct_img.reshape(tiled_mask.shape[0], tiled_mask.shape[0], 3)

                original_image_reconstructed = img_reshape(original_image_reconstructed)

                num_original_images += 1
                original_images.append(t)

                reconstructed_img_rle = mask_2d_to_rle(original_image_reconstructed)
                reconstructed_img_dict['''whatever the id is for the raw image'''] = reconstructed_img_rle

                shape = original_image_reconstructed.shape

    return original_image_reconstructed, shape # expected output is original image reconstructed (dx, dy, 3)
