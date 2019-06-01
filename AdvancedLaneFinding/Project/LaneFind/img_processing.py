import cv2
import numpy as np
import glob

def calibrate(imgs_path, nx, ny):
    images = glob.glob(imgs_path + "/cal*.jpg")
    objpoints = []
    imgpoints = []


    for fname in images:
        img = cv2.imread(fname)
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints


def get_undist_img(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs, = cv2.calibrate_camera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def abs_sobel_thresh(img, sobel_kernel=3, orient='x',  thresh=(100,255)):
    if orient=='x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient=='y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelabs = np.abs(sobel)
    scaledsobel = np.uint8(255*sobelabs/np.max(sobelabs))
    sbinary = np.zeros_like(scaledsobel)
    sbinary[(scaledsobel >= thresh[0]) & (scaledsobel <= thresh[1])] = 255
    return sbinary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt((sobelx)**2 + (sobely)**2)
    scaledsobel = np.uint8(255*magnitude/np.max(magnitude))
    sbinary = np.zeros_like(scaledsobel)
    sbinary[(scaledsobel >= mag_thresh[0]) & (scaledsobel <= mag_thresh[1])] = 255
    return sbinary


def dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 255
    return binary_output


def perspective_transform(undist_img):
    offset = 50
    src = np.float32([[558, 462],
                  [740, 462],
                  [1230, undist_img.shape[0]],
                  [undist_img.shape[1] / 7, undist_img.shape[0]]
                 ])

    dst = np.float32([[undist_img.shape[1] / 9 , offset],
                 [undist_img.shape[1] + offset, offset],
                 [undist_img.shape[1] - offset, undist_img.shape[0]],
                 [undist_img.shape[1] / 6, undist_img.shape[0]]])

    Minv = cv2.getPerspectiveTransform(dst, src)
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(undist_img, M, undist_img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def color_pipeline(img, l_thresh = (120, 255), s_thresh=(120, 255), sx_thresh=(20, 255)):
    img = np.copy(img)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[...,2]
    l_channel = hls[...,1]
    h_channel = hls[...,0]

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Sobel x with l_channel for shadow in the road
    sobelx_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx_l = np.absolute(sobelx_l)
    scaled_sobel_l = np.uint8(255*abs_sobelx_l/np.max(abs_sobelx_l))

    # sobelx in s channel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255

    # sobelx in l channel
    sxbinary_l = np.zeros_like(scaled_sobel_l)
    sxbinary_l[(scaled_sobel_l >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255


    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 255

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))


    combined_binary = np.zeros_like(sxbinary)
    combined_binary[ ((sxbinary == 255) | (sxbinary_l == 255)) | ((s_binary == 255) & (l_binary == 255)) ] = 255

    gradient_combined = np.zeros_like(sxbinary)
    gradient_combined[((sxbinary==255) | (sxbinary_l) == 255)] = 255


    color_combined = np.zeros_like(sxbinary)
    color_combined[((s_binary==255) & (l_binary==255))] = 255
    return combined_binary
