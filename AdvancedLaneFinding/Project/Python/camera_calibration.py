import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from Pathlib import Path



def calibrate_camera(imgs_path, nx, ny):
    images = glob.glob(imgs_path)
    objpoints, imgpoints = [], []
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints



def get_undist_img(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def perspective_transform(undist_img):
    src = np.float32([[566, 457],
                  [750, 457],
                  [1215, undist_img.shape[0]],
                  [130, undist_img.shape[0]]
                 ])

    dst = np.float32([[offset, offset],
                 [undist_img.shape[1] - offset, offset],
                 [undist_img.shape[1] - offset, undist_img.shape[0] - offset],
                 [offset, undist_img.shape[0] - offset]])


    Minv = cv2.getPerspectiveTransform(dst, src)
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(undist, M, undist.shape[1::-1], flags=cv2.INTER_LINEAR)
    return warped, M, Minv


class Gradients:
    def __init__(self, img):
        """Constructor. It must be a warped img"""
        self.img = img


    def abs_sobel_thresh(self, img, sobel_kernel=3, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if orient=='x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        if orient=='y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobelabs = np.abs(sobel)

        scaledsobel = np.uint8(255*sobelabs/np.max(sobelabs))
        sbinary = np.zeros_like(scaledsobel)
        sbinary[(scaledsobel >= thresh[0]) & (scaledsobel <= thresh[1])] = 1

        return sbinary


    def mag_thresh(self, img, sobel_kernel=5, mag_thresh=(20, 200)):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt((sobelx)**2 + (sobely)**2)

        scaledsobel = np.uint8(255*magnitude/np.max(magnitude))
        sbinary = np.zeros_like(scaledsobel)
        sbinary[(scaledsobel >= mag_thresh[0]) & (scaledsobel <= mag_thresh[1])] = 1

        return sbinary


    def dir_threshold(self, img, sobel_kernel=3, thresh=(0.7, 1.3)):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        direction = np.arctan2(abs_sobely, abs_sobelx)
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return binary_output


  #   def combine_thresholds(img, sobel, mag, direction):
        # """
        # The img should be the warped image.
        # TODO: Get the warped image. The current doubt is that
        # I don't know the best approach to get the warped image.
        # With everything from the same class or just function.
        # """
        # grad_binary = self.abs_sobel_thresh(img, sobel_kernel=5, orient='x', thresh=(20, 100))
        # mag_img = self.mag_thres(img)
        # dir_binary = self.dir_threshold(img, sobel_kernel=5, thres=(0.7, 1.3))
        # combined = np.zeros_like(dir_binary)

        # combined[((grad



    def thresh_pipeline(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        """ Main pipelne of thresholds, color and sobel"""
        img = np.copy(img)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        return color_binary


def show_image(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(args):
    calibrate_path = Path(args.calibrate)
    img_path = Path(args.input)
    img_output = Path(args.output)
    assert img_path.exists(), 'Error: {} does not exists.'.format(img_path)
    img_output.mkdir(exist_ok=True)
    objpoints, imgpoints = calibrate_camera(calibrate_path, 9, 6)
    img = cv2.imread(img_path)
    undist = get_undist_img(img, objpoints, imgpoints)
    show_image(undist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='The path where the images or video are located')
    parser.add_argument('-o', '--output', required=True,
                        help='The path where the images or videos should be saved')
    parser.add_argument('-c', '--calibrate', required=True,
                        help='The path to the images to calibrate the camera')
    args = parser.parse_args()
    main(args)

