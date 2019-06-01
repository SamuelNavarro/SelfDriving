import cv2

def show_image(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def measure_curvature_pixels():
    leftx, lefty, rightx, righty, _ = find_lane_pixels(combined_binary)


    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )

    y_eval = np.max(ploty)

    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad, right_curverad


def measure_curvature_real(img):
    ym_per_pix = 30/720
    xm_per_pix = 3.7/900

    leftx, lefty, rightx, righty, _ = find_lane_pixels(img)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)


    y_eval = np.max(ploty)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


def compute_location(warped, left_pts, right_pts):
    xm_per_pix = 3.7/900
    dist_right = right_pts[-1][0]
    dist_left = left_pts[-1][0]
    main_dist = (dist_right+dist_left)/2 - warped.shape[1]/2
    main_dist = main_dist*xm_per_pix

    return main_dist
