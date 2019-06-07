import numpy as np
import cv2


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
    gradient_combined[((sxbinary==255) | (sxbinary_l) == 255)] =255


    color_combined = np.zeros_like(sxbinary)
    color_combined[((s_binary==255) & (l_binary==255))] = 255
    return combined_binary


def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50

    window_height = np.int(binary_warped.shape[0]//nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0, 255, 0), 3)
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0, 255, 0), 3)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(undist, binary_warped, Minv, left_fit, right_fit):
    margin = 100

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                      (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    lanes_img = np.zeros_like(out_img)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(lanes_img, np.int_([pts]), (0, 55, 0))
    ## Yellow fitted lines. To see if this is usefull to 
    # get the position of the car
    left_pts = np.vstack((left_fitx, ploty)).astype(np.int32).T
    right_pts = np.vstack((right_fitx, ploty)).astype(np.int32).T
#     cv2.polylines(result, [left_pts], False, (255, 255, 0), 10)
#     cv2.polylines(result, [right_pts], False, (255, 255, 0), 10)

    # Color in left and right line pixels
    lanes_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    lanes_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

    newwarp = cv2.warpPerspective(lanes_img, Minv, (lanes_img.shape[1], lanes_img.shape[0]))

    output = cv2.addWeighted(undist, 0.8, newwarp, 1, 0)
    return output, left_pts, right_pts


