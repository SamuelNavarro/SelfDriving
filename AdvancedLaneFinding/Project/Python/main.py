import cv2
import numpy as np
import glob
import img_processing
import lane_lines
import utils


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def measure_curvature_real(self, img):
        ym_per_pix = 30/720
        xm_per_pix = 3.7/900

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

        left_fit_cr = np.polyfit(self.ally[0]*ym_per_pix, self.allx[0]*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ally[1]*ym_per_pix, self.allx[1]*xm_per_pix, 2)

        y_eval = np.max(ploty)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad

    def compute_location(self, warped, left_fitx, right_fitx):
        xm_per_pix = 3.7/900
        dist_right = right_fitx[-1]
        dist_left = left_fitx[-1]
        main_dist = (dist_right+dist_left)/2 - warped.shape[1]/2
        main_dist = main_dist*xm_per_pix

        return main_dist

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            print("The function failed to fit a line")
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        return left_fitx, right_fitx, ploty

    def find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
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

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
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
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.allx = (leftx, rightx)
        self.ally = (lefty, righty)

        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        return left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds

    def search_around_poly(self, binary_warped):
        margin = 100

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = np.polyfit(self.ally[0], self.allx[0], 2)
        right_fit = np.polyfit(self.ally[1], self.allx[1], 2)

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                          (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.allx = (leftx, rightx)
        self.ally = (lefty, righty)

        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        return left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds

    def return_to_undist(self, undist, binary_warped, Minv, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds):

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        lanes_img = np.zeros_like(out_img)

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lanes_img, np.int_([pts]), (0, 55, 0))

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Color in left and right line pixels
        lanes_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
        lanes_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

        newwarp = cv2.warpPerspective(lanes_img, Minv, (lanes_img.shape[1], lanes_img.shape[0]))

        output = cv2.addWeighted(undist, 0.8, newwarp, 1, 0)

        return output


def write_video(img, objpoints, imgpoints):
    lines = Line()
    undist = img_processing.get_undist_img(img, objpoints, imgpoints)
    warped, M, Minv = img_processing.perspective_transform(undist)
    combined_binary = lane_lines.color_pipeline(warped)

    if lines.detected is True:
        left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds = lines.search_around_poly(combined_binary)
        left_curverad, right_curverad = lines.measure_curvature_real(combined_binary)
        lines.radius_of_curvature = str(int(left_curverad))
        lines.line_base_pos = lines.compute_location(combined_binary, left_fitx, right_fitx)
        output = lines.return_to_undist(undist, combined_binary, Minv, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds)
        cv2.putText(output, "Radius: " + str(int(lines.radius_of_curvature)) + "(m)",
                (output.shape[0] // 2, output.shape[1] // 8),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        cv2.putText(output, "Vehicle is : " + str(round(lines.line_base_pos, 2)) + " (m) to the left",
                (output.shape[0] // 2, output.shape[1] // 6),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
    else:
        left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds = lines.find_lane_pixels(combined_binary)
        left_curverad, right_curverad = lines.measure_curvature_real(combined_binary)
        lines.radius_of_curvature = str(int(left_curverad))
        lines.line_base_pos = lines.compute_location(combined_binary, left_fitx, right_fitx)
        output = lines.return_to_undist(undist, combined_binary, Minv, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds)
        cv2.putText(output, "Radius: " + str(int(lines.radius_of_curvature)) + "(m)",
                (output.shape[0] // 2, output.shape[1] // 8),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        cv2.putText(output, "Vehicle is : " + str(round(lines.line_base_pos, 2)) + " (m) to the left",
                (output.shape[0] // 2, output.shape[1] // 6),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        lines.detected = True

    return output


def process_video(video_path, objpoints, imgpoints):
    video_capture = cv2.VideoCapture(video_path)
    count = 0
    lines = Line()

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter('../test_videos_output/project_video_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

    while (video_capture.isOpened()):
        ret, frame = video_capture.read()
        if ret:
            undist = img_processing.get_undist_img(frame, objpoints, imgpoints)
            warped, M, Minv = img_processing.perspective_transform(undist)
            combined_binary = lane_lines.color_pipeline(warped)

            if lines.detected == True:
                left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds = lines.search_around_poly(combined_binary)
                left_curverad, right_curverad = lines.measure_curvature_real(combined_binary)
                lines.radius_of_curvature = str(int(left_curverad))
                lines.line_base_pos = lines.compute_location(combined_binary, left_fitx, right_fitx)
                output = lines.return_to_undist(undist, combined_binary, Minv, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds)
                cv2.putText(output, "Radius: " + str(int(lines.radius_of_curvature)) + "(m)",
                        (output.shape[0] // 2, output.shape[1] // 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
                cv2.putText(output, "Vehicle is : " + str(round(lines.line_base_pos, 2)) + " (m) to the left",
                        (output.shape[0] // 2, output.shape[1] // 6),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
            else:
                left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds = lines.find_lane_pixels(combined_binary)
                left_curverad, right_curverad = lines.measure_curvature_real(combined_binary)
                lines.radius_of_curvature = str(int(left_curverad))
                lines.line_base_pos = lines.compute_location(combined_binary, left_fitx, right_fitx)
                output = lines.return_to_undist(undist, combined_binary, Minv, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds)
                cv2.putText(output, "Radius: " + str(int(lines.radius_of_curvature)) + "(m)",
                        (output.shape[0] // 2, output.shape[1] // 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
                cv2.putText(output, "Vehicle is : " + str(round(lines.line_base_pos, 2)) + " (m) to the left",
                        (output.shape[0] // 2, output.shape[1] // 6),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
                lines.detected = True
            # cv2.imshow('frame',output)
            out.write(output)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    objpoints, imgpoints = img_processing.calibrate_camera("../camera_cal/", 9, 6)
    process_video("../test_videos/project_video.mp4", objpoints, imgpoints)
