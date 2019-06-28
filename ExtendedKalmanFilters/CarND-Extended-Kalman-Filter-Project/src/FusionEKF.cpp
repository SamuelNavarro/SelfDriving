#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;


FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  ekf_.x_ = VectorXd(4);
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  ekf_.F_ = MatrixXd(4,4);
  ekf_.P_ = MatrixXd(4,4);
  Hj_ = MatrixXd(3, 4);
  Hj_ << 0, 0, 0, 0,
	     0, 0, 0, 0,
	     0, 0, 0, 0;


  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  // Using the same P as the lectures exercise.
  ekf_.P_ << 1, 0, 0, 0,
	         0, 1, 0, 0,
			 0, 0, 1000, 0,
			 0, 0, 0, 1000;


  ekf_.F_ << 1, 0, 1, 0,
	         0, 1, 0, 1,
			 0, 0, 1, 0,
			 0, 0, 0, 1;

  H_laser_ << 1, 0, 0, 0,
	          0, 1, 0, 0;


}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */
      

    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    //cout << "EKF: " << ekf_.x_ << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      double rho = measurement_pack.raw_measurements_[0]; 
	  double phi = measurement_pack.raw_measurements_[1];
	  double rhodot = measurement_pack.raw_measurements_[2];
	  
	  // Range: rho is radial distance from origin to our pedestrian
	  // Bearing: phi is the angle between rho and x
	  // Radial Vel: rho dot is the change of rho.
	  double x = rho * cos(phi); 
	  double y = rho * sin(phi);
	  double vx = rho * cos(rhodot);
	  double vy = rho * sin(rhodot);

      ekf_.x_ << x, y, vx, vy;
	  // If we calculate h here it'll go out of scope
	 

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
	  // Initial location and zero velocity
	  ekf_.x_ << measurement_pack.raw_measurements_[0],
	             measurement_pack.raw_measurements_[1],
				 0.0,
				 0.0;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /** Prediction */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
 
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;


  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;


  float noise_ax = 9.0;
  float noise_ay = 9.0;

  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ << pow(dt, 4)/4 * noise_ax, 0, pow(dt, 3) / 2 * noise_ax, 0,
			0, pow(dt, 4) / 4 * noise_ay, 0 , pow(dt, 3) / 2 * noise_ay,
			pow(dt, 3) / 2 * noise_ax, 0 , pow(dt, 2) * noise_ax, 0,
			0, pow(dt, 3) / 2 * noise_ay, 0 , pow(dt, 2) * noise_ay;


  ekf_.Predict();
  /** Update */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
	ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // TODO: Laser updates
    ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
