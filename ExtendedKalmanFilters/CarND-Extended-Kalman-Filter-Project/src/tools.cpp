#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /** TODO: Calculate the RMSE here.  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // TODO: YOUR CODE HERE
  if (estimations.size() == 0 || estimations.size() != ground_truth.size()){
    std::cout << "Invalid estimation or ground truth data" << std::endl;
	return rmse;
  }
  // TODO: accumulate squared residuals
  for (unsigned i=0; i < estimations.size(); ++i) {
    // ... your code here
    VectorXd residual = estimations[i] - ground_truth[i];
	residual = residual.array()*residual.array();
	rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}



MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  if (sqrt(pow(px, 2) + pow(py, 2)) == 0){
    std::cout << "Error - Divison by Zero" << "\n" << std::endl;
	Hj << 0, 0, 0, 0,
	      0, 0, 0, 0,
	      0, 0, 0, 0;
  }

  double denom = pow(px, 2) + pow(py, 2);
  Hj << px / sqrt(denom), py / sqrt(denom), 0, 0,
	    -py / denom, px / denom, 0, 0,
		py * (vx*py - vy *px)/pow(denom, 3.0/2.0), px * (vy*px - vx*py)/pow(denom, 3.0/2.0), px/sqrt(denom), py/sqrt(denom);
  return Hj;
}
