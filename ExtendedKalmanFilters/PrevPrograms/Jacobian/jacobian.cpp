#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {
  /**
   * Compute the Jacobian Matrix
   */

  // predicted state example
  // px = 1, py = 2, vx = 0.2, vy = 0.4
  VectorXd x_predicted(4);
  x_predicted << 1, 2, 0.2, 0.4;

  MatrixXd Hj = CalculateJacobian(x_predicted);

  cout << "Hj:" << endl << Hj << endl;

  return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // TODO: YOUR CODE HERE 

  // check division by zero
  if ( sqrt(pow(px, 2) + pow(py, 2)) == 0 ){
	std::cout << "Error - Division by Zero" << "\n" << std::endl;
	Hj << 0, 0, 0, 0,
	      0, 0, 0, 0,
	      0, 0, 0, 0,
	      0, 0, 0, 0;
  }

  // compute the Jacobian matrix
  float denom = pow(px, 2) + pow(py, 2);
  Hj << px / sqrt(denom), py / sqrt(denom), 0, 0,
	    -py / denom, px / denom, 0, 0,
		py * (vx*py - vy *px) / pow(denom, 3/2), px * (vy*px - vx*py) / pow(denom, 3/2), px / sqrt(denom), py / sqrt(denom);


  return Hj;
}
