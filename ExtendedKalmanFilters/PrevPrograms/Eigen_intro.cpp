#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>


using namespace std;
using namespace Eigen;


Eigen::VectorXd x;
Eigen::MatrixXd P;
Eigen::VectorXd u;
Eigen::MatrixXd F;
Eigen::MatrixXd H;
Eigen::MatrixXd R;
Eigen::MatrixXd I;
Eigen::MatrixXd Q;

vector<VectorXd> measurements;
void filter(VectorXd &x, MatrixXd &P);


int main(){
	VectorXd my_vector(2);
	my_vector << 10, 20;
	cout << my_vector << "\n" << endl;


	MatrixXd my_matrix(2, 2);
	my_matrix << 1, 2, 3, 4;

	cout << my_matrix << "\n" << endl;
	

	// We can set each matrix value
	// explicitly.
	my_matrix(1,0) = 11;
	my_matrix(0,1) = 22;

	cout << my_matrix << "\n" << endl;


	MatrixXd my_matrix_t = my_matrix.transpose();

	cout << my_matrix_t << "\n" << endl;

	MatrixXd my_matrix_i = my_matrix.inverse();
	cout << my_matrix_i << "\n" << endl;


	MatrixXd another_matrix = my_matrix*my_vector;
	cout << another_matrix << "\n" << endl;


	return 0;

}
