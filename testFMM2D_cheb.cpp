#include "kernel.hpp"
#include "ACA.hpp"
#include "FMM2DTree_cheb.hpp"

int main(int argc, char* argv[]) {
	int sqrtRootN		=	atoi(argv[1]);
	int nParticlesInLeafAlong1D	=	atoi(argv[2]); // assuming the particles are located at tensor product chebyNodes/uniform
	int L			=	atoi(argv[3]); // half side length of square centered at origin
	int TOL_POW = atoi(argv[4]); //tolerance of ACA
	double start, end;
	int nLevels		=	ceil(log(double(sqrtRootN)/nParticlesInLeafAlong1D)/log(2));
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	std::vector<pts2D> particles_X, particles_Y; // these get defined in the FMM2DTree object
	userkernel* mykernel		=	new userkernel(particles_X, particles_Y);
	FMM2DTree<userkernel>* A	=	new FMM2DTree<userkernel>(mykernel, sqrtRootN, nLevels, nParticlesInLeafAlong1D, L, TOL_POW);

	// A->set_Uniform_Nodes();
	A->set_Standard_Cheb_Nodes();

	A->createTree();
	A->assign_Tree_Interactions();
	A->assign_Center_Location();
	A->assignChargeLocations();
	A->assignNonLeafChargeLocations();
	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);
	std::cout << std::endl << "N: " << A->N << std::endl;
	std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;
	/////////////////////////////////////////////////////////////////////////
	start	=	omp_get_wtime();
	A->getNodes();
	end		=	omp_get_wtime();

	double timegetNodes=	(end-start);

	start	=	omp_get_wtime();
	A->assemble_M2L();
	end		=	omp_get_wtime();

	double timeassemble=	(end-start);
	std::cout << std::endl << "Total Time taken to assemble is: " << timegetNodes+timeassemble << std::endl;
	/////////////////////////////////////////////////////////////////////////
	int N = A->N;
	Eigen::VectorXd b = Eigen::VectorXd::Random(N);
	A->assignLeafCharges(b);

	start	=	omp_get_wtime();
	A->evaluate_M2M();
	A->evaluate_M2L();
	A->evaluate_L2L();
	A->evaluate_NearField();
	Eigen::VectorXd AFMM_Ab;
	A->collectPotential(AFMM_Ab);
	A->reorder(AFMM_Ab);
	end		=	omp_get_wtime();
	double timeMatVecProduct=	(end-start);
	std::cout << std::endl << "Time taken to do Mat-Vec product is: " << timeMatVecProduct << std::endl;

	double sum;
	A->findMemory(sum);

	double err;
	start		=	omp_get_wtime();
	Eigen::VectorXd true_Ab = Eigen::VectorXd::Zero(N);
	#pragma omp parallel for
	for (size_t i = 0; i < N; i++) {
		#pragma omp parallel for
		for (size_t j = 0; j < N; j++) {
			true_Ab(i) += A->K->getMatrixEntry(i,j)*b(j);
		}
	}
	end		=	omp_get_wtime();
	double exact_time =	(end-start);

	err = (true_Ab - AFMM_Ab).norm()/true_Ab.norm();
	std::cout << std::endl << "Error in the solution is: " << err << std::endl << std::endl;
}
