This library does matrix vector product for H2 matrices with well-separated admissibility condition (\eta=\sqrt(2)).

The low-rank approximations of matrix sub-blocks is constructed using a new Nested Cross Approximation.

The charges are distributed at the tensor product Chebyshev nodes of the square [-L,L]^2.

The matrix entries are to be defined in function "getMatrixEntry(i,j)" of the kernel.hpp file.

The vector to be applied to the matrix is to be defined in VectorXd "b" of the testFMM2D_cheb.cpp.

It takes the following inputs at run-time.

sqrtRootN: square root of N. It considers nodes to be located on the tensor product Chebyshev grid with sqrtRootN nodes in each dimension.

nParticlesInLeafAlong1D: maximum number of particles in a leaf cell.

L: half side length of the square domain

TOL_POW: tolerance given for NCA algorithm

A sample run looks like this:

./testFMM2D_cheb 100 10 1 12

N: 10000

Time taken to create the tree is: 0.002234

Total Time taken to assemble is: 2.50384

Time taken to do Mat-Vec product is: 0.032645

Error in the solution is: 1.71297e-10
