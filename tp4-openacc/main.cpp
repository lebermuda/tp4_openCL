#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <sstream>

using namespace std;

struct mpi_double_int {
    double value;
    int location;
};

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {

    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    // traiter chaque rangée
    for (size_t k = 0; k < iA.rows(); ++k) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        size_t p = k;
        double lMax = fabs(lAI(k, k));
        for (size_t i = k; i < lAI.rows(); ++i) {
            if (fabs(lAI(i, k)) > lMax) {
                lMax = fabs(lAI(i, k));
                p = i;
            }
        }
        // vérifier que la matrice n'est pas singulière
        if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

        // échanger la ligne courante avec celle du pivot
        if (p != k) lAI.swapRows(p, k);

        double lValue = lAI(k, k);
        for (size_t j = 0; j < lAI.cols(); ++j) {
            // On divise les éléments de la rangée k
            // par la valeur du pivot.
            // Ainsi, lAI(k,k) deviendra égal à 1.
            lAI(k, j) /= lValue;
        }

        // Pour chaque rangée...
        for (size_t i = 0; i < lAI.rows(); ++i) {
            if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k) * lValue;
            }
        }

        //cout << "Matrice " << k << ": \n" << lAI.str() << endl;
    }

    // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    for (unsigned int i = 0; i < iA.rows(); ++i) {
        iA.getRowSlice(i) = lAI.getDataArray()[slice(i * lAI.cols() + iA.cols(), iA.cols(), 1)];
    }
}

void invertSequential2(Matrix& iA) {
    assert(iA.rows() == iA.cols());

    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    mpi_double_int gMax;
    double rowPivot[lAI.cols()];

    for (size_t k = 0; k < iA.rows(); k++) {
        gMax.value = 0;
        gMax.location = 0;

        for (size_t i = k; i < lAI.rows(); i++) {
            if (fabs(lAI(i, k)) > gMax.value) {
                gMax.value = fabs(lAI(i, k));
                gMax.location = i;
            }
        }

        double lValue = lAI(gMax.location, k);
        for (int j = 0; j < lAI.cols(); j++) {
            lAI(gMax.location, j) /= lValue;
            rowPivot[j] = lAI(gMax.location, j);
        }

        lAI.swapRows(k, gMax.location);

        for (int i = 0; i < lAI.rows(); ++i) {
            if (i != k) {
                double lValue = lAI(i, k);

                for (int j = 0; j < lAI.cols(); j++) {
                    lAI(i, j) -= rowPivot[j] * lValue;
                }
            }
        }
    }

    for (int i = 0; i < lAI.rows(); ++i) {
        for (int j = iA.cols(); j < lAI.cols(); ++j) {
            iA(i, j - iA.cols()) = lAI(i, j);
        }
    }
}

void invertParallel(Matrix& iA) {
    assert(iA.rows() == iA.cols());

    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    int location;
    double value;

    size_t rows = lAI.rows();
    size_t cols = lAI.cols();

    double* rowPivot = (double*)malloc(cols * sizeof(double));
    double* dataPointer = std::begin(lAI.getDataArray());

    for (size_t k = 0; k < iA.rows(); k++) {
        location = 0;
        value = 0;
        //There is no easy solution in OpenACC for Max+Index https://forums.developer.nvidia.com/t/best-approach-for-reduction-problem/134817/2, https://stackoverflow.com/questions/67912346/is-there-a-faster-argmin-argmax-implementation-in-openacc
#pragma acc parallel loop copyin(dataPointer[0:rows * cols]) reduction(max:value)
        for (size_t i = k; i < rows; i++) {
            if (fabs(dataPointer[i * cols + k]) > value) {
                value = fabs(dataPointer[i * cols + k]);
            }
        }

        for (size_t i = k; i < rows; i++) {
            if (fabs(dataPointer[i * cols + k]) == value) {
                location = i;
            }
        }

        //cout << "Pivot " << k << ": " << value << " l: " << location << "\n" << endl;

        double lValue = lAI(location, k);
#pragma acc parallel loop copy(dataPointer[0:rows * cols]) copyout(rowPivot[0:cols])
        for (int j = 0; j < cols; j++) {
            dataPointer[location * cols + j] /= lValue;
            rowPivot[j] = dataPointer[location * cols + j];
        }

        lAI.swapRows(k, location);

#pragma acc parallel loop copy(dataPointer[0:rows * cols]) copyin(rowPivot[0:cols]) gang worker
        for (int i = 0; i < rows; ++i) {
            if (i != k) {
                double lValue = dataPointer[i * cols + k];

#pragma acc loop vector
                for (int j = 0; j < cols; j++) {
                    dataPointer[i * cols + j] -= rowPivot[j] * lValue;
                }
            }
        }

        //cout << "Matrice " << k << ": \n" << lAI.str() << endl;
    }

    for (int i = 0; i < lAI.rows(); ++i) {
        for (int j = iA.cols(); j < lAI.cols(); ++j) {
            iA(i, j - iA.cols()) = lAI(i, j);
        }
    }

    free(rowPivot);
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for (size_t i = 0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for (size_t j = 0; j < lRes.cols(); ++j) {
            lRes(i, j) = (iMat1.getRowCopy(i) * iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char** argv) {
    int a = 0;
    
    srand((unsigned)time(NULL));

    unsigned int lS = 5;
    if (argc >= 2) {
        lS = atoi(argv[1]);
    }

    MatrixRandom lA(lS, lS);
    Matrix lC(lA);
    Matrix lP(lA);

    //std::cout << "Matrice :\n" << lA.str() << endl;

    std::cout << "---Sequential Start" << endl;
    auto startSeq = std::chrono::high_resolution_clock::now();
    //invertSequential2(lC);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::cout << "---Sequential End" << endl;

    //std::cout << "Matrice inverse:\n" << lC.str() << endl;

    //Matrix lResSeq = multiplyMatrix(lA, lC);

    //std::cout << "Erreur Sequential : " << lResSeq.getDataArray().sum() - lS << endl;


    std::cout << "\n---Parallel Start" << endl;
    auto startPar = std::chrono::high_resolution_clock::now();
    invertParallel(lP);

    auto endPar = std::chrono::high_resolution_clock::now();
    std::cout << "---Parallel End" << endl;
    //std::cout << "Matrice inverse:\n" << lP.str() << endl;


    //Matrix lRes = multiplyMatrix(lA, lP);
    //std::cout << "Produit des deux matrices:\n" << lRes.str() << endl;

    //std::cout << "Erreur Parallel : " << lRes.getDataArray().sum() - lS << endl;

    std::cout << "Time Sequential : " << ((std::chrono::duration<double>)(endSeq - startSeq)).count() << "s" << " , Time Parallel : " << ((std::chrono::duration<double>)(endPar - startPar)).count() << "s" << endl;

    return 0;
}


//int main(int argc, char* argv[])
//{
//    // Size of vectors
//    int n = 10000;
//
//    // Input vectors
//    double* __restrict a;
//    double* __restrict b;
//    // Output vector
//    double* __restrict c;
//
//    // Size, in bytes, of each vector
//    size_t bytes = n * sizeof(double);
//
//    // Allocate memory for each vector
//    a = (double*)malloc(bytes);
//    b = (double*)malloc(bytes);
//    c = (double*)malloc(bytes);
//
//    // Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
//    int i;
//    for (i = 0; i < n; i++) {
//        a[i] = sin(i) * sin(i);
//        b[i] = cos(i) * cos(i);
//    }
//
//    // sum component wise and save result into vector c
//#pragma acc kernels copyin(a[0:n],b[0:n]), copyout(c[0:n])
//    for (i = 0; i < n; i++) {
//        c[i] = a[i] + b[i];
//    }
//
//    // Sum up vector c and print result divided by n, this should equal 1 within error
//    double sum = 0.0;
//    for (i = 0; i < n; i++) {
//        sum += c[i];
//    }
//    sum = sum / n;
//    printf("final result: %f\n", sum);
//
//    // Release memory
//    free(a);
//    free(b);
//    free(c);
//
//    return 0;
//}