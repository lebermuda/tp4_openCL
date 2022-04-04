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

// Inverser la matrice par la m�thode de Gauss-Jordan; implantation s�quentielle.
void invertSequential(Matrix& iA) {

    // v�rifier que la matrice est carr�e
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    // traiter chaque rang�e
    for (size_t k = 0; k < iA.rows(); ++k) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilit� num�rique).
        size_t p = k;
        double lMax = fabs(lAI(k, k));
        for (size_t i = k; i < lAI.rows(); ++i) {
            if (fabs(lAI(i, k)) > lMax) {
                lMax = fabs(lAI(i, k));
                p = i;
            }
        }
        // v�rifier que la matrice n'est pas singuli�re
        if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

        // �changer la ligne courante avec celle du pivot
        if (p != k) lAI.swapRows(p, k);

        double lValue = lAI(k, k);
        for (size_t j = 0; j < lAI.cols(); ++j) {
            // On divise les �l�ments de la rang�e k
            // par la valeur du pivot.
            // Ainsi, lAI(k,k) deviendra �gal � 1.
            lAI(k, j) /= lValue;
        }

        // Pour chaque rang�e...
        for (size_t i = 0; i < lAI.rows(); ++i) {
            if (i != k) { // ...diff�rente de k
                // On soustrait la rang�e k
                // multipli�e par l'�l�ment k de la rang�e courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k) * lValue;
            }
        }

        //cout << "Matrice " << k << ": \n" << lAI.str() << endl;
    }

    // On copie la partie droite de la matrice AI ainsi transform�e
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
    double* data = (double*)malloc(rows * cols * sizeof(double));

    for (size_t k = 0; k < iA.rows(); k++) {

        std::copy(std::begin(lAI.getDataArray()), std::end(lAI.getDataArray()), data);

        location = -1;
        value = 0;

        //There is no easy solution in OpenACC for Max+Index https://forums.developer.nvidia.com/t/best-approach-for-reduction-problem/134817/2, https://stackoverflow.com/questions/67912346/is-there-a-faster-argmin-argmax-implementation-in-openacc
        #pragma acc parallel loop copyin(data[0:rows * cols]) reduction(max:value)
        for (size_t i = k; i < rows; i++) {
            if (fabs(data[i * cols + k]) > value) {
                value = fabs(data[i * cols + k]);
            }
        }

        for (size_t i = k; i < rows; i++) {
            if (fabs(data[i * cols + k]) == value) {
                location = i;
            }
        }


        cout << "Pivot " << k << ": " << value << " l: " << location << "\n" << endl;

        double lValue = lAI(location, k);
        std::copy(std::begin(lAI.getDataArray()), std::end(lAI.getDataArray()), data);

#pragma acc parallel loop copy(data[0:rows * cols]) copyout(rowPivot[0:cols])
        for (int j = 0; j < cols; j++) {
            data[location * cols + j] /= lValue;
            rowPivot[j] = data[location * cols + j];
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                lAI(i, j) = data[i * cols + j];
            }
        }

        lAI.swapRows(k, location);

        std::copy(std::begin(lAI.getDataArray()), std::end(lAI.getDataArray()), data);

#pragma acc parallel loop copy(data[0:rows * cols]) copyin(rowPivot[0:cols])
        for (int i = 0; i < rows; ++i) {
            if (i != k) {
                double lValue = data[i * cols + k];

                for (int j = 0; j < cols; j++) {
                    data[i * cols + j] -= rowPivot[j] * lValue;
                }
            }
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                lAI(i, j) = data[i * cols + j];
            }
        }

        cout << "Matrice " << k << ": \n" << lAI.str() << endl;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            lAI(i, j) = data[i * cols + j];
        }
    }

    for (int i = 0; i < lAI.rows(); ++i) {
        for (int j = iA.cols(); j < lAI.cols(); ++j) {
            iA(i, j - iA.cols()) = lAI(i, j);
        }
    }
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

    // v�rifier la compatibilit� des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rang�e
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

//    std::valarray<double> data = lA.getDataArray();
//
//    //double* test = (double*)malloc(lS * lS * sizeof(double));
//#pragma acc parallel loop
//    for (int i = 0; i < lS; ++i) {
//        pointData[i] = pointData[i] + 1;
//    }

    std::cout << "Matrice :\n" << lA.str() << endl;

    std::cout << "---Sequential Start" << endl;
    auto startSeq = chrono::system_clock::now();
    invertSequential2(lC);
    auto endSeq = chrono::system_clock::now();
    std::cout << "---Sequential End" << endl;

    std::cout << "Matrice inverse:\n" << lC.str() << endl;

    Matrix lResSeq = multiplyMatrix(lA, lC);

    std::cout << "Erreur Sequential : " << lResSeq.getDataArray().sum() - lS << endl;


    std::cout << "\n---Parallel Start" << endl;
    auto startPar = chrono::system_clock::now();
    invertParallel(lP);

    auto endPar = chrono::system_clock::now();
    std::cout << "---Parallel End" << endl;
    std::cout << "Matrice inverse:\n" << lP.str() << endl;


    Matrix lRes = multiplyMatrix(lA, lP);
    std::cout << "Produit des deux matrices:\n" << lRes.str() << endl;

    std::cout << "Erreur Parallel : " << lRes.getDataArray().sum() - lS << endl;

    std::cout << "Time Sequential : " << (endSeq - startSeq).count() << " , Time Parallel : " << (endPar - startPar).count() << endl;
    

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