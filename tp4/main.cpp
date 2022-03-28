#define CL_TARGET_OPENCL_VERSION 220
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
// System includes
#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
// #include <OpenCL/cl.h>
#include <CL/cl.h>


// Project includes

// Constants, globals
const int ELEMENTS = 2048;   // elements in each vector

// Signatures
char* readSource(const char *sourceFilename); 

int main(int argc, char ** argv)
{
	printf("Running Vector Addition program\n\n");
	
	size_t datasize = sizeof(int)*ELEMENTS;
	
	int *A, *B;   // Input arrays
	int *C;       // Output array
	
	// Allocate space for input/output data
	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);
	if(A == NULL || B == NULL || C == NULL) {
		perror("malloc");
		exit(-1);
	}
	
	// Initialize the input data
	for(int i = 0; i < ELEMENTS; i++) {
		A[i] = i;
		B[i] = i;
	}
	
	cl_int status;  // use as return value for most OpenCL functions
	
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms;
	
	// Query for the number of recongnized platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // status=clGetPlatformIDs(1, platforms, &numPlatforms);
	if(status != CL_SUCCESS) {
		printf("clGetPlatformIDs failed\n");
		exit(-1);
	}
	
	// Make sure some platforms were found 
	if(numPlatforms == 0) {
		printf("No platforms detected.\n");
		exit(-1);
	}
	
	// Allocate enough space for each platform
	platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
	if(platforms == NULL) {
		perror("malloc");
		exit(-1);
	}
	
	// Fill in platforms
	clGetPlatformIDs(numPlatforms, platforms, NULL);
	if(status != CL_SUCCESS) {
		printf("clGetPlatformIDs failed\n");
		exit(-1);
	}
	
	// Print out some basic information about each platform
	printf("%u platforms detected\n", numPlatforms);
	for(unsigned int i = 0; i < numPlatforms; i++) {
		char buf[100];
		printf("Platform %u: \n", i);
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
								   sizeof(buf), buf, NULL);
		printf("\tVendor: %s\n", buf);
		status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
									sizeof(buf), buf, NULL);
		printf("\tName: %s\n", buf);
		
		if(status != CL_SUCCESS) {
			printf("clGetPlatformInfo failed\n");
			exit(-1);
		}
	}
	printf("\n");
	
	cl_uint numDevices = 0;
	cl_device_id *devices;
	
	// Retrive the number of devices present
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, 
							&numDevices);
	if(status != CL_SUCCESS) {
		printf("clGetDeviceIDs failed\n");
		exit(-1);
	}
	
	// Make sure some devices were found
	if(numDevices == 0) {
		printf("No devices detected.\n");
		exit(-1);
	}
	
	// Allocate enough space for each device
	devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
	if(devices == NULL) {
		perror("malloc");
		exit(-1);
	}
	
	// Fill in devices
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices,
							devices, NULL);
	if(status != CL_SUCCESS) {
		printf("clGetDeviceIDs failed\n");
		exit(-1);
	}   
	
	// Print out some basic information about each device
	printf("%u devices detected\n", numDevices);
	for(unsigned int i = 0; i < numDevices; i++) {
		char buf[100];
		printf("Device %u: \n", i);
		status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR,
								 sizeof(buf), buf, NULL);
		printf("\tDevice: %s\n", buf);
		status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
								  sizeof(buf), buf, NULL);
		printf("\tName: %s\n", buf);
		
		if(status != CL_SUCCESS) {
			printf("clGetDeviceInfo failed\n");
			exit(-1);
		}
	}
	printf("\n");
	
	cl_context context;
	
	// Create a context and associate it with the devices
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if(status != CL_SUCCESS || context == NULL) {
		printf("clCreateContext failed\n");
		exit(-1);
	}
	
	cl_command_queue cmdQueue;
	
	// Create a command queue and associate it with the device you 
	// want to execute on
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	if(status != CL_SUCCESS || cmdQueue == NULL) {
		printf("clCreateCommandQueue failed\n");
		exit(-1);
	}
	
	cl_mem d_A, d_B;  // Input buffers on device
	cl_mem d_C;       // Output buffer on device
	
	// Create a buffer object (d_A) that contains the data from the host ptr A
	d_A = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
						 datasize, A, &status);
	if(status != CL_SUCCESS || d_A == NULL) {
		printf("clCreateBuffer failed\n");
		exit(-1);
	}
	
	// Create a buffer object (d_B) that contains the data from the host ptr B
	d_B = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
						 datasize, B, &status);
	if(status != CL_SUCCESS || d_B == NULL) {
		printf("clCreateBuffer failed\n");
		exit(-1);
	}
	
	// Create a buffer object (d_C) with enough space to hold the output data
	d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, 
						 datasize, NULL, &status);
	if(status != CL_SUCCESS || d_C == NULL) {
		printf("clCreateBuffer failed\n");
		exit(-1);
	}
	
	
	cl_program program;
	
	char *source;
	const char *sourceFile = "VectorAdd.cl";
	// This function reads in the source code of the program
	source = readSource(sourceFile);
	
	//printf("Program source is:\n%s\n", source);
	
	// Create a program. The 'source' string is the code from the 
	// vectoradd.cl file.
	program = clCreateProgramWithSource(context, 1, (const char**)&source, 
										NULL, &status);
	if(status != CL_SUCCESS) {
		printf("clCreateProgramWithSource failed\n");
		exit(-1);
	}
	
	cl_int buildErr;
	// Build (compile & link) the program for the devices.
	// Save the return value in 'buildErr' (the following 
	// code will print any compilation errors to the screen)
	buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	
	// If there are build errors, print them to the screen
	if(buildErr != CL_SUCCESS) {
		printf("Program failed to build.\n");
		cl_build_status buildStatus;
		for(unsigned int i = 0; i < numDevices; i++) {
			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
								  sizeof(cl_build_status), &buildStatus, NULL);
			if(buildStatus == CL_SUCCESS) {
				continue;
			}
			
			char *buildLog;
			size_t buildLogSize;
			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
								  0, NULL, &buildLogSize);
			buildLog = (char*)malloc(buildLogSize);
			if(buildLog == NULL) {
				perror("malloc");
				exit(-1);
			}
			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
								  buildLogSize, buildLog, NULL);
			buildLog[buildLogSize-1] = '\0';
			printf("Device %u Build Log:\n%s\n", i, buildLog);   
			free(buildLog);
		}
		exit(0);
	}
	else {
		printf("No build errors\n");
	}
	
	
	cl_kernel kernel;
	
	// Create a kernel from the vector addition function (named "vecadd")
	kernel = clCreateKernel(program, "vecadd", &status);
	if(status != CL_SUCCESS) {
		printf("clCreateKernel failed\n");
		exit(-1);
	}
	
	// Associate the input and output buffers with the kernel 
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
	if(status != CL_SUCCESS) {
		printf("clSetKernelArg failed\n");
		exit(-1);
	}
	
	// Define an index space (global work size) of threads for execution.  
	// A workgroup size (local work size) is not required, but can be used.
	size_t globalWorkSize[1];  // There are ELEMENTS threads
	globalWorkSize[0] = ELEMENTS;
	
	// Execute the kernel.
	// 'globalWorkSize' is the 1D dimension of the work-items
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, 
									NULL, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		printf("clEnqueueNDRangeKernel failed\n");
		exit(-1);
	}
	
	// Read the OpenCL output buffer (d_C) to the host output array (C)
	clEnqueueReadBuffer(cmdQueue, d_C, CL_TRUE, 0, datasize, C, 
						0, NULL, NULL);
	
	// Verify correctness
	bool result = true;
	for(int i = 0; i < ELEMENTS; i++) {
		if(C[i] != i+i) {
			result = false;
			break;
		}
	}
	if(result) {
		printf("Output is correct\n");
	}
	else {
		printf("Output is incorrect\n");
	}
	
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	clReleaseContext(context);
	
	free(A);
	free(B);
	free(C);
	free(source);
	free(platforms);
	free(devices);
	
}

char* readSource(const char *sourceFilename) {
	
	FILE *fp;
	int err;
	int size;
	
	char *source;
	
	fp = fopen(sourceFilename, "rb");
	if(fp == NULL) {
		printf("Could not open kernel file: %s\n", sourceFilename);
		exit(-1);
	}
	
	err = fseek(fp, 0, SEEK_END);
	if(err != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	
	size = ftell(fp);
	if(size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}
	
	err = fseek(fp, 0, SEEK_SET);
	if(err != 0) {
		printf("Error seeking to start of file\n");
		exit(-1);
	}
	
	source = (char*)malloc(size+1);
	if(source == NULL) {
		printf("Error allocating %d bytes for the program source\n", size+1);
		exit(-1);
	}
	
	err = fread(source, 1, size, fp);
	if(err != size) {
		printf("only read %d bytes\n", err);
		exit(0);
	}
	
	source[size] = '\0';
	
	return source;
}



// //
// //  main.cpp
// //

// #include "Matrix.hpp"

// #include <cstdlib>
// #include <ctime>
// #include <iostream>
// #include <stdexcept>

// using namespace std;

// // Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
// void invertSequential(Matrix& iA) {

//     // vérifier que la matrice est carrée
//     assert(iA.rows() == iA.cols());
//     // construire la matrice [A I]
//     MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

//     // traiter chaque rangée
//     for (size_t k=0; k<iA.rows(); ++k) {
//         // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
//         // (pour une meilleure stabilité numérique).
//         size_t p = k;
//         double lMax = fabs(lAI(k,k));
//         for(size_t i = k; i < lAI.rows(); ++i) {
//             if(fabs(lAI(i,k)) > lMax) {
//                 lMax = fabs(lAI(i,k));
//                 p = i;
//             }
//         }
//         // vérifier que la matrice n'est pas singulière
//         if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

//         // échanger la ligne courante avec celle du pivot
//         if (p != k) lAI.swapRows(p, k);

//         double lValue = lAI(k, k);
//         for (size_t j=0; j<lAI.cols(); ++j) {
//             // On divise les éléments de la rangée k
//             // par la valeur du pivot.
//             // Ainsi, lAI(k,k) deviendra égal à 1.
//             lAI(k, j) /= lValue;
//         }

//         // Pour chaque rangée...
//         for (size_t i=0; i<lAI.rows(); ++i) {
//             if (i != k) { // ...différente de k
//                 // On soustrait la rangée k
//                 // multipliée par l'élément k de la rangée courante
//                 double lValue = lAI(i, k);
//                 lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
//             }
//         }
//     }

//     // On copie la partie droite de la matrice AI ainsi transformée
//     // dans la matrice courante (this).
//     for (unsigned int i=0; i<iA.rows(); ++i) {
//         iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
//     }
// }

// // Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
// void invertParallel(Matrix& iA) {
//     // vous devez coder cette fonction
// }

// // Multiplier deux matrices.
// Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

//     // vérifier la compatibilité des matrices
//     assert(iMat1.cols() == iMat2.rows());
//     // effectuer le produit matriciel
//     Matrix lRes(iMat1.rows(), iMat2.cols());
//     // traiter chaque rangée
//     for(size_t i=0; i < lRes.rows(); ++i) {
//         // traiter chaque colonne
//         for(size_t j=0; j < lRes.cols(); ++j) {
//             lRes(i,j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
//         }
//     }
//     return lRes;
// }

// int main(int argc, char** argv) {

//     srand((unsigned)time(NULL));

//     unsigned int lS = 5;
//     if (argc == 2) {
//         lS = atoi(argv[1]);
//     }

//     MatrixRandom lA(lS, lS);
//     cout << "Matrice random:\n" << lA.str() << endl;

//     Matrix lB(lA);
//     invertSequential(lB);
//     cout << "Matrice inverse:\n" << lB.str() << endl;

//     Matrix lRes = multiplyMatrix(lA, lB);
//     cout << "Produit des deux matrices:\n" << lRes.str() << endl;

//     cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;

//     return 0;
// }

