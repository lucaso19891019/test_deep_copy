#include <omp.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

int main(int argc, char *argv[]) {
  srand(time(0));

  int Ncell = 10000;
  int VertPerCell = 642;
  int size = Ncell * VertPerCell;
  double *array = (double *)malloc(size * sizeof(double));
  double *reuseableTempMemory = (double *)malloc(size * sizeof(double));
  double *arrayDeviceView = (double *)malloc(size * sizeof(double));

  // Initialize the array with random values
  for (int i = 0; i < size; ++i) {
    array[i] = double(rand() % 100) / 100.0;
  }

  // Copy data from host array to device memory
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    arrayDeviceView[i] = array[i];
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 2030; ++i) { // about 2k cells being deleted in HARVEY
    int nIndsToDelete = 642; // same as nVertsToDelete
    int startIndex = 0;      // same as startVertex
    int preNIndices = Ncell * VertPerCell; // same as totalNVertices
    int nDataToMove = preNIndices - (startIndex + nIndsToDelete);

    double *dstPtr = arrayDeviceView + startIndex;
    double *srcPtr = arrayDeviceView + startIndex + nIndsToDelete;
    double *tmpBufferPtr = reuseableTempMemory;

    // Copy data using OpenMP parallel for
    #pragma omp parallel for
    for (int j = 0; j < nDataToMove; ++j) {
      tmpBufferPtr[j] = srcPtr[j];
    }

    #pragma omp parallel for
    for (int j = 0; j < nDataToMove; ++j) {
      dstPtr[j] = tmpBufferPtr[j];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  double timeInSeconds = duration.count();

  std::cout << "Runtime (sec): " << timeInSeconds << std::endl;

  // Free allocated memory
  free(array);
  free(reuseableTempMemory);
  free(arrayDeviceView);

  return 0;
}
