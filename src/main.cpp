#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace sycl;

int main(int argc, char *argv[]) {
  // Initialize SYCL queue with OpenMP backend
  queue q{cpu_selector{}, property::queue::in_order()};
  srand(time(0));

  int Ncell = 10000;
  int VertPerCell = 642;
  int size = Ncell * VertPerCell;
  double *array = (double *)malloc(size * sizeof(double));

  // Allocate memory in SYCL
  double *reuseableTempMemory = malloc_shared<double>(size, q);
  double *arrayDeviceView = malloc_shared<double>(size, q);

  // Initialize the array with random values
  for (int i = 0; i < size; ++i) {
    array[i] = double(rand() % 100) / 100.0;
  }

  // Copy data from host array to device memory
  q.memcpy(arrayDeviceView, array, size * sizeof(double)).wait();

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 2030; ++i) { // about 2k cells being deleted in HARVEY
    int nIndsToDelete = 642; // same as nVertsToDelete
    int startIndex = 0;      // same as startVertex
    int preNIndices = Ncell * VertPerCell; // same as totalNVertices
    int nDataToMove = preNIndices - (startIndex + nIndsToDelete);

    double *dstPtr = arrayDeviceView + startIndex;
    double *srcPtr = arrayDeviceView + startIndex + nIndsToDelete;
    double *tmpBufferPtr = reuseableTempMemory;

    // Copy data using SYCL queue
    q.memcpy(tmpBufferPtr, srcPtr, nDataToMove * sizeof(double)).wait();
    q.memcpy(dstPtr, tmpBufferPtr, nDataToMove * sizeof(double)).wait();
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  double timeInSeconds = duration.count();

  std::cout << "Runtime (sec): " << timeInSeconds << std::endl;

  // Free allocated memory
  free(array);
  free(reuseableTempMemory, q);
  free(arrayDeviceView, q);

  return 0;
}
