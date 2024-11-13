#include <Kokkos_Core.hpp>
#include<cstdio>
#include <chrono>
#include <ctime>
#include <iostream>


#define MyBackend Kokkos::OpenMP
#define MyDeviceMemorySpace Kokkos::HostSpace
#define MyHostMemorySpace Kokkos::HostSpace
#define MyUSpace Kokkos::HostSpace

#define MyView(Type) Kokkos::View<Type *, MyDeviceMemorySpace>
#define MyUnmanagedHostView(Type) Kokkos::View<Type *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
#define MyUnmanagedView(Type) Kokkos::View<Type *, MyDeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>



int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
   srand(time(0));
   int Ncell=10000;
   int VertPerCell=642;
   int size=Ncell*VertPerCell;
   double *array = (double*)malloc(size*sizeof(double));
   MyView(double) reuseableTempMemory("reusableTempMemory",size);
   for (int i = 0; i < size; ++i) {
        array[i] = double(rand() % 100)/100.0;
   }

   MyUnmanagedHostView(double) arrayView(array,size);

   MyView(double) arrayDeviceView("arrayDeviceView",size);

   Kokkos::deep_copy(arrayDeviceView,arrayView);


//   int *cellDel=(int*)malloc(100*sizeof(int));
//   for (int i=0; i<100; ++i){
//	cellDel[i] = rand() % Ncell + 1;
//   }
   
   auto start = std::chrono::high_resolution_clock::now();   
   for (int i=0; i<2030; ++i){//about 2k cells being deleted in HARVEY
        int nIndsToDelete = 642; //same as nVertsToDelete
        int startIndex = 0; // same as startVertex
        int preNIndices = Ncell*VertPerCell; //same as totalNVertices
        int nDataToMove= preNIndices - (startIndex + nIndsToDelete);//10000000;//(Ncell-cellDel[i])*VertPerCell;
	double* dstPtr = arrayDeviceView.data()+startIndex;
	double* srcPtr = arrayDeviceView.data()+startIndex + nIndsToDelete;
	double* tmpBufferPtr = reuseableTempMemory.data();
        MyUnmanagedView(double) tmpBufferPtrView((double*)tmpBufferPtr,nDataToMove);
        MyUnmanagedView(double) dstPtrView(dstPtr,nDataToMove);
        MyUnmanagedView(double) srcPtrView(srcPtr,nDataToMove);
        Kokkos::deep_copy(tmpBufferPtrView,srcPtrView);
        Kokkos::deep_copy(dstPtrView,tmpBufferPtrView);
	}
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> duration = end - start;
   double timeInSeconds = duration.count();

   std::cout << "Runtime (sec): " << timeInSeconds << std::endl;
   free(array);
  }
  Kokkos::finalize();
}
