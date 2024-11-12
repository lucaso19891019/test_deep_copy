#include<Kokkos_Core.hpp>
#include<cstdio>
#include "KokkosHeader.hh"

#include <ctime>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
   srand(time(0));
   int Ncell=10000;
   int VertPerCell=642;
   int size=Ncell*VertPerCell;
   double *array = (double*)malloc(size*sizeof(double));

   for (int i = 0; i < size; ++i) {
        array[i] = double(rand() % 100)/100.0;
   }

   MyUnmanagedHostView(double) arrayView(array,size);

   MyView(double) arrayDeviceView("arrayDeviceView",size);

   Kokkos::deep_copy(arrayDeviceView,arrayView);


   int *cellDel=(int*)malloc(100*sizeof(int));
   for (int i=0; i<100; ++i){
	cellDel[i] = rand() % Ncell + 1;
   }

   for (int i=0; i<100; ++i){
	int tmpSize=(Ncell-cellDel[i])*VertPerCell;
	MyView(double) tmp("tmp",tmpSize);
	MyUnmanagedView(double) src(arrayDeviceView.data()+cellDel[i]*VertPerCell,tmpSize);
	MyUnmanagedView(double) dst(arrayDeviceView.data()+(cellDel[i]-1)*VertPerCell,tmpSize);
	Kokkos::deep_copy(tmp,src);
	Kokkos::deep_copy(dst,tmp);
   }

  }
  Kokkos::finalize();
}
