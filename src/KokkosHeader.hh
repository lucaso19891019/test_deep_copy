#ifndef KOKKOS_HEADER_HH
#define KOKKOS_HEADER_HH

#include <math.h>
#include <stdio.h>
#include <Kokkos_Core.hpp>

#define _USE_MATH_DEFINES

#ifdef USE_KOKKOS_CUDA
#define MyBackend Kokkos::Cuda
#define MyDeviceMemorySpace Kokkos::CudaSpace
#define MyHostMemorySpace Kokkos::CudaHostPinnedSpace
#define MyUSpace Kokkos::CudaUVMSpace

//#elifdef USE_KOKKOS_SYCL
//#define MyBackend Kokkos::Experimental::SYCL
//#define MyDeviceMemorySpace Kokkos::Experimental::SYCLDeviceUSMSpace
//#define MyDeviceMemorySpace Kokkos::Experimental::SYCLSharedUSMSpace
//#define MyHostMemorySpace Kokkos::Experimental::SYCLHostUSMSpace
//#define MyUSpace Kokkos::Experimental::SYCLSharedUSMSpace
#elifdef USE_KOKKOS_SYCL
#define MyBackend Kokkos::OpenMP
#define MyDeviceMemorySpace Kokkos::HostSpace
//#define MyDeviceMemorySpace Kokkos::Experimental::SYCLSharedUSMSpace
#define MyHostMemorySpace Kokkos::HostSpace
#define MyUSpace Kokkos::HostSpace

#elifdef USE_KOKKOS_OPENMPTARGET

#elifdef USE_KOKKOS_OPENACC
#define MyBackend Kokkos::Experimental::OpenACC
#define MyDeviceMemorySpace Kokkos::Experimental::OpenACCSpace
#define MyHostMemorySpace Kokkos::HostSpace
#define MyDeviceExecSpace MyDeviceMemorySpace::execution_space
#define MyPolicy Kokkos::RangePolicy<MyDeviceExecSpace>
#define My2DPolicy Kokkos::MDRangePolicy<MyDeviceExecSpace,Kokkos::Rank<2>>
#define MyUSpace Kokkos::Experimental::OpenACCSpace

#elifdef USE_KOKKOS_HIP
#define MyBackend Kokkos::HIP
#define MyDeviceMemorySpace Kokkos::HIPSpace
#define MyHostMemorySpace Kokkos::HIPHostPinnedSpace
#define MyUSpace Kokkos::HIPManagedSpace

#endif


#define MyDeviceExecSpace MyDeviceMemorySpace::execution_space
#define MyPolicy Kokkos::RangePolicy<MyDeviceExecSpace>
#define MyTeamPolicy Kokkos::TeamPolicy<MyDeviceExecSpace>
#define My2DPolicy Kokkos::MDRangePolicy<MyDeviceExecSpace,Kokkos::Rank<2>>

#define MyTeamPolicy Kokkos::TeamPolicy<MyDeviceExecSpace>
#define ThreadType MyTeamPolicy::member_type
#define SharedType(Type) Kokkos::View<Type*, MyTeamPolicy::execution_space::scratch_memory_space>

#define MyView(Type) Kokkos::View<Type *, MyDeviceMemorySpace>
#define MyUnmanagedHostView(Type) Kokkos::View<Type *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
#define MyUnmanagedView(Type) Kokkos::View<Type *, MyDeviceMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
#define MyConstView(Type) Kokkos::View<const Type *, MyDeviceMemorySpace>
#define MyUView(Type) Kokkos::View<Type *, MyUSpace>
#define MyHostView(Type) Kokkos::View<Type *, MyHostMemorySpace>

#define MyDeviceArray(Type, Size) static_cast<Type*>(Kokkos::kokkos_malloc<MyDeviceMemorySpace >( Size * sizeof( Type )))

namespace kokkos_thrust{
template<typename T>
struct MyMin{
public:
  T val;
  MyMin(){val=Kokkos::reduction_identity<T>::min();}
  bool operator()(const T& a, const T& b)const{ return a>b;}
};

template<typename T>
struct MyMax{
public:
  T val;
  MyMax(){val=Kokkos::reduction_identity<T>::max();}
  bool operator()(const T& a, const T& b)const{ return a<b;}
};

template<typename T>
struct Updater{
  T reduce;
  size_t ind;
};

template<typename ValueType, typename OutputType, template<typename>typename Operator>
class ReduceByIndex{
public:
  using execution_space = MyBackend;
  using value_type = Updater<ValueType>;
  using size_type = size_t;

  ReduceByIndex (const size_t* index, size_type keyLength, const ValueType* value, OutputType* output): index_(index),keyLength_(keyLength),value_(value),output_(output) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const size_type i, value_type& update, const bool is_final) const{
    update.ind=index_[i];
    if(op(update.reduce,value_[i])) update.reduce=value_[i];
     
    if (is_final){
      if(i==keyLength_-1) output_[index_[i]]=update.reduce;
      else if(index_[i]!=index_[i+1])
        output_[index_[i]]=update.reduce;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join (value_type& dst, const value_type& src) const{
    if(dst.ind==src.ind){
      if (op(dst.reduce , src.reduce)) {
	dst.reduce = src.reduce;
      }
    }    
  }

  KOKKOS_INLINE_FUNCTION
  void init (value_type& dst) const{ dst.reduce = op.val;}

  private:
  const size_t* index_;
  const ValueType* value_;
  size_type keyLength_;
  OutputType* output_;
  Operator<ValueType> op;
};

 
template<typename ValueType, typename OutputType, template<typename>typename Operator>
void reduceByKey(size_t* key, size_t keyLength, ValueType* value, OutputType* output){
  MyView(size_t) index("index",keyLength);
  Kokkos::parallel_scan("get_index",MyPolicy(0,keyLength),KOKKOS_LAMBDA(const size_t i, size_t& ind, const bool is_final){
    unsigned short offset;
    offset=1;
    if(i==0) offset=0;
    else if(key[i]==key[i-1]) offset=0;
    ind+=offset;
    if (is_final) index(i)=ind;
  });
  Kokkos::parallel_scan("ReduceByIndex",MyPolicy(0,keyLength),ReduceByIndex<ValueType,OutputType,Operator>(index.data(),keyLength,value,output));
}
}

template<typename T>
class MyConstDeviceView{
  private:
  MyView(const T) _data;
  public:
  void allocate(const std::string& label, const T* ptr, const int size){
    MyView(T) _data_d(label,size);
    MyUnmanagedHostView(const T) _data_h(ptr,size);
    Kokkos::deep_copy(_data_d,_data_h);
    _data=_data_d;
  }
  void deallocate(){_data=MyView(const T)();}
  const T* getPtr(){return _data.data();}
};

KOKKOS_INLINE_FUNCTION
void sincospi(double x, double *sin_, double *cos_){
    *sin_=sin(x*M_PI);
    *cos_=cos(x*M_PI);
}

#endif
