// CUDA-only storage adapters for the training engine. No CPU rule changes.
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/optional>
#include <cuda/std/variant>
#include <cuda/std/tuple>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>
#include <cuda/std/string_view>
#include <cuda/std/cmath>
#include <cuda/std/memory>


[[noreturn]] __device__ void gpu_fail(int line) {
    printf("GPU engine invariant failed at generated source line %d\n", line);
    __trap();
}
// Each lane owns its arena. The current arena pointer lives in per-block
// shared memory, so concurrent streams/environments do not share allocator state.
struct GpuPool { unsigned long long used,capacity; };
struct GpuArena {
    unsigned offset,capacity;
    unsigned long long bins[24];
    GpuPool* pool=nullptr;
    void initialize(unsigned bytes) {
        offset=(sizeof(GpuArena)+15u)&~15u;capacity=bytes;pool=nullptr;
        for(auto& b:bins)b=0;
    }
};
#ifdef __CUDACC_RTC__
extern __shared__ unsigned char gpu_arena_context[];
void bind_arena(GpuArena* p) {reinterpret_cast<GpuArena**>(gpu_arena_context)[threadIdx.x]=p;}
GpuArena* current_arena() {return reinterpret_cast<GpuArena**>(gpu_arena_context)[threadIdx.x];}
void* engine_allocate(unsigned long long bytes) {
    auto* a=current_arena();unsigned bucket=5,block=32;
    while(block<bytes+16){block*=2;++bucket;if(bucket>=24)gpu_fail(__LINE__);}
    unsigned long long address=a->bins[bucket];
    if(address)a->bins[bucket]=reinterpret_cast<unsigned long long*>(address)[0];
    else {
        if(a->pool) {
            const auto at=atomicAdd(&a->pool->used,static_cast<unsigned long long>(block));
            if(at+block>a->pool->capacity)gpu_fail(__LINE__);
            address=reinterpret_cast<unsigned long long>(a->pool+1)+at;
        } else {
            const auto at=a->offset;if(block>a->capacity-at)gpu_fail(__LINE__);
            address=reinterpret_cast<unsigned long long>(a)+at;
        }
        a->offset+=block;
    }
    auto* p=reinterpret_cast<unsigned long long*>(address);p[1]=bucket;
    return reinterpret_cast<char*>(p)+16;
}
void engine_free(void* ptr) {
    if(!ptr)return;auto* a=current_arena();auto* p=reinterpret_cast<unsigned long long*>(static_cast<char*>(ptr)-16);
    const unsigned bucket=p[1];p[0]=a->bins[bucket];a->bins[bucket]=reinterpret_cast<unsigned long long>(p);

}
#else
void bind_arena(GpuArena*) {}
void* engine_allocate(unsigned long long bytes) {return malloc(bytes);}
void engine_free(void* p) {free(p);}
#endif
namespace gs {
using namespace cuda::std;
using string = cuda::std::string_view;

template<class T> class vector {
    T* ptr_=nullptr;
    size_t size_=0, capacity_=0;
public:
    using value_type=T;
    using iterator=T*;
    using const_iterator=const T*;
    vector() = default;
    explicit vector(size_t n) { resize(n); }
    vector(size_t n, const T& value) { assign(n,value); }
    vector(initializer_list<T> values) { for(const auto& v:values)push_back(v); }
    template<class It> vector(It b, It e) { for(;b!=e;++b)push_back(*b); }
    vector(const vector& v) { reserve(v.size_);for(const auto& x:v)push_back(x); }
    vector(vector&& v) noexcept : ptr_(v.ptr_),size_(v.size_),capacity_(v.capacity_) {
        v.ptr_=nullptr;v.size_=v.capacity_=0;
    }
    ~vector() { clear();engine_free(ptr_); }
    vector& operator=(const vector& v) {
        if(this!=&v) {clear();reserve(v.size_);for(const auto& x:v)push_back(x);}return *this;
    }
    vector& operator=(vector&& v) noexcept {
        if(this!=&v) {clear();engine_free(ptr_);ptr_=v.ptr_;size_=v.size_;capacity_=v.capacity_;
            v.ptr_=nullptr;v.size_=v.capacity_=0;}return *this;
    }
    vector& operator=(initializer_list<T> v) { clear();for(const auto& x:v)push_back(x);return *this; }
    T* begin() {return ptr_;} const T* begin() const {return ptr_;}
    T* end() {return ptr_+size_;} const T* end() const {return ptr_+size_;}
    T* data() {return ptr_;} const T* data() const {return ptr_;}
    size_t size() const {return size_;} bool empty() const {return !size_;}
    T& operator[](size_t i) {return ptr_[i];} const T& operator[](size_t i) const {return ptr_[i];}
    T& at(size_t i) {if(i>=size_)gpu_fail(__LINE__);return ptr_[i];}
    const T& at(size_t i) const {if(i>=size_)gpu_fail(__LINE__);return ptr_[i];}
    T& front() {return ptr_[0];} const T& front() const {return ptr_[0];}
    T& back() {return ptr_[size_-1];} const T& back() const {return ptr_[size_-1];}
    void clear() {while(size_)ptr_[--size_].~T();}
    void reserve(size_t n) {
        if(n<=capacity_)return;
        size_t cap=capacity_ ? capacity_*2 : 4;if(cap<n)cap=n;
        T* p=static_cast<T*>(engine_allocate(cap*sizeof(T)));if(!p)gpu_fail(__LINE__);
        for(size_t i=0;i<size_;++i){new(p+i)T(gs::move(ptr_[i]));ptr_[i].~T();}
        engine_free(ptr_);ptr_=p;capacity_=cap;
    }
    void resize(size_t n) {reserve(n);while(size_>n)ptr_[--size_].~T();while(size_<n)new(ptr_+size_++)T();}
    void assign(size_t n,const T& v) {clear();reserve(n);while(size_<n)push_back(v);}
    template<class It> void assign(It b,It e) {clear();for(;b!=e;++b)push_back(*b);}
    void push_back(const T& v) {reserve(size_+1);new(ptr_+size_++)T(v);}
    void push_back(T&& v) {reserve(size_+1);new(ptr_+size_++)T(gs::move(v));}
    template<class... A> T& emplace_back(A&&... a) {reserve(size_+1);new(ptr_+size_++)T(gs::forward<A>(a)...);return back();}
    void pop_back() {ptr_[--size_].~T();}
    iterator erase(const_iterator b,const_iterator e) {
        size_t pos=b-ptr_,n=e-b;
        for(size_t i=pos;i+n<size_;++i)ptr_[i]=gs::move(ptr_[i+n]);
        for(size_t i=0;i<n;++i)pop_back();return ptr_+pos;
    }
    iterator erase(const_iterator p) {return erase(p,p+1);}
    iterator insert(const_iterator pos,const T& v) {
        size_t i=pos-ptr_;reserve(size_+1);
        if(i==size_){push_back(v);return ptr_+i;}
        new(ptr_+size_)T(gs::move(back()));++size_;
        for(size_t j=size_-2;j>i;--j)ptr_[j]=gs::move(ptr_[j-1]);ptr_[i]=v;return ptr_+i;
    }
    template<class It> iterator insert(const_iterator pos,It b,It e) {
        size_t i=pos-ptr_,start=i;for(;b!=e;++b)insert(ptr_+i++,*b);return ptr_+start;
    }
    bool operator==(const vector& v) const {if(size_!=v.size_)return false;for(size_t i=0;i<size_;++i)if(!(ptr_[i]==v[i]))return false;return true;}
    bool operator!=(const vector& v) const {return !(*this==v);}
    bool operator<(const vector& v) const {size_t n=size_<v.size_?size_:v.size_;for(size_t i=0;i<n;++i){if(ptr_[i]<v[i])return true;if(v[i]<ptr_[i])return false;}return size_<v.size_;}
};
// History uses a ring; evicting its oldest event never shifts the remaining
// tokens. Physical storage is reused across episode resets.
template<class T> class deque {
    vector<T> storage_;
    size_t start_=0,count_=0;
public:
    struct const_iterator {
        const deque* owner;size_t position;
        const T& operator*()const{return owner->storage_[(owner->start_+position)%owner->storage_.size()];}
        const T* operator->()const{return &**this;}
        const_iterator& operator++(){++position;return *this;}
        bool operator!=(const const_iterator& o)const{return position!=o.position;}
    };
    size_t size()const{return count_;}
    const T& operator[](size_t i)const{return storage_[(start_+i)%storage_.size()];}
    void clear(){start_=count_=0;}
    void pop_front(){start_=(start_+1)%storage_.size();--count_;}
    void push_back(T v){
        if(count_==storage_.size()) {
            if(start_!=0)gpu_fail(__LINE__);storage_.push_back(gs::move(v));
        } else storage_[(start_+count_)%storage_.size()]=gs::move(v);
        ++count_;
    }
    const_iterator begin()const{return {this,0};}
    const_iterator end()const{return {this,count_};}
};
template<class T> struct unordered_set {
    vector<T> values;
    pair<T*,bool> insert(const T& v) {for(auto& x:values)if(x==v)return {&x,false};values.push_back(v);return {&values.back(),true};}
    bool contains(const T& v) const {for(const auto& x:values)if(x==v)return true;return false;}
};
template<class T> struct shared_ptr {
    T* p=nullptr;
    shared_ptr()=default;
    T& operator*()const{return *p;}T* operator->()const{return p;}
    explicit operator bool()const{return p!=nullptr;}
};
template<class It,class P> It find_if(It b,It e,P p){for(;b!=e;++b)if(p(*b))break;return b;}
template<class It,class T> It find(It b,It e,const T& v){return gs::find_if(b,e,[&](const auto& x){return x==v;});}
template<class It,class P> bool any_of(It b,It e,P p){return gs::find_if(b,e,p)!=e;}
template<class It,class P> bool none_of(It b,It e,P p){return !gs::any_of(b,e,p);}
template<class It,class P> bool all_of(It b,It e,P p){for(;b!=e;++b)if(!p(*b))return false;return true;}
template<class It,class P> size_t count_if(It b,It e,P p){size_t n=0;for(;b!=e;++b)n+=p(*b);return n;}
template<class It,class T> size_t count(It b,It e,const T& v){return gs::count_if(b,e,[&](const auto& x){return x==v;});}
template<class It,class T> void fill(It b,It e,const T& v){for(;b!=e;++b)*b=v;}
template<class It> It next(It b){return b+1;}
template<class It> It adjacent_find(It b,It e){if(b==e)return e;for(auto n=b+1;n!=e;++b,++n)if(*b==*n)return b;return e;}
template<class It> bool is_sorted(It b,It e){if(b==e)return true;for(auto n=b+1;n!=e;++b,++n)if(*n<*b)return false;return true;}
template<class It,class T> It lower_bound(It b,It e,const T& v){for(;b!=e && *b<v;++b){}return b;}
template<class It,class P> void stable_sort(It b,It e,P p){if(b==e)return;for(It i=b+1;i!=e;++i){auto v=gs::move(*i);auto j=i;while(j!=b && p(v,*(j-1))){*j=gs::move(*(j-1));--j;}*j=gs::move(v);}}
template<class It,class P> void sort(It b,It e,P p){gs::stable_sort(b,e,p);}
template<class It> void sort(It b,It e){gs::stable_sort(b,e,[](const auto& a,const auto& c){return a<c;});}
template<class T,class P> size_t erase_if(vector<T>& v,P p){size_t n=0;for(size_t i=0;i<v.size();){if(p(v[i])){v.erase(v.begin()+i);++n;}else ++i;}return n;}
template<class T> T min(T a,T b){return a<b?a:b;}
template<class T> T max(T a,T b){return a>b?a:b;}
template<class T> T clamp(T v,T a,T b){return gs::min(gs::max(v,a),b);}
}
