// Compiled only by test_gpu.py; never included in the training module.
extern "C" __global__ void gpu_test_counter(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    auto& e=((Env*)storage)[i];
    const_cast<gf::MatchSnapshot&>(e.match.snapshot()).decisions=511;
    e.tokens.clear();e.selection.reset();e.encoded=false;
    drive(e,outputs(pointers),i,history_len,seed,canonical);
}
