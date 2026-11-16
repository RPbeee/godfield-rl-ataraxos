// Test-only scenes, mirrored by the independent CPU _test_* fixtures.
extern "C" __global__ void gpu_search_fixture(void* storage,void** pointers,int n,int history_len,unsigned long long base_seed,bool canonical,void* heaps,unsigned arena_bytes,int kind,int model,bool fog,unsigned long long seed) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i)return;
    bind_arena(reinterpret_cast<GpuArena*>(heaps));auto& e=((Env*)storage)[0];auto o=outputs(pointers);
    if(kind==4) { // Same-observation counterfactual: erase the true hidden hand.
        const int seat=e.selection->actor_seat;const auto id=e.match.snapshot().state.players[seat].id;
        e.match=e.match.determinized(id,gs::span<const gf::CardId>{},seed);
        e.tokens.clear();e.selection.reset();e.encoded=false;e.waiting=false;drive(e,o,0,history_len,base_seed,canonical);return;
    }
    auto state=e.match.snapshot().state;
    for(auto& p:state.players){p.items.clear();p.curses.clear();p.guardian=gf::Guardian::none;}
    bool response=kind==1 || kind==3;
    if(kind==0 || kind==1) {
        int seat=response?1:0;state.players[seat].curses={gf::Curse::dream};
        state.players[seat].items={{200,static_cast<gf::CardId>(model),7,false},{201,190,{},false},{202,120,{},false}};
        state.players[1-seat].items={{100,6,{},false},{101,7,{},false}};
    } else if(kind==2) {
        state.players[0].items={{100,94,{},false},{101,232,{},false}};
        if(model)state.players[1].items={{200,209,{},false}};
    } else {
        state.players[0].items={{100,static_cast<gf::CardId>(model),{},false},{101,19,{},false}};
        state.players[1].items={{200,8,{},false},{201,120,{},false},{202,190,{},false}};
        if(fog)state.players[0].curses={gf::Curse::fog};
    }
    gf::Command launch{state.players[0].id,gf::CommandKind::play,{100},
        (response && gf::find_card(model)->hit_rate)?gs::nullopt:gs::optional<gf::PlayerId>{state.players[1].id}};
    if(kind==1)launch={state.players[0].id,gf::CommandKind::play,{100},state.players[1].id};
    if(kind==3)for(;;++seed){e.match.load_scene(state,state.players[0].id,seed);auto trial=e.match;trial.step(launch);if(trial.snapshot().phase==gf::MatchPhase::response)break;}
    e.match.load_scene(state,state.players[0].id,seed);
    for(auto& h:e.history)h=History(history_len);e.tokens.clear();e.selection.reset();e.encoded=false;e.waiting=false;
    drive(e,o,0,history_len,base_seed,canonical);
    if(response){before(e);e.match.step(launch);record(e,o,0,false);e.tokens.clear();e.selection.reset();e.encoded=false;drive(e,o,0,history_len,base_seed,canonical);}
}
