// Search entry points. No network sees private storage: inference reads Outputs only.
namespace gpu_training {
bool unused(const Env& e,int seat,const int* assignment,gs::array<int,297>& counts) {
    unsigned long long total=0;const auto& m=e.match;
    for(int c=0;c<297;++c){if(assignment[c]<0)return false;counts[c]=assignment[c];total+=assignment[c];}
    if(counts[0] || total>m.snapshot().config.hand_limit)return false;
    for(const auto& card:gf::catalog())if(counts[card_code(card.id)] && !m.gift_supported(card.id))return false;
    int used=0;
    for(const auto& item:m.snapshot().state.players[1-seat].items)if(item.used){++used;auto& c=counts[card_code(item.model)];c=gs::max(0,c-1);}
    int retained=used;const auto& snap=m.snapshot();
    if(snap.phase==gf::MatchPhase::response && snap.batch) {
        const auto& b=*snap.batch;const auto& opponent=snap.state.players[1-seat];retained=0;
        for(const auto& item:b.initial_state.players[1-seat].items) {
            bool selected=b.attack.actor==opponent.id && gs::find(b.attack.items.begin(),b.attack.items.end(),item.token)!=b.attack.items.end();
            for(const auto& command:b.responses)if(command.actor==opponent.id && gs::find(command.items.begin(),command.items.end(),item.token)!=command.items.end())selected=true;
            retained+=item.used || selected;
        }
    }
    unsigned long long physical=retained;for(auto c:counts)physical+=c;
    return physical<=snap.config.hand_limit;
}
int support(const Env& e) {
    return e.waiting || !e.selection || e.selection->substep!=0 ? -1 :
        int(e.match.determinize_support(e.match.snapshot().state.players[e.selection->actor_seat].id,true));
}
void clear_world(Env& e,Outputs o,int i,int k) {
    stop(e,o,i);e.tokens.clear();e.dirty=false;e.encoded=false;e.status=2;e.searcher=-1;
    o.decisions[i]=0;o.decision_done[i]=false;o.episode_done[i]=true;o.truncated[i]=false;
    o.outcome[2*i]=o.outcome[2*i+1]=0;const int t=history_offset+k;
    for(int j=0;j<t;++j){o.obs_type[i*t+j]=0;o.obs_valid[i*t+j]=false;o.obs_aux_card[i*t+j]=0;}
    for(int j=0;j<t*8;++j)o.obs_cards[i*t*8+j]=0;
    for(int j=0;j<t*C;++j)o.obs_cat[i*t*C+j]=0;
    for(int j=0;j<t*F;++j)o.obs_num[i*t*F+j]=0;
    for(int j=0;j<297;++j)o.label_opp_hand[i*297+j]=0;o.label_opp_hand_count[i]=0;
    for(int j=0;j<297;++j)o.label_opp_unused_hand[i*297+j]=0;o.label_opp_unused_hand_count[i]=0;
}
}
extern "C" __global__ void gpu_search_init(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,void* pool,bool search,int depth,long long games,unsigned long long eval_seed) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto* arena=reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes));
    arena->initialize(arena_bytes);arena->pool=(GpuPool*)pool;bind_arena(arena);
    auto& e=*new((Env*)storage+i)Env();e.original_index=i;e.game=i;e.games=games;e.eval_seed=eval_seed;e.parallelism=n;
    e.search=search;e.tracking=!search;e.max_depth=depth;for(auto& h:e.history)h=History(history_len);
    auto o=outputs(pointers);o.ready[i]=127;
    if(search)clear_world(e,o,i,history_len);
    else {reset(e,o,i,seed);drive(e,o,i,history_len,seed,canonical);
        o.decision_done[i]=o.episode_done[i]=o.truncated[i]=false;o.outcome[2*i]=o.outcome[2*i+1]=0;}
}
extern "C" __global__ void gpu_search_support(void* storage,const int* rows,int n,int* out) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=support(((Env*)storage)[rows[i]]);
}
extern "C" __global__ void gpu_search_unused(void* storage,const int* rows,const int* assignments,int n,bool* valid) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& root=((Env*)storage)[rows[i]];gs::array<int,297> counts;
    valid[i]=support(root)==0 && unused(root,root.selection->actor_seat,assignments+i*297,counts);
}
extern "C" __global__ void gpu_search_load(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,const void* source,void** source_pointers,const int* rows,const int* assignments,const unsigned long long* seeds,int* statuses,bool lean) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    auto& e=((Env*)storage)[i];const auto& root=((const Env*)source)[rows[i]];auto o=outputs(pointers);
    clear_world(e,o,i,history_len);o.episode_id[i]=outputs(source_pointers).episode_id[rows[i]];statuses[i]=2;
    if(support(root)!=0)return;
    const int seat=root.selection->actor_seat;const auto id=root.match.snapshot().state.players[seat].id;
    gs::array<int,297> counts;statuses[i]=1;if(!unused(root,seat,assignments+i*297,counts))return;
    gs::vector<gf::CardId> models,own;
    for(const auto& card:gf::catalog())for(int j=0;j<counts[card_code(card.id)];++j)models.push_back(card.id);
    auto rng=mix(seeds[i]^0x36d8a971c2ef504bULL);
    for(const auto& item:root.match.view(id).board.hand)if(item.disguised) {
        gs::vector<gf::CardId> tickets;
        for(const auto& card:gf::catalog())if(root.match.gift_supported(card.id) && gf::match_dream_compatible(card.id,item.face))
            for(int j=0;j<card.gift_rate;++j)tickets.push_back(card.id);
        if(tickets.empty() && root.match.gift_supported(item.face))tickets.push_back(item.face);
        if(tickets.empty()){statuses[i]=3;return;}
        const auto bound=gs::uint64_t(tickets.size()),threshold=-bound%bound;
        do{rng=mix(rng);}while(rng<threshold);own.push_back(tickets[rng%bound]);
    }
    e.match=own.empty()?root.match.determinized(id,models,seeds[i],true):root.match.determinized(id,models,own,seeds[i],true);
    if(e.match.gpu_search_status){statuses[i]=e.match.gpu_search_status;return;}
    const_cast<gf::MatchSnapshot&>(e.match.snapshot()).gpu_lean=lean;
    e.history[seat]=root.history[seat];e.history[1-seat]=root.history[seat].projected();
    e.chance_rng=mix(seeds[i]^0xa4093822299f31d0ULL);e.encoded=false;e.waiting=false;e.selection.reset();
    e.searcher=seat;e.status=0;statuses[i]=0;o.episode_done[i]=false;drive(e,o,i,history_len,seed,canonical);
}
extern "C" __global__ void gpu_search_actors(void* storage,int n,int* actors,int* searchers,int* result) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& e=((Env*)storage)[i];const auto& s=e.match.snapshot();actors[i]=-1;searchers[i]=e.searcher;result[i]=0;
    if(e.status==0) {
        if(!e.match.done()){for(int j=0;j<2;++j)if(s.actor && s.state.players[j].id==*s.actor)actors[i]=j;}
        else result[i]=s.truncated?0:s.rewards[e.searcher];
    }
}
// Leaf observations use independent storage and never overwrite the policy outputs.
extern "C" __global__ void gpu_search_observe(void* storage,void** pointers,int n,int history_len,void* heaps,unsigned arena_bytes,const int* rows,const int* seats) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int row=rows[i];bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+row*static_cast<unsigned long long>(arena_bytes)));
    const auto& e=((Env*)storage)[row];const auto& players=e.match.snapshot().state.players;const int seat=seats[i];
    gs::vector<gf::ItemToken> order;for(const auto& item:players[seat].items)order.push_back(item.token);
    auto v=e.match.view(players[seat].id);gs::vector<Token> ts;encode(v,players[seat].id,order,ts);ts.resize(history_offset+history_len);
    int j=history_offset;for(const auto& event:e.history[seat].events()){ts[j]=event.token;ts[j++].num[Age]=history_len?float(v.decisions-event.decision)/history_len:0;}
    const int t=history_offset+history_len;auto o=outputs(pointers);
    for(int k=0;k<t;++k){const auto& x=ts[k];int at=i*t+k;o.obs_type[at]=x.type;o.obs_aux_card[at]=x.aux;o.obs_valid[at]=x.type!=0;
        for(int c=0;c<C;++c)o.obs_cat[at*C+c]=x.cat[c];for(int c=0;c<F;++c)o.obs_num[at*F+c]=x.num[c];for(int c=0;c<8;++c)o.obs_cards[at*8+c]=x.cards[c];}
}
// Fork an already validated, untouched root world. No belief draw, replay, or
// real opponent data is consulted; histories/RNG are exactly the base world's.
extern "C" __global__ void gpu_search_clone(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,const void* source,void** source_pointers,const int* indices,int* statuses,int actual) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    auto& e=((Env*)storage)[i];const auto& base=((const Env*)source)[indices[i]];auto o=outputs(pointers);
    clear_world(e,o,i,history_len);statuses[i]=2;if(i>=actual)return;
    o.episode_id[i]=outputs(source_pointers).episode_id[indices[i]];
    statuses[i]=base.status;
    if(base.status!=0)return;
    e.match=base.match;e.history[0]=base.history[0];e.history[1]=base.history[1];e.chance_rng=base.chance_rng;
    e.searcher=base.searcher;e.status=0;e.waiting=false;e.selection.reset();o.episode_done[i]=false;
    drive(e,o,i,history_len,seed,canonical);
}
namespace gpu_training {
struct CandidateCursor { gs::optional<Selection> selection;int base=0; };
void candidate_publish(const Selection& s,Outputs o,int i) {
    o.stage[i]=int(s.stage);o.actor_seat[i]=s.actor_seat;
    for(int k=0;k<21;++k)o.mask[21*i+k]=s.mask[k];
    o.chosen_kind[i]=s.kind_chosen?int(s.command.kind):-1;
    o.chosen_target[i]=s.stage>=Stage::alloc_hp?(s.command.target?(*s.command.target==s.command.actor?1:2):0):-1;
    o.chosen_alloc_hp[i]=s.stage==Stage::alloc_mp?(s.command.allocation->hp==99?20:s.command.allocation->hp/5):-1;
    o.chosen_count[i]=s.count;o.substep[i]=s.substep;
    for(int k=0;k<18;++k)o.chosen_cards[18*i+k]=s.cards[k];
}
void candidate_stop(Outputs o,int i) {
    o.stage[i]=o.actor_seat[i]=o.chosen_kind[i]=o.chosen_target[i]=o.chosen_alloc_hp[i]=-1;
    o.chosen_count[i]=0;o.substep[i]=0;o.episode_done[i]=true;
    for(int k=0;k<21;++k)o.mask[21*i+k]=false;for(int k=0;k<18;++k)o.chosen_cards[18*i+k]=-1;
}
}
extern "C" __global__ void gpu_candidate_sizes(unsigned long long* size){*size=sizeof(gpu_training::CandidateCursor);}
extern "C" __global__ void gpu_candidate_init(void* storage,void** pointers,int n,const void* bases,const int* indices,void* heaps,unsigned arena_bytes) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto* arena=reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes));arena->initialize(arena_bytes);bind_arena(arena);
    auto& c=*new((CandidateCursor*)storage+i)CandidateCursor();c.base=indices[i];
    const auto& b=((const Env*)bases)[c.base];auto o=outputs(pointers);candidate_stop(o,i);
    if(b.status!=0 || !b.selection)return;
    c.selection=*b.selection;o.episode_done[i]=false;candidate_publish(*c.selection,o,i);
}
extern "C" __global__ void gpu_candidate_step(void* storage,void** pointers,int n,const void* bases,const int* choices,bool canonical,void* heaps,unsigned arena_bytes) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& c=((CandidateCursor*)storage)[i];if(!c.selection || c.selection->complete)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    const auto& match=((const Env*)bases)[c.base].match;auto& s=*c.selection;auto o=outputs(pointers);
    int choice=choices[i];if(choice<0 || choice>=21 || !s.mask[choice])gpu_fail(__LINE__);
    ++s.substep;s.choose(match,choice,canonical);
    while(!s.complete) {
        int count=0,only=-1;for(int k=0;k<21;++k)if(s.mask[k]){++count;only=k;}
        if(count>=2){candidate_publish(s,o,i);return;}
        if(count!=1)gpu_fail(__LINE__);s.choose(match,only,canonical);
    }
    candidate_stop(o,i);
}
extern "C" __global__ void gpu_candidate_destroy(void* storage,int n,void* heaps,unsigned arena_bytes) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    ((CandidateCursor*)storage)[i].~CandidateCursor();
}
