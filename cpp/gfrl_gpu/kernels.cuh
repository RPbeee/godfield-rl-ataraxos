namespace gpu_training {
namespace gf=godfield::official;
static_assert(!gf::MatchConfig{}.include_fate && !gf::MatchConfig{}.external_grants
    && !gf::MatchConfig{}.external_artifacts && !gf::MatchConfig{}.local_guardian_payloads,
    "GPU training specialization needs the current VecEnv default MatchConfig");
using namespace gfrl;
struct Outputs {
    signed char *obs_type,*obs_cat;
    short *obs_cards,*obs_aux_card;
    float *obs_num;
    bool *obs_valid;
    unsigned char *label_opp_hand,*label_opp_hand_count;
    signed char *stage,*actor_seat,*chosen_kind,*chosen_cards,*chosen_count,*chosen_target,*chosen_alloc_hp,*outcome;
    bool *mask;
    short *substep;
    bool *decision_done,*episode_done,*truncated;
    unsigned long long *episode_id,*decisions,*stats;
    signed char *ready;
};
struct Env {
    gf::Match match;
    gs::optional<Selection> selection;
    gs::uint64_t chance_rng=0,episode=0;
    History history[2];
    gf::MatchView before[2];
    gs::vector<Token> tokens;
    bool encoded=false,dirty=false;
    int original_index=0;
};
unsigned long long mix(unsigned long long x) {
    x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
void reset(Env& e,Outputs o,int i,unsigned long long seed) {
    auto s=mix(seed^mix(e.original_index)^mix(e.episode+0x243f6a8885a308d3ULL));
    e.chance_rng=mix(s^0xa4093822299f31d0ULL);
    const gs::array<gf::PlayerSetup,2> p{{{0,gf::Team::solo,"A"},{1,gf::Team::solo,"B"}}};
    e.match.reset(p,mix(s^0x13198a2e03707344ULL));
    e.selection.reset();e.encoded=false;for(auto& h:e.history)h.clear();o.episode_id[i]=e.episode;
}
void before(Env& e) {for(int s=0;s<2;++s)e.before[s]=e.match.view(e.match.snapshot().state.players[s].id,true);}
void record(Env& e,Outputs o,int i,bool automatic) {
    for(int s=0;s<2;++s){auto a=e.match.view(e.before[s].board.viewer,true);auto d=e.match.last_decision(e.before[s].board.viewer);
        if(!d)gpu_fail(__LINE__);e.history[s].append(e.before[s],a,*d,*e.before[s].board.viewer);}
    e.encoded=false;o.stats[12*i+1]++;o.stats[12*i+3]+=automatic;o.decision_done[i]=true;e.selection.reset();
}
void encode_view(Env& e,Outputs o,int i,int history_len) {
    const auto& s=*e.selection;
    if(!e.encoded) {
        auto& players=e.match.snapshot().state.players;auto seat=s.actor_seat;
        gs::vector<gf::ItemToken> order;for(const auto& item:players[seat].items)order.push_back(item.token);
        auto v=e.match.view(players[seat].id);
        // History already lives in a device ring. The cooperative publication
        // kernel reads it directly instead of copying every event here.
        e.tokens.reserve(history_offset);encode(v,players[seat].id,order,e.tokens);
        for(int j=0;j<history_offset;++j) {
            const auto& t=e.tokens[j];o.stats[12*i+8]+=t.num[CountOverflow];
            if(t.type==Player)o.stats[12*i+9]+=t.num[UsedOverflow];
        }
        e.dirty=true;
        e.encoded=true;
    }
}
void publish_prefix(Env& e,Outputs o,int i) {
    auto& s=*e.selection;
    o.stage[i]=int(s.stage);o.actor_seat[i]=s.actor_seat;
    for(int k=0;k<21;++k)o.mask[21*i+k]=s.mask[k];
    o.chosen_kind[i]=s.kind_chosen?int(s.command.kind):-1;
    o.chosen_target[i]=s.stage>=Stage::alloc_hp?(s.command.target?(*s.command.target==s.command.actor?1:2):0):-1;
    o.chosen_alloc_hp[i]=s.stage==Stage::alloc_mp?(s.command.allocation->hp==99?20:s.command.allocation->hp/5):-1;
    o.chosen_count[i]=s.count;o.substep[i]=s.substep;
    for(int k=0;k<18;++k)o.chosen_cards[18*i+k]=s.cards[k];
}
void drive(Env& e,Outputs o,int i,int history_len,unsigned long long seed,bool canonical,bool encode_now=true) {
    for(;;) {
        const auto& snap=e.match.snapshot();
        for(const auto& p:snap.state.players)if(p.items.size()>18)gpu_fail(__LINE__);
        if(e.match.done()) {
            if(o.episode_done[i])gpu_fail(__LINE__);
            o.episode_done[i]=true;o.truncated[i]=snap.truncated;
            for(int s=0;s<2;++s)o.outcome[2*i+s]=snap.truncated?0:snap.rewards[s];
            ++o.stats[12*i];o.stats[12*i+4]+=snap.truncated;++e.episode;reset(e,o,i,seed);continue;
        }
        // VecEnv accepts only default MatchConfig. Positive confusion is
        // created only by external phenomenon289/fate207, and grant pauses
        // require external_* configuration. Neither can occur in this API.
        // Keep guards so future changes cannot silently broaden the domain.
        if(snap.phase==gf::MatchPhase::grant)gpu_fail(__LINE__);
        for(const auto& p:snap.state.players)if(p.confusion_count)gpu_fail(__LINE__);
        if(!e.selection)e.selection.emplace(e.match,canonical);
        auto& s=*e.selection;
        if(s.complete) {bool automatic=s.substep==0;before(e);e.match.step(s.command);record(e,o,i,automatic);continue;}
        int count=0,choice=-1;for(int k=0;k<21;++k)if(s.mask[k]){++count;choice=k;}
        if(count>=2){if(encode_now)encode_view(e,o,i,history_len);publish_prefix(e,o,i);return;}
        if(count!=1)gpu_fail(__LINE__);s.choose(e.match,choice,canonical);
    }
}
Outputs outputs(void** p) {
    Outputs o;int k=0;
    o.obs_type=(signed char*)p[k++];o.obs_cat=(signed char*)p[k++];
    o.obs_cards=(short*)p[k++];o.obs_aux_card=(short*)p[k++];o.obs_num=(float*)p[k++];o.obs_valid=(bool*)p[k++];
    o.label_opp_hand=(unsigned char*)p[k++];o.label_opp_hand_count=(unsigned char*)p[k++];
    o.stage=(signed char*)p[k++];o.actor_seat=(signed char*)p[k++];o.chosen_kind=(signed char*)p[k++];
    o.chosen_cards=(signed char*)p[k++];o.chosen_count=(signed char*)p[k++];o.chosen_target=(signed char*)p[k++];
    o.chosen_alloc_hp=(signed char*)p[k++];o.outcome=(signed char*)p[k++];o.mask=(bool*)p[k++];o.substep=(short*)p[k++];
    o.decision_done=(bool*)p[k++];o.episode_done=(bool*)p[k++];o.truncated=(bool*)p[k++];
    o.episode_id=(unsigned long long*)p[k++];o.decisions=(unsigned long long*)p[k++];o.stats=(unsigned long long*)p[k++];o.ready=(signed char*)p[k++];return o;
}
}
extern "C" __global__ void gpu_sizes(unsigned long long* p) {
    p[0]=sizeof(gpu_training::Env);p[1]=gfrl::C;p[2]=gfrl::F;
}
extern "C" __global__ void gpu_tables() {
    using namespace godfield::official;
    for(const auto& c:catalog()) {
        gpu_can_start_main[c.id]=can_start_main(c.id);
        gpu_single_cost[c.id]=magic_cost({&c.id,1});
        gpu_response_starts[c.id][0]=gpu_response_starts[c.id][1]=0;
        for(unsigned category=0;category<8;++category)for(unsigned element=0;element<7;++element)for(unsigned attack=0;attack<2;++attack){
            ResponseContext context;context.incoming_category=static_cast<Category>(category);
            if(element<6)context.incoming_element=static_cast<Element>(element);
            context.incoming_is_attack=attack;
            const unsigned bit=(category*7+element)*2+attack;
            if(can_append_response({}, {c.id,false},context))gpu_response_starts[c.id][bit/64]|=1ull<<(bit%64);
        }
    }
    int n=0;for(const auto& c:catalog())if(match_supported(c.id))for(int j=0;j<c.gift_rate;++j)gpu_gift_tickets[n++]=c.id;
    if(n!=499)gpu_fail(__LINE__);
    for(const auto& a:catalog()) {
        gpu_dream_count[a.id]=0;
        for(const auto& b:catalog())if(match_dream_compatible(a.id,b.id))gpu_dream_faces[a.id][gpu_dream_count[a.id]++]=b.id;
    }
    for(int i=0;i<11;++i)gpu_guardian_count[i]=0;
    for(const auto& c:catalog())if(match_guardian_supported(c.id)) {
        int g=int(c.guardian);int w=c.guardian_attack_rate?c.guardian_attack_rate:1;
        for(int j=0;j<w;++j)gpu_guardian_moves[g][gpu_guardian_count[g]++]=c.id;
    }
}
extern "C" __global__ void gpu_pool_init(void* memory,unsigned long long bytes) {
    auto* pool=(GpuPool*)memory;pool->used=0;pool->capacity=bytes;
}
static_assert(sizeof(GpuArena)<=256);
extern "C" __global__ void gpu_init(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,void* pool) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto* arena=reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes));
    arena->initialize(arena_bytes);arena->pool=(GpuPool*)pool;bind_arena(arena);
    auto& e=*new((Env*)storage+i)Env();e.original_index=i;for(auto& h:e.history)h=History(history_len);
    auto o=outputs(pointers);reset(e,o,i,seed);drive(e,o,i,history_len,seed,canonical);
    o.ready[i]=127;
    o.decision_done[i]=o.episode_done[i]=o.truncated[i]=false;o.outcome[2*i]=o.outcome[2*i+1]=0;
}
extern "C" __global__ void gpu_step(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,const int* choices,const int* indices,const int* active_count) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(active_count){if(i>=*active_count)return;i=indices[i];}
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    auto& e=((Env*)storage)[i];auto o=outputs(pointers);int c=choices[i];
    if(c==-1 && e.selection && e.selection->substep==0)return;
    if(c<0 || c>=21 || !e.selection || !e.selection->mask[c])gpu_fail(__LINE__);
    o.decision_done[i]=o.episode_done[i]=o.truncated[i]=false;o.outcome[2*i]=o.outcome[2*i+1]=0;
    ++o.stats[12*i+2];++e.selection->substep;e.selection->choose(e.match,c,canonical);drive(e,o,i,history_len,seed,canonical);
}
// Keep cursor-only selections out of the much larger world-transition call
// graph. The second kernel executes only rows whose command is complete.
extern "C" __global__ void gpu_choose(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,const int* choices,const int* indices,const int* active_count) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(active_count){if(i>=*active_count)return;i=indices[i];}
    auto& e=((Env*)storage)[i];int c=choices[i];
    if(c==-1 && e.selection && e.selection->substep==0)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    auto o=outputs(pointers);
    if(c<0 || c>=21 || !e.selection || !e.selection->mask[c])gpu_fail(__LINE__);
    o.decision_done[i]=o.episode_done[i]=o.truncated[i]=false;o.outcome[2*i]=o.outcome[2*i+1]=0;
    ++o.stats[12*i+2];++e.selection->substep;e.selection->choose(e.match,c,canonical);
    while(!e.selection->complete) {
        int count=0,choice=-1;for(int k=0;k<21;++k)if(e.selection->mask[k]){++count;choice=k;}
        if(count>=2){publish_prefix(e,o,i);return;}
        if(count!=1)gpu_fail(__LINE__);e.selection->choose(e.match,choice,canonical);
    }
    o.ready[i]=static_cast<signed char>(e.selection->command.kind);
}
extern "C" __global__ void gpu_advance(void* storage,void** pointers,int n,int history_len,unsigned long long seed,bool canonical,void* heaps,unsigned arena_bytes,const int* choices,const int* indices,const int* active_count) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(active_count){if(i>=*active_count)return;i=indices[i];}
    auto& e=((Env*)storage)[i];if(!e.selection || !e.selection->complete)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    auto o=outputs(pointers);drive(e,o,i,history_len,seed,canonical,false);o.ready[i]=127;
}
extern "C" __global__ void gpu_encode_view(void* storage,void** pointers,int n,int history_len,void* heaps,unsigned arena_bytes) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto& e=((Env*)storage)[i];if(e.encoded)return;
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    encode_view(e,outputs(pointers),i,history_len);
}
extern "C" __global__ void gpu_destroy(void* storage,int n,void* heaps,unsigned arena_bytes) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
    bind_arena(reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes)));
    ((gpu_training::Env*)storage)[i].~Env();}
}

extern "C" __global__ void gpu_stats(void* storage,void** pointers,int n,unsigned* high_water,void* heaps,unsigned arena_bytes) {
    using namespace gpu_training;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto& e=((Env*)storage)[i];auto o=outputs(pointers);
    o.stats[12*i+7]=e.history[0].dropped+e.history[1].dropped;
    o.stats[12*i+11]=e.history[0].card_overflow+e.history[1].card_overflow;
    high_water[i]=reinterpret_cast<GpuArena*>(static_cast<char*>(heaps)+i*static_cast<unsigned long long>(arena_bytes))->offset;
}

// Each block copies one newly encoded view with contiguous global transactions.
// Padding is produced here; it never occupies the serial rule thread's arena.
extern "C" __global__ void gpu_observe(void* storage,void** pointers,int n,int history_len) {
    using namespace gpu_training;
#ifdef __CUDACC_RTC__
    const int i=blockIdx.x;
    const int lane=threadIdx.x,stride=blockDim.x;
#else
    const int i=blockIdx.x;
    const int lane=0,stride=1;
#endif
    if(i>=n)return;auto& e=((Env*)storage)[i];if(!e.dirty)return;
    const auto& events=e.history[e.selection->actor_seat].events();
    auto o=outputs(pointers);const int t=history_offset+history_len,have=history_offset+events.size();
    for(int card=lane;card<297;card+=stride)o.label_opp_hand[i*297+card]=0;
#ifdef __CUDACC_RTC__
    __syncthreads();
#endif
    if(lane==0) {
        const auto& hand=e.match.snapshot().state.players[1-e.selection->actor_seat].items;
        o.label_opp_hand_count[i]=hand.size();
        for(const auto& item:hand)++o.label_opp_hand[i*297+card_code(item.model)];
    }
    for(int a=lane;a<t;a+=stride) {
        const auto* token=a<have?(a<history_offset?&e.tokens[a]:&events[a-history_offset].token):nullptr;
        o.obs_type[i*t+a]=token?token->type:0;
        o.obs_aux_card[i*t+a]=token?token->aux:0;
        o.obs_valid[i*t+a]=token && token->type!=0;
    }
    for(int a=lane;a<t*8;a+=stride) {
        const int row=a/8;const auto* token=row<have?(row<history_offset?&e.tokens[row]:&events[row-history_offset].token):nullptr;
        o.obs_cards[i*t*8+a]=token?token->cards[a%8]:0;
    }
    for(int a=lane;a<t*C;a+=stride) {
        const int row=a/C;const auto* token=row<have?(row<history_offset?&e.tokens[row]:&events[row-history_offset].token):nullptr;
        o.obs_cat[i*t*C+a]=token?token->cat[a%C]:0;
    }
    for(int a=lane;a<t*F;a+=stride) {
        const int row=a/F;float value=0;
        if(row<history_offset)value=e.tokens[row].num[a%F];
        else if(row<have) {
            const auto& event=events[row-history_offset];
            value=a%F==Age?(history_len?float(e.match.snapshot().decisions-event.decision)/history_len:0):event.token.num[a%F];
        }
        o.obs_num[i*t*F+a]=value;
    }
#ifdef __CUDACC_RTC__
    __syncthreads();
#endif
    if(lane==0)e.dirty=false;
}

// Reuse the live Token objects; initialize their bytes cooperatively before a
// rule thread fills a new view. Substep-only calls keep the published tensors.
extern "C" __global__ void gpu_clear_tokens(void* storage,int n,const int* indices,const int* count,const int* choices) {
    using namespace gpu_training;int i=blockIdx.x;
    if(i>=n)return;if(count){if(i>=*count)return;i=indices[i];}
    if(choices[i]<0)return;auto& e=((Env*)storage)[i];
    static_assert(sizeof(Token)%4==0);
#ifdef __CUDACC_RTC__
    const int lane=threadIdx.x,stride=blockDim.x;
#else
    const int lane=0,stride=1;
#endif
    auto* p=reinterpret_cast<unsigned char*>(e.tokens.data());
    for(int at=lane*4;at<history_offset*sizeof(Token);at+=stride*4) {
        p[at]=0;p[at+1]=0;p[at+2]=0;p[at+3]=0;
    }
}
extern "C" __global__ void gpu_clear_unencoded(void* storage,int n) {
    using namespace gpu_training;int i=blockIdx.x;if(i>=n)return;
    auto& e=((Env*)storage)[i];if(e.encoded)return;
    auto* p=reinterpret_cast<unsigned*>(e.tokens.data());
    for(int at=threadIdx.x;at<history_offset*sizeof(Token)/4;at+=blockDim.x)p[at]=0;
}
