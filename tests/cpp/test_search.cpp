#include "gfrl_core/vec_env.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
using namespace gfrl;
void check(bool b,const char* what){if(!b)throw std::runtime_error(what);}
namespace gfrl {
struct TestAccess {
    static void inject(VecEnv& env,const gf::Match& match) {
        auto& e=env.envs_[0];e.match=match;e.selection.reset();e.encoded=false;e.waiting=false;
        for(auto& h:e.history)h=History(env.history_len);
        env.drive(0);
    }
    static void accept(VecEnv& env,const gf::Command& c) {
        env.before_decision(0);check(env.envs_[0].match.legal(c),"fixture illegal");
        env.envs_[0].match.step(c);env.record_decision(0,false);
    }
    static void drive(VecEnv& env){env.envs_[0].encoded=false;env.drive(0);}
    static const History& history(const VecEnv& env,int seat){return env.envs_[0].history[seat];}
    static void copy_searcher(VecEnv& dst,const VecEnv& src,int seat) {dst.envs_[0].history[seat]=src.envs_[0].history[seat];}
    static void poison_opponent(VecEnv& env,int seat) {env.envs_[0].history[seat]=History(0);}
    static void no_projection(const VecEnv& env) {
        check(!env.track_projection,"tracking unexpectedly enabled");
        for(const auto& e:env.envs_) {
            check(!e.projected_before,"projection views allocated while disabled");
            for(const auto& h:e.history)for(const auto& event:h.events())
                check(!event.opposite,"projection recorded while disabled");
        }
    }
    static std::uint64_t chance(const VecEnv& env){return env.envs_[0].chance_rng;}
};
}
gf::Match scene(std::vector<gf::CardId> a={120,120,120},std::vector<gf::CardId> b={120,120,120}) {
    gf::Match m;auto s=m.snapshot().state;
    for(auto& p:s.players){p.items.clear();p.curses.clear();p.guardian=gf::Guardian::none;}
    gf::ItemToken token=100;
    for(auto id:a)s.players[0].items.push_back({token++,id,{},false});
    token=200;for(auto id:b)s.players[1].items.push_back({token++,id,{},false});
    m.load_scene(s,0,19);return m;
}
template<class T>void equal(const Buffer<T>& a,const Buffer<T>& b) {
    check(a.size==b.size && std::memcmp(a.data(),b.data(),a.size*sizeof(T))==0,"buffer differs");
}
void obs_equal(const VecEnv& a,const VecEnv& b) {
    equal(a.obs_type,b.obs_type);equal(a.obs_cat,b.obs_cat);equal(a.obs_cards,b.obs_cards);
    equal(a.obs_aux_card,b.obs_aux_card);equal(a.obs_num,b.obs_num);equal(a.obs_valid,b.obs_valid);
}
void history_equal(const History& a,const History& b) {
    check(a.events().size()==b.events().size(),"history length differs");
    for(std::size_t i=0;i<a.events().size();++i) {
        check(a.events()[i].token==b.events()[i].token,"history token differs");
        check(a.events()[i].decision==b.events()[i].decision,"history time differs");
    }
}
const std::array<std::int32_t,1> row{0};
VecEnv projection_root(std::size_t n=1,std::size_t threads=1,std::uint64_t seed=0) {
    return VecEnv(n,threads,seed,true,{},64,{Controller::external,Controller::external},
                  std::nullopt,Controller::greedy,0,true);
}
void worlds() {
    auto root=projection_root(),other=projection_root();auto m=scene();TestAccess::inject(root,m);
    // Make a history using only public discards. Return to seat0 main.
    TestAccess::accept(root,{0,gf::CommandKind::discard,{100},{}});
    TestAccess::accept(root,{1,gf::CommandKind::discard,{200},{}});TestAccess::drive(root);
    m=root.match(0);check(root.determinize_support(row)[0]==0,"supported root");
    TestAccess::inject(other,m.determinized(0,std::array<gf::CardId,2>{6,7},777));
    TestAccess::copy_searcher(other,root,0);TestAccess::poison_opponent(other,1);
    std::array<std::int32_t,297> assignment{};assignment[card_code(120)]=2;
    const std::array<std::uint64_t,1> seed{42};
    SearchEnv a(1,1),b(1,1);
    a.load_worlds(root,row,assignment,seed);
    check(a.world_status[0]==0,"load failed");obs_equal(root,a);
    history_equal(TestAccess::history(root,0),TestAccess::history(a,0));
    history_equal(TestAccess::history(root,1),TestAccess::history(a,1));
    check(!a.track_projection,"SearchEnv tracks nested projections");
    const auto& copied=TestAccess::history(a,0).events();
    const auto& original=TestAccess::history(root,0).events();
    for(std::size_t i=0;i<copied.size();++i)
        check(copied[i].opposite==original[i].opposite,"copied projection differs");
    // Full physical snapshot and both histories must ignore the true opposing hand/history.
    a.load_worlds(root,row,assignment,seed);b.load_worlds(other,row,assignment,seed);
    check(a.match(0).snapshot()==b.match(0).snapshot(),"world snapshot leaked hidden source hand/allocator");
    obs_equal(a,b);equal(a.mask,b.mask);
    for(int seat=0;seat<2;++seat){a.observe(row,std::array<std::int32_t,1>{seat});b.observe(row,std::array<std::int32_t,1>{seat});equal(a.leaf_obs_num,b.leaf_obs_num);equal(a.leaf_obs_cards,b.leaf_obs_cards);}
    std::mt19937_64 rng(456);
    for(int i=0;i<3000 && !a.episode_done[0];++i) {
        const auto choices=random_choices({a.mask.data(),1},{a.stage.data(),1},rng);
        a.step(choices);b.step(choices);obs_equal(a,b);
        check(a.match(0).snapshot()==b.match(0).snapshot(),"world transitions differ");
        equal(a.decisions,b.decisions);equal(a.result,b.result);
    }
    check(a.episode_done[0],"did not finish");
    const auto snapshot=a.match(0).snapshot();const auto result=a.result[0];
    a.step(std::array<std::int32_t,1>{-999});check(a.match(0).snapshot()==snapshot && a.result[0]==result,"terminal reset");
    check(result==(snapshot.truncated ? 0 : snapshot.rewards[a.searcher_seat[0]]),"wrong relative outcome");
    // Seed-independent source observation with a completely different sampled hand.
    assignment.fill(0);assignment[card_code(6)]=1;
    a.load_worlds(root,row,assignment,std::array<std::uint64_t,1>{999});obs_equal(root,a);
    check(TestAccess::chance(a)!=TestAccess::chance(root),"chance RNG copied");
    // At a later stage support must be the dedicated negative value.
    root.step(std::array<std::int32_t,1>{int(gf::CommandKind::discard)});
    check(root.substep[0]>0 && root.determinize_support(row)[0]==-1,"partial decision accepted");
    a.load_worlds(root,row,assignment,seed);check(a.world_status[0]==2 && a.episode_done[0],"unsupported root moved");
}
void private_history() {
    gf::CardId cure=0;
    for(const auto& card:gf::catalog())if(card.ability==gf::Ability::remove_all_curses && gf::match_supported(card.id)){cure=card.id;break;}
    check(cure!=0,"missing cure fixture");
    for(int actor=0;actor<2;++actor) {
        auto m=scene({cure,19,120},{cure,19,120});auto state=m.snapshot().state;
        state.players[actor].curses={gf::Curse::dream};state.players[actor].mp=99;
        state.players[actor].items[1].fake_model=27;m.load_scene(state,actor,42);
        auto root=projection_root();TestAccess::inject(root,m);
        TestAccess::accept(root,{gf::PlayerId(actor),gf::CommandKind::play,{gf::ItemToken(actor ? 200 : 100)},gf::PlayerId(actor)});
        check(!root.match(0).snapshot().last_reveals.empty(),"private cure reveal missing");
        const auto rebuilt=TestAccess::history(root,0).projected();
        const auto& truth=TestAccess::history(root,1);
        std::vector<Token> expected;
        for(const auto& event:truth.events())if(event.token.type!=Reveal)expected.push_back(event.token);
        check(rebuilt.events().size()==expected.size(),"private history approximation length");
        for(std::size_t i=0;i<expected.size();++i)check(rebuilt.events()[i].token==expected[i],"private history changed public fields");
    }
}
void conversion() {
    auto m=scene({120},{235,235,120});auto s=m.snapshot().state;
    s.players[1].items[0].used=s.players[1].items[1].used=true;m.load_scene(s,0,42);
    std::array<std::int32_t,297> assignment{};assignment[card_code(235)]=1;assignment[card_code(120)]=3;
    auto counts=unused_assignment(m,0,assignment);
    check(counts[card_code(235)]==0 && counts[card_code(120)]==3,"clamped used subtraction");
    auto root=projection_root();TestAccess::inject(root,m);SearchEnv env(1,1);
    env.load_worlds(root,row,assignment,std::array<std::uint64_t,1>{1});check(env.world_status[0]==0,"valid assignment rejected");
    check(env.match(0).snapshot().state.players[1].items.size()==5,"used physical count");
    for(int variant=0;variant<5;++variant) {
        auto bad=assignment;
        if(variant==0)bad[0]=1;
        if(variant==1)bad[card_code(120)]=-1;
        if(variant==2)bad[card_code(120)]=19;
        if(variant==3)bad[card_code(288)]=1;
        if(variant==4){bad.fill(0);bad[card_code(120)]=18;}
        env.load_worlds(root,row,bad,std::array<std::uint64_t,1>{1});check(env.world_status[0]==1,"bad assignment accepted");
        env.step(std::array<std::int32_t,1>{-999});check(env.episode_done[0] && env.stage[0]==-1 && env.decisions[0]==0,"invalid row moved");
    }
}
void projection() {
    // Synthetic journals exercise every history-only field without privileged opponent facts.
    auto m=scene();auto b=m.view(0),a=b;auto snap=m.snapshot();
    a.decisions=1;a.board.players[0].status->hp-=5;
    a.board.players[1].status->mp+=3;
    snap.last_reveals={{0,19,27,false},{0,6,7,true},{1,13,14,true}};
    snap.last_guardian_changes={{0,gf::Guardian::none,gf::Guardian::mars,gf::GuardianChangeReason::summon,0},
        {1,gf::Guardian::none,gf::Guardian::venus,gf::GuardianChangeReason::summon,2}};
    for(const auto& e:snap.last_reveals)if(!e.owner_only || e.owner==0)a.last_reveals.push_back({e.owner,e.face,e.model});
    for(const auto& e:snap.last_guardian_changes)a.last_guardian_changes.push_back({e.player,e.before,e.after,e.reason});
    a.last_removal=gf::RemovalEvent{0,0,1,205,{6,7}};a.last_sale=gf::SaleEvent{1,1,0,6,1,{}};
    a.last_purchase=gf::PurchaseEvent{0,1,0,7,10,gf::PurchaseReason::bought,{}};
    a.last_danger=gf::DangerEvent{0,1,{94},false,99,{}};a.last_devil=gf::DevilEvent{1,240,{}};
    a.last_grant=gf::DevilEvent{1,0,{}};a.last_fate=gf::FateEvent{0,288};a.last_phenomenon=gf::PhenomenonView{288,1};a.last_redraws={1};
    gf::PublicDecision d;d.actor=0;d.target=1;d.kind=gf::CommandKind::discard;d.faces={120};
    auto pa=project_history(a,snap,0),pb=project_history(b,m.snapshot(),0);
    check(pa.last_reveals.size()==1 && pa.last_guardian_changes.size()==1,"private events leaked");
    History source,opposite;
    source.append(b,a,d,0);opposite.append(pb,pa,d,1);source.set_projection(opposite);
    history_equal(source.projected(),opposite);
    // Own-only facts of the opponent remain absent; public companion values match exact opposing history.
    auto exact=pa;exact.last_reveals.push_back({1,13,14});History truth;truth.append(pb,exact,d,1);
    check(truth.events().size()==opposite.events().size()+1,"missing private reveal fixture");
    // Fog hides the original searcher's status after the swap, including numeric deltas.
    b.board.players[1].status->curses={gf::Curse::fog};a.board.players[1].status->curses={gf::Curse::fog};
    pb=project_history(b,m.snapshot(),0);pa=project_history(a,snap,0);
    check(!pb.board.players[0].status && !pa.board.players[0].status,"fog pane leaked");
    History fog;fog.append(pb,pa,d,1);
    check(fog.events().front().token.num[Delta0+18+3]==1 && fog.events().front().token.num[Delta0+18]==0,"fog delta leaked");
    // A short ring cannot resurrect a source event evicted by private events.
    History short_source(2);short_source.append(b,a,d,0);short_source.set_projection(fog);
    check(short_source.projected().events().size()<=2,"projection resurrected evicted history");
}
void tracking_disabled() {
    VecEnv env(8,2,42);std::mt19937_64 rng(91);
    for(int i=0;i<1000;++i) {
        TestAccess::no_projection(env);
        env.step(random_choices({env.mask.data(),8},{env.stage.data(),8},rng));
    }
    TestAccess::no_projection(env);
}
void measurement() {
    auto env=projection_root(32,4,20261108);SearchEnv worlds(32,4);std::mt19937_64 rng(101);
    std::array<std::int32_t,32> indices{};std::array<std::uint64_t,32> seeds{};
    std::array<std::int32_t,32*297> assignments{};
    for(int i=0;i<32;++i){indices[i]=i;seeds[i]=10000+i;}
    std::uint64_t loaded=0;
    std::uint64_t positions=0,with_used=0,used=0;
    for(int step=0;step<2000;++step) {
        for(std::size_t i=0;i<env.num_envs;++i)if(env.substep[i]==0) {
            const auto& players=env.match(i).snapshot().state.players;
            const auto n=std::count_if(players[1-env.actor_seat[i]].items.begin(),players[1-env.actor_seat[i]].items.end(),[](const auto& item){return item.used;});
            ++positions;with_used+=n>0;used+=n;
        }
        if(step%10==0) {
            const auto support=env.determinize_support(indices);
            std::copy_n(env.label_opp_hand.data(),assignments.size(),assignments.begin());
            worlds.load_worlds(env,indices,assignments,seeds);
            for(int i=0;i<32;++i)if(support[i]==0) {
                ++loaded;check(worlds.world_status[i]==0,"self-play root failed loading");
                const auto t=env.tokens;
                check(std::memcmp(env.obs_num.data()+i*t*F,worlds.obs_num.data()+i*t*F,t*F*sizeof(float))==0,"self-play root numeric observation changed");
                check(std::memcmp(env.obs_cat.data()+i*t*C,worlds.obs_cat.data()+i*t*C,t*C)==0,"self-play root categorical observation changed");
                check(std::memcmp(env.obs_cards.data()+i*t*8,worlds.obs_cards.data()+i*t*8,t*8*sizeof(std::int16_t))==0,"self-play root cards changed");
            }
            std::array<std::int32_t,32> choices{};
            for(int i=0;i<32;++i)if(worlds.stage[i]>=0)
                choices[i]=std::find(worlds.mask[i].begin(),worlds.mask[i].end(),true)-worlds.mask[i].begin();
            worlds.step(choices);
        }
        env.step(random_choices({env.mask.data(),32},{env.stage.data(),32},rng));
    }
    check(loaded>1000,"too few varied roots tested");
    std::cout<<"varied self-play worlds="<<loaded<<"\n";
    std::cout<<"used-card measurement: positions="<<positions<<" with_used="<<with_used<<" fraction="<<double(with_used)/positions<<" mean="<<double(used)/positions<<" conditional_mean="<<double(used)/with_used<<"\n";
}
int main(){try{worlds();private_history();conversion();projection();tracking_disabled();measurement();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
