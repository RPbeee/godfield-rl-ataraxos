#include "gfrl_core/vec_env.hpp"
#include <algorithm>
#include <iostream>
using namespace gfrl;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
gf::Match scene(std::vector<gf::CardId> cards) {
    gf::Match m;auto s=m.snapshot().state;
    for(auto& p:s.players){p.items.clear();p.curses.clear();p.guardian=gf::Guardian::none;}
    gf::ItemToken token=100;for(auto f:cards)s.players[0].items.push_back({token++,f,{},false});
    s.players[1].items={{200,6,{},false},{201,7,{},false}};
    m.load_scene(s,0,123);return m;
}
void leaks(){
    for(bool dream:{false,true}) {
        auto a=scene({13,19,120,27}),b=a;auto s=a.snapshot().state;
        if(dream){s.players[0].curses={gf::Curse::dream};s.players[0].items[1].fake_model=27;a.load_scene(s,0,123);s.players[0].items[1].model=27;}
        else {s.players[1].items[0].model=120;s.players[1].items[1].model=27;}
        b.load_scene(s,0,123);
        for(auto bot:{Controller::random,Controller::greedy})for(std::uint64_t seed=0;seed<1000;++seed){
            auto ra=seed,rb=seed;check(bot_action(bot,a,0,ra)==bot_action(bot,b,0,rb),"hidden information changed command");check(ra==rb,"hidden information changed RNG");
        }
    }
    // Proposal wraps the engine sample_action and is intentionally excluded from leak testing.
    std::cout<<"Random/Greedy: 4000 leak comparisons; Proposal excluded (engine sampler)\n";
}
void paired(){
    for(std::uint64_t pair=0;pair<100;++pair){gf::Match a,b;std::uint64_t ca,cb;std::array<std::uint64_t,2> pa,pb;
        reset_evaluation(a,42,2*pair,ca,pa);reset_evaluation(b,42,2*pair+1,cb,pb);
        check(a.snapshot()==b.snapshot(),"pair initial snapshots differ");
        check(ca==cb,"chance seed differs");
        check(evaluation_result(a,2*pair).agent_seat==0 && evaluation_result(b,2*pair+1).agent_seat==1,"pair agent seats not switched");
        for(int seat=0;seat<2;++seat)check(pa[seat]==pb[1-seat],"role policy seed differs");
    }
}
std::vector<EvalResult> run(std::size_t n,std::size_t threads){
    VecEnv env(n,threads,7,true,{},0,{},13,Controller::greedy,123);
    std::vector<std::int32_t> choices(n);
    int steps=0;
    while(!env.eval_done()){
        for(std::size_t i=0;i<n;++i){
            if(env.stage[i]<0){check(std::none_of(env.mask[i].begin(),env.mask[i].end(),[](bool b){return b;}),"idle mask nonempty");choices[i]=-999;}
            else choices[i]=std::find(env.mask[i].begin(),env.mask[i].end(),true)-env.mask[i].begin();
        }
        env.step(choices);check(++steps<100000,"eval never completed");
    }
    const auto results=env.eval_results();check(results.size()==26,"missing games");
    std::fill(choices.begin(),choices.end(),-999);env.step(choices);check(env.eval_results()==results,"idle step changed results");
    for(std::size_t g=0;g<results.size();++g)check(results[g].g==g && results[g].pair==g/2 && results[g].agent_seat==int(g%2),"bad result metadata");
    return results;
}
std::vector<EvalResult> direct_first(bool external=false) {
    std::vector<EvalResult> results;
    for(std::uint64_t g=0;g<26;++g) {
        gf::Match m;std::uint64_t chance;std::array<std::uint64_t,2> policy;
        reset_evaluation(m,123,g,chance,policy);
        while(!m.done()) {
            const auto v=m.view({});
            if(v.phase==gf::MatchPhase::grant){m.grant_step(m.sample_grant(chance));continue;}
            const auto actor=*v.actor;const auto own=m.view(actor);
            if(v.phase==gf::MatchPhase::main && own.board.players[actor].status->confused){m.confusion_step(m.sample_confusion(chance));continue;}
            gf::Command c;
            if(external || actor==g%2) {
                Selection s(m);
                while(!s.complete)s.choose(m,std::find(s.mask.begin(),s.mask.end(),true)-s.mask.begin(),true);
                c=s.command;
            } else c=Greedy::act(m,actor,policy[actor]);
            check(m.legal(c),"direct policy illegal");m.step(c);
        }
        results.push_back(evaluation_result(m,g));
    }
    return results;
}
void determinism(){auto a=run(1,1);check(a==direct_first(),"VecEnv and direct Match evaluator differ");check(a==run(7,2),"env count changed results");check(a==run(32,8),"thread count changed results");
    check(evaluate_bots(Controller::random,Controller::greedy,20,1,42)==evaluate_bots(Controller::random,Controller::greedy,20,8,42),"direct evaluator nondeterministic");
}
void external_eval(){
    VecEnv env(26,3,7,true,{},0,{},13,Controller::external,123);
    VecEnv bot(26,3,7,true,{},0,{},13,Controller::greedy,123);
    for(std::size_t g=0;g<26;++g){
        gf::Match expected;std::uint64_t chance;std::array<std::uint64_t,2> policy;
        reset_evaluation(expected,123,g,chance,policy);
        check(env.match(g).snapshot()==expected.snapshot(),"external deal differs from evaluation seed");
        check(env.match(g).snapshot()==env.match(g^1).snapshot(),"external pair deals differ");
        // Even games expose seat zero immediately in both bot and external modes.
        if(g%2==0)check(env.match(g).snapshot()==bot.match(g).snapshot(),"bot and external deals differ");
    }
    bool seen[2]={false,false};
    std::vector<std::int32_t> choices(26);
    int steps=0;
    while(!env.eval_done()){
        for(std::size_t i=0;i<26;++i){
            if(env.stage[i]<0){choices[i]=-999;continue;}
            seen[env.actor_seat[i]]=true;
            choices[i]=std::find(env.mask[i].begin(),env.mask[i].end(),true)-env.mask[i].begin();
        }
        env.step(choices);check(++steps<100000,"external eval never completed");
    }
    check(seen[0] && seen[1],"external evaluation did not return both seats");
    const auto results=env.eval_results();
    check(results==direct_first(true),"external result differs from direct evaluation");
    bool non_draw=false;
    for(std::size_t g=0;g<26;++g){
        const auto& snapshot=env.match(g).snapshot();
        const int expected=snapshot.truncated ? 0 : snapshot.rewards.at(g%2);
        check(results[g].result==expected,"external result has wrong agent perspective");
        check(results[g].agent_seat==int(g%2),"external result has wrong agent seat");
        non_draw |= expected!=0;
    }
    check(non_draw,"external perspective test needs decisive games");
}
void greedy(){
    auto m=scene({6,13});std::uint64_t rng=0;auto c=Greedy::act(m,0,rng);check(c.kind==gf::CommandKind::play && c.items==std::vector<gf::ItemToken>{101} && c.target==1,"greedy did not choose stronger attack");
    m=scene({191,194});auto state=m.snapshot().state;state.players[0].hp=20;m.load_scene(state,0,123);
    c=Greedy::act(m,0,rng);check(c.items==std::vector<gf::ItemToken>{101} && c.target==0,"greedy heal priority");
    m=scene({120});c=Greedy::act(m,0,rng);check(c.kind==gf::CommandKind::pray,"greedy prayer");
    m=scene({13});state=m.snapshot().state;state.players[1].items={{200,120,{},false},{201,120,{},false}};m.load_scene(state,0,123);
    m.step({0,gf::CommandKind::play,{100},1});c=Greedy::act(m,1,rng);check(c.items.size()==2,"greedy defense did not minimize damage");
    m=scene({13});state=m.snapshot().state;state.players[1].items={{200,120,{},false}};m.load_scene(state,0,123);
    m.step({0,gf::CommandKind::play,{100},1});c=Greedy::act(m,1,rng);check(c.items.size()==1,"greedy defense missing");
    m=scene({5});m.step({0,gf::CommandKind::buy,{100},1});
    if(m.view(0).phase==gf::MatchPhase::response)m.step({1,gf::CommandKind::defend,{},{}});
    c=Greedy::act(m,0,rng);check(c.kind==gf::CommandKind::accept,"greedy purchase");
}
void legal(){
    for(auto bot:{Controller::random,Controller::proposal,Controller::greedy}){
        // Every nonautomatic decision is checked by the direct harness before step().
        auto results=evaluate_bots(bot,bot,1000,8,456);
        check(results.size()==2000,"missing legality games");
        std::uint64_t decisions=0;for(auto& r:results)decisions+=r.decisions;
        std::cout<<"bot="<<int(bot)<<" games="<<results.size()<<" decisions="<<decisions<<'\n';
    }
}
int main(int argc,char** argv){try{check(argc==2,"case required");std::string s=argv[1];if(s=="external")external_eval();else if(s=="leaks")leaks();else if(s=="pairs")paired();else if(s=="determinism")determinism();else if(s=="legal")legal();else if(s=="greedy")greedy();else throw std::runtime_error("unknown case");}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
