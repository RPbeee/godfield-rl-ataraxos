#include "gfrl_core/vec_env.hpp"
#include <cstring>
#include <iostream>
#include <random>
using namespace gfrl;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
namespace gfrl {
struct TestAccess {
    static const History& history(const VecEnv& e,int seat){return e.envs_[0].history[seat];}
    static std::uint64_t chance(const VecEnv& e){return e.envs_[0].chance_rng;}
};
}
void compare(const VecEnv& a,const VecEnv& b) {
    check(a.match(0).snapshot()==b.match(0).snapshot(),"private snapshot differs");
    for(int seat=0;seat<2;++seat) {
        const auto& x=TestAccess::history(a,seat).events();const auto& y=TestAccess::history(b,seat).events();
        check(x.size()==y.size(),"private history size differs");
        for(std::size_t i=0;i<x.size();++i) {
            check(x[i].token==y[i].token && x[i].decision==y[i].decision,"private history differs");
        }
    }
#define SAME(name) check(std::memcmp(a.name.data(),b.name.data(),a.name.size*sizeof(*a.name.data()))==0,#name " differs")
    SAME(obs_type);SAME(obs_cat);SAME(obs_cards);SAME(obs_aux_card);SAME(obs_num);SAME(obs_valid);
    SAME(stage);SAME(mask);SAME(substep);SAME(chosen_cards);SAME(actor_seat);
    SAME(label_opp_hand);SAME(label_opp_unused_hand);
#undef SAME
}
int main(){try {
    VecEnv src(1,1,43); // No projected history or determinization support needed.
    SearchEnv a(1,1,64,10000),b(1,2,64,10000),seeded(1,1);
    const std::array<std::int32_t,1> row{0};const std::array<std::uint64_t,1> seed{951};
    std::mt19937 rng(341);
    bool saw_response=false,saw_mid=false,saw_unsupported=false;
    for(int t=0;t<1000;++t) {
        const auto original=src.match(0).snapshot();
        a.diagnostic_copy_exact(src,row,row,seed,false);
        b.diagnostic_copy_exact(src,row,row,seed,false);
        compare(src,a);compare(a,b);
        check(TestAccess::chance(src)==TestAccess::chance(a),"external chance was not copied");
        seeded.diagnostic_copy_exact(src,row,row,seed,true);
        auto expected=original;expected.rng=seed[0];
        check(seeded.match(0).snapshot()==expected,"reseed changed state or historical anchor");
        check(src.match(0).snapshot()==original,"source mutated");
        saw_response |= original.phase==gf::MatchPhase::response;
        saw_mid |= src.substep[0]>0;
        if(src.substep[0]==0)saw_unsupported |= src.determinize_support(row)[0]!=0;
        std::vector<int> legal;
        for(int c=0;c<21;++c)if(src.mask[0][c])legal.push_back(c);
        const std::array<std::int32_t,1> choice{legal[rng()%legal.size()]};
        a.step(choice);b.step(choice);
        compare(a,b);
        // The re-seeded stream must also execute response journals successfully.
        seeded.step(choice);
        src.step(choice);
    }
    check(saw_response && saw_mid && saw_unsupported,"missing response/cursor/unsupported coverage");
    bool rejected=false;
    try{src.diagnostic_copy_exact(a,row,row,seed,false);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"ordinary env accepted diagnostic load");
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
