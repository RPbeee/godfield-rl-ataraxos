// Binary CPU replay witness for the CUDA truncation/reset regression.
#include "gfrl_core/vec_env.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace gfrl {
struct TestAccess {
    static void counter(VecEnv& env, std::uint64_t value) {
        auto& row=env.envs_.at(0);
        auto snapshot=row.match.snapshot();snapshot.decisions=value;
        row.match.restore(snapshot);
        row.selection.reset();row.encoded=false;env.drive(0);
    }
};
}

template<class T> void write(std::ostream& out,const T& value) {
    out.write(reinterpret_cast<const char*>(&value),sizeof(T));
}
template<class T> void buffer(std::ostream& out,const gfrl::Buffer<T>& value) {
    out.write(reinterpret_cast<const char*>(value.data()),value.size*sizeof(T));
}
void frame(std::ostream& out,gfrl::VecEnv& env,std::int32_t choice) {
    write(out,choice);
#define FIELD(name) buffer(out,env.name)
    FIELD(obs_type);FIELD(obs_cat);FIELD(obs_cards);FIELD(obs_aux_card);FIELD(obs_num);FIELD(obs_valid);
    FIELD(label_opp_hand);FIELD(label_opp_hand_count);FIELD(stage);FIELD(actor_seat);FIELD(chosen_kind);
    FIELD(chosen_cards);FIELD(chosen_count);FIELD(chosen_target);FIELD(chosen_alloc_hp);FIELD(outcome);
    FIELD(mask);FIELD(substep);FIELD(decision_done);FIELD(episode_done);FIELD(truncated);FIELD(episode_id);FIELD(decisions);
#undef FIELD
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("usage: gfrl_gpu_oracle OUTPUT");
        gfrl::VecEnv env(1,1,42);
        gfrl::TestAccess::counter(env,511);
        std::ofstream out(argv[1],std::ios::binary|std::ios::trunc);
        if(!out)throw std::runtime_error("cannot open witness");
        std::uint32_t count=0;write(out,count);
        frame(out,env,-1);++count;
        while(!env.episode_done[0] && count<24) {
            std::int32_t choice=-1;
            for(int k=0;k<21;++k)if(env.mask[0][k]){choice=k;break;}
            if(choice<0)throw std::runtime_error("empty CPU mask");
            env.step({&choice,1});frame(out,env,choice);++count;
        }
        const auto stats=env.stats();
        if(!env.truncated[0] || stats.truncations!=1 || env.episode_id[0]!=1)
            throw std::runtime_error("fixture did not reach truncation/reset");
        for(auto value:{stats.episodes,stats.decisions,stats.policy_stages,stats.automatic_decisions,
                       stats.truncations,stats.confusion_decisions,stats.grant_decisions,stats.history_dropped,
                       stats.card_overflow,stats.used_overflow,stats.token_overflow})write(out,value);
        out.seekp(0);write(out,count);
        std::cout<<"CPU truncation witness: "<<count<<" states\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
