#include "vec_env.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
namespace nb = nanobind;
using namespace nb::literals;
using namespace gfrl;
namespace {
template<class T> nb::object view(VecEnv& env, T* data, std::initializer_list<std::size_t> shape) {
    nb::object owner = nb::cast(&env, nb::rv_policy::reference);
    auto array = nb::cast(nb::ndarray<nb::numpy, T>(data, shape, owner));
    array.attr("setflags")("write"_a = false);
    return array;
}
nb::list result_list(const std::vector<EvalResult>& results) {
    nb::list out;
    for(const auto& r:results){nb::dict d;d["g"]=r.g;d["pair"]=r.pair;d["agent_seat"]=r.agent_seat;
        d["result"]=r.result;d["truncated"]=r.truncated;d["decisions"]=r.decisions;d["hp"]=nb::cast(r.hp);out.append(d);}
    return out;
}
nb::dict stats_dict(const VecEnv& env) {
    const auto s = env.stats();
    nb::dict d;
    d["episodes"] = s.episodes; d["decisions"] = s.decisions;
    d["policy_stages"] = s.policy_stages; d["automatic_decisions"] = s.automatic_decisions;
    d["truncations"] = s.truncations; d["confusion_decisions"] = s.confusion_decisions;
    d["grant_decisions"] = s.grant_decisions;
    d["history_dropped"]=s.history_dropped; d["card_overflow"]=s.card_overflow;
    d["used_overflow"]=s.used_overflow; d["token_overflow"]=s.token_overflow;
    return d;
}
nb::dict schema(std::size_t k) {
    nb::dict d; d["T"]=history_offset+k;d["K"]=k;d["C"]=int(C);d["F"]=int(F);
    nb::dict cats,nums,types,positions;
    for(int j=0;j<C;++j) {nb::dict x;x["index"]=j;x["vocab_size"]=cat_vocab[j]; cats[nb::str(cat_names[j].data())]=x;}
    auto names=num_names();for(int j=0;j<F;++j) nums[nb::str(names.at(j).c_str())]=j;
    const char* tn[]={"empty","cls","player","hand","pending","offer","used","decision","reveal","removal","sale","purchase","danger","devil","grant","fate","phenomenon","guardian_change","redraw"};
    for(int j=0;j<19;++j)types[nb::str(tn[j])]=j;
    positions["cls"]=0;positions["self"]=1;positions["opponent"]=2;positions["hand_start"]=3;
    positions["pending"]=21;positions["offer"]=22;positions["self_used_start"]=23;positions["opponent_used_start"]=39;positions["history_start"]=55;
    nb::dict scales;
    for(const auto& name:names) scales[nb::str(name.c_str())]=1.0;
    for(auto j:{Hp,Mp,Cp,Amount,Price,AllocationHp,AllocationMp,AllocationCp}) scales[nb::str(names[j].c_str())]=99.0;
    for(int offset:{int(Delta0),int(Delta0)+18}) {
        for(int j=0;j<3;++j) scales[nb::str(names[offset+j].c_str())]=99.0;
        for(int j=7;j<9;++j) scales[nb::str(names[offset+j].c_str())]=10.0;
    }
    for(auto j:{Gf,Tiebreak,Decisions}) scales[nb::str(names[j].c_str())]=512.0;
    for(auto j:{HandCount,Reflections,Bounces}) scales[nb::str(names[j].c_str())]=18.0;
    scales["hit_rate"]=100.0;scales["age"]=k;
    d["num_divisors"]=scales;
    d["power_transform"]="sign(power)*log1p(abs(power))";
    nb::dict cf;
    const char* feature_names[]={"category","element","is_plus_atk","atk_log1p","hit_rate","defense_log1p","ability","ability_value","curse","guardian","price","cost","gift_rate"};
    const int widths[]={8,6,1,1,1,1,61,1,9,11,1,1,1};
    int column=0;
    for(int j=0;j<13;++j) {nb::dict x;x["index"]=column;x["width"]=widths[j];cf[nb::str(feature_names[j])]=x;column+=widths[j];}
    d["card_features"]=cf;d["Fc"]=card_feature_count;
    d["cat"]=cats;d["num"]=nums;d["types"]=types;d["positions"]=positions;
    d["card_encoding"]="card_index(id)+1; 0=absent";
    d["category_encoding"]="enum+1; 0=absent; seats: 1=self, 2=opponent";
    return d;
}

}
NB_MODULE(_core, m) {
    nb::class_<gf::MatchConfig>(m, "MatchConfig")
        .def(nb::init<>())
        .def_rw("max_decisions", &gf::MatchConfig::max_decisions)
        .def_rw("initial_hand", &gf::MatchConfig::initial_hand)
        .def_rw("hand_limit", &gf::MatchConfig::hand_limit)
        .def_rw("initial_hp", &gf::MatchConfig::initial_hp)
        .def_rw("initial_mp", &gf::MatchConfig::initial_mp)
        .def_rw("initial_cp", &gf::MatchConfig::initial_cp)
        .def_rw("include_fate", &gf::MatchConfig::include_fate)
        .def_rw("external_grants", &gf::MatchConfig::external_grants)
        .def_rw("external_artifacts", &gf::MatchConfig::external_artifacts)
        .def_rw("local_guardian_payloads", &gf::MatchConfig::local_guardian_payloads);
    auto cls = nb::class_<VecEnv>(m, "VecEnv")
        .def("__init__", [](VecEnv* self, std::size_t n,std::size_t threads,std::uint64_t seed,bool canonical,gf::MatchConfig config,std::size_t k,
                             std::array<std::string,2> controls,std::optional<std::size_t> pairs,std::string opponent,std::uint64_t eval_seed){
            new(self)VecEnv(n,threads,seed,canonical,config,k,{controller(controls[0]),controller(controls[1])},pairs,controller(opponent),eval_seed);
        }, "num_envs"_a,"num_threads"_a,"base_seed"_a=0,"canonical"_a=true,"config"_a=gf::MatchConfig{},"history_len"_a=64,
           "controllers"_a=std::array<std::string,2>{"external","external"},"eval_pairs"_a=nb::none(),"opponent"_a="greedy","eval_seed"_a=0)
        .def("eval_done", &VecEnv::eval_done)
        .def("eval_results", [](const VecEnv& e){return result_list(e.eval_results());})
        .def_prop_ro("num_envs", [](const VecEnv& e) { return e.num_envs; })
        .def_prop_ro("num_threads", [](const VecEnv& e) { return e.num_threads; })
        .def("step", [](VecEnv& env, nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig> choices) {
            if (choices.shape(0) != env.num_envs) throw std::invalid_argument("choices must have shape [num_envs]");
            const std::vector<std::int32_t> stable(choices.data(), choices.data() + choices.shape(0));
            nb::gil_scoped_release release;
            env.step(stable);
        }, "choices"_a.noconvert())
        .def("stats", &stats_dict)
        .def("obs_schema", [](const VecEnv& e){return schema(e.history_len);});
#define ONE(name) cls.def_prop_ro(#name, [](VecEnv& e) { return view(e, e.name.data(), {e.num_envs}); });
    ONE(stage) ONE(actor_seat) ONE(chosen_kind) ONE(chosen_count) ONE(chosen_target) ONE(chosen_alloc_hp) ONE(substep)
    ONE(decision_done) ONE(episode_done) ONE(truncated) ONE(episode_id)
#undef ONE
    cls.def_prop_ro("mask", [](VecEnv& e) { return view(e, e.mask.data()->data(), {e.num_envs, 21}); });
    cls.def_prop_ro("chosen_cards", [](VecEnv& e) { return view(e, e.chosen_cards.data(), {e.num_envs, 18}); });
    cls.def_prop_ro("outcome", [](VecEnv& e) { return view(e, e.outcome.data(), {e.num_envs, 2}); });
#define OBS2(name) cls.def_prop_ro(#name, [](VecEnv& e) {return view(e,e.name.data(),{e.num_envs,e.tokens});});
    OBS2(obs_type) OBS2(obs_aux_card) OBS2(obs_valid)
#undef OBS2
    cls.def_prop_ro("obs_cards",[](VecEnv& e){return view(e,e.obs_cards.data(),{e.num_envs,e.tokens,8});});
    cls.def_prop_ro("obs_cat",[](VecEnv& e){return view(e,e.obs_cat.data(),{e.num_envs,e.tokens,C});});
    cls.def_prop_ro("obs_num",[](VecEnv& e){return view(e,e.obs_num.data(),{e.num_envs,e.tokens,F});});
    cls.def_prop_ro("label_opp_hand",[](VecEnv& e){return view(e,e.label_opp_hand.data(),{e.num_envs,297});},"特権情報。方策の入力に使わないこと");
    cls.def_prop_ro("label_opp_hand_count",[](VecEnv& e){return view(e,e.label_opp_hand_count.data(),{e.num_envs});},"特権情報。方策の入力に使わないこと");
    m.def("evaluate_bots",[](const std::string& a,const std::string& b,std::size_t pairs,std::size_t threads,std::uint64_t seed){
        std::vector<EvalResult> results;
        {nb::gil_scoped_release release;results=evaluate_bots(controller(a),controller(b),pairs,threads,seed);}
        return result_list(results);
    },"bot_a"_a,"bot_b"_a,"pairs"_a,"threads"_a,"eval_seed"_a);
    m.def("obs_schema",&schema,"history_len"_a=64);
    m.def("card_features",[]{
        const auto& f=card_features();
        auto a=nb::cast(nb::ndarray<nb::numpy,const float>(f.data(),{297,card_feature_count}));
        a.attr("setflags")("write"_a=false);return a;
    });

}
