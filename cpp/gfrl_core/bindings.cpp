#include "vec_env.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
namespace nb = nanobind;
using namespace nb::literals;
using namespace gfrl;
// Private fixture for Python end-to-end search leak tests. No production path
// calls this; replacement is restricted to supported decision boundaries.
namespace gfrl {
struct TestAccess {
    static void danger_root(VecEnv& env, bool opponent_209) {
        auto& e=env.envs_.at(0);
        auto state=e.match.snapshot().state;
        for(auto& p:state.players) {p.items.clear();p.curses.clear();p.guardian=gf::Guardian::none;}
        state.players[0].items={{100,94,{},false},{101,232,{},false}};
        if(opponent_209)state.players[1].items={{200,209,{},false}};
        e.match.load_scene(state,state.players[0].id,19);
        for(auto& h:e.history)h=History(env.history_len);
        e.selection.reset();e.encoded=false;e.waiting=false;env.drive(0);
    }
    static void replace_hidden_hand(VecEnv& env, std::size_t row, std::uint64_t seed,
                                    std::optional<std::uint64_t> decisions) {
        auto& e=env.envs_.at(row);
        if(env.substep[row]!=0 || !e.selection)throw std::invalid_argument("not a root");
        const auto seat=e.selection->actor_seat;
        const auto id=e.match.snapshot().state.players[seat].id;
        // An empty hidden hand is a deliberately different physical source.
        e.match=e.match.determinized(id,std::span<const gf::CardId>{},seed);
        if(decisions) {auto state=e.match.snapshot();state.decisions=*decisions;e.match.restore(state);}
        e.selection.reset();e.encoded=false;
        env.drive(row);
    }
};
}
namespace {
template<class T> nb::object view(VecEnv& env, T* data, std::initializer_list<std::size_t> shape) {
    nb::object owner = nb::cast(&env, nb::rv_policy::reference);
    auto array = nb::cast(nb::ndarray<nb::numpy, T>(data, shape, owner));
    array.attr("setflags")("write"_a = false);
    return array;
}
using Rows = nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig>;
using Assignments = nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<2>, nb::c_contig>;
using Seeds = nb::ndarray<nb::numpy, const std::uint64_t, nb::ndim<1>, nb::c_contig>;
nb::object int_array(std::span<const std::int32_t> values) {
    auto* p=new std::int32_t[values.size()];std::copy(values.begin(),values.end(),p);
    nb::capsule owner(p,[](void* ptr) noexcept {delete[] static_cast<std::int32_t*>(ptr);});
    auto out=nb::cast(nb::ndarray<nb::numpy,std::int32_t>(p,{values.size()},owner));
    out.attr("setflags")("write"_a=false);return out;
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
    m.def("_test_danger_root", &TestAccess::danger_root);
    m.def("_test_replace_hidden_hand", &TestAccess::replace_hidden_hand,
          "env"_a, "row"_a, "seed"_a, "decisions"_a=nb::none());
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
                             std::array<std::string,2> controls,std::optional<std::size_t> pairs,std::string opponent,std::uint64_t eval_seed,bool track_projection){
            new(self)VecEnv(n,threads,seed,canonical,config,k,{controller(controls[0]),controller(controls[1])},pairs,controller(opponent),eval_seed,track_projection);
        }, "num_envs"_a,"num_threads"_a,"base_seed"_a=0,"canonical"_a=true,"config"_a=gf::MatchConfig{},"history_len"_a=64,
           "controllers"_a=std::array<std::string,2>{"external","external"},"eval_pairs"_a=nb::none(),"opponent"_a="greedy","eval_seed"_a=0,"track_projection"_a=false)
        .def_ro("track_projection", &VecEnv::track_projection)
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
    cls.def("determinize_support",[](const VecEnv& e,Rows rows) {
        const auto values=e.determinize_support({rows.data(),rows.shape(0)});return int_array(values);
    },"rows"_a.noconvert());
    cls.def("unused_assignment",[](const VecEnv& e,std::size_t row,Rows assignment) {
        const auto counts=e.unused_assignment(row,{assignment.data(),assignment.shape(0)});return int_array(counts);
    },"row"_a,"assignment"_a.noconvert());
    cls.def("load_worlds",[](VecEnv& e,const VecEnv& src,Rows rows,Assignments assignments,Seeds seeds) {
        if(assignments.shape(0)!=e.num_envs || assignments.shape(1)!=297)throw std::invalid_argument("assignments must have shape [num_envs,297]");
        const std::vector<std::int32_t> r(rows.data(),rows.data()+rows.shape(0));
        const std::vector<std::int32_t> a(assignments.data(),assignments.data()+assignments.size());
        const std::vector<std::uint64_t> sd(seeds.data(),seeds.data()+seeds.shape(0));
        {nb::gil_scoped_release release;e.load_worlds(src,r,a,sd);}
        return view(e,e.world_status.data(),{e.num_envs});
    },"src_env"_a,"src_rows"_a.noconvert(),"assignments"_a.noconvert(),"seeds"_a.noconvert());
    cls.def("next_actor_seats",[](const VecEnv& e,Rows rows) {
        return int_array(e.next_actor_seats({rows.data(),rows.shape(0)}));
    },"rows"_a.noconvert());
    cls.def("observe",[](VecEnv& e,Rows rows,nb::object seat) {
        const std::vector<std::int32_t> r(rows.data(),rows.data()+rows.shape(0));
        std::vector<std::int32_t> seats;
        if(nb::isinstance<nb::int_>(seat)) seats.assign(r.size(),nb::cast<int>(seat));
        else {auto a=nb::cast<Rows>(seat);seats.assign(a.data(),a.data()+a.shape(0));}
        {nb::gil_scoped_release release;e.observe(r,seats);}
        const auto n=r.size();nb::dict out;
        out["obs_type"]=view(e,e.leaf_obs_type.data(),{n,e.tokens});
        out["obs_aux_card"]=view(e,e.leaf_obs_aux_card.data(),{n,e.tokens});
        out["obs_valid"]=view(e,e.leaf_obs_valid.data(),{n,e.tokens});
        out["obs_cards"]=view(e,e.leaf_obs_cards.data(),{n,e.tokens,8});
        out["obs_cat"]=view(e,e.leaf_obs_cat.data(),{n,e.tokens,C});
        out["obs_num"]=view(e,e.leaf_obs_num.data(),{n,e.tokens,F});
        return out;
    },"rows"_a.noconvert(),"seat"_a);
    nb::class_<SearchEnv,VecEnv>(m,"SearchEnv")
        .def(nb::init<std::size_t,std::size_t,std::size_t,std::size_t,bool>(),
             "num_envs"_a,"num_threads"_a,"history_len"_a=64,"max_decisions"_a=512,"canonical"_a=true);
    m.attr("WORLD_OK")=int(WorldStatus::ok);
    m.attr("WORLD_INVALID_ASSIGNMENT")=int(WorldStatus::invalid_assignment);
    m.attr("WORLD_UNSUPPORTED_ROOT")=int(WorldStatus::unsupported_root);
    m.attr("NOT_DECISION_START")=-1;
#define ONE(name) cls.def_prop_ro(#name, [](VecEnv& e) { return view(e, e.name.data(), {e.num_envs}); });
    ONE(stage) ONE(actor_seat) ONE(chosen_kind) ONE(chosen_count) ONE(chosen_target) ONE(chosen_alloc_hp) ONE(substep)
    ONE(decision_done) ONE(episode_done) ONE(truncated) ONE(episode_id)
    ONE(decisions) ONE(world_status) ONE(searcher_seat) ONE(result)
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
