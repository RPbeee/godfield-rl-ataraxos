#include "vec_env.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
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
nb::dict stats_dict(const VecEnv& env) {
    const auto s = env.stats();
    nb::dict d;
    d["episodes"] = s.episodes; d["decisions"] = s.decisions;
    d["policy_stages"] = s.policy_stages; d["automatic_decisions"] = s.automatic_decisions;
    d["truncations"] = s.truncations; d["confusion_decisions"] = s.confusion_decisions;
    d["grant_decisions"] = s.grant_decisions;
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
        .def(nb::init<std::size_t, std::size_t, std::uint64_t, bool, gf::MatchConfig>(),
            "num_envs"_a, "num_threads"_a, "base_seed"_a, "canonical"_a = true, "config"_a = gf::MatchConfig{})
        .def_prop_ro("num_envs", [](const VecEnv& e) { return e.num_envs; })
        .def_prop_ro("num_threads", [](const VecEnv& e) { return e.num_threads; })
        .def("step", [](VecEnv& env, nb::ndarray<nb::numpy, const std::int32_t, nb::ndim<1>, nb::c_contig> choices) {
            if (choices.shape(0) != env.num_envs) throw std::invalid_argument("choices must have shape [num_envs]");
            const std::vector<std::int32_t> stable(choices.data(), choices.data() + choices.shape(0));
            nb::gil_scoped_release release;
            env.step(stable);
        }, "choices"_a.noconvert())
        .def("stats", &stats_dict);
#define ONE(name) cls.def_prop_ro(#name, [](VecEnv& e) { return view(e, e.name.data(), {e.num_envs}); });
    ONE(stage) ONE(actor_seat) ONE(chosen_kind) ONE(chosen_count) ONE(substep)
    ONE(decision_done) ONE(episode_done) ONE(truncated) ONE(episode_id)
#undef ONE
    cls.def_prop_ro("mask", [](VecEnv& e) { return view(e, e.mask.data()->data(), {e.num_envs, 21}); });
    cls.def_prop_ro("chosen_cards", [](VecEnv& e) { return view(e, e.chosen_cards.data(), {e.num_envs, 18}); });
    cls.def_prop_ro("outcome", [](VecEnv& e) { return view(e, e.outcome.data(), {e.num_envs, 2}); });
}
