#pragma once
#include <godfield/official_match.hpp>
#include <array>
#include <random>

namespace gfrl {
namespace gf = godfield::official;
enum class Stage : std::int8_t { kind, card, target, alloc_hp, alloc_mp };
using Mask = std::array<bool, 21>;
static_assert(sizeof(Mask) == 21 * sizeof(bool), "numpy mask rows must be contiguous");
int bucket_value(int bucket);
// A pure selection cursor over one immutable Match. Does not execute Commands.
class Selection {
public:
    Stage stage = Stage::kind;
    Mask mask{};
    gf::Command command;
    std::array<std::int8_t, 18> cards{};
    int actor_seat = 0, count = 0, substep = 0;
    bool complete = false;
    bool kind_chosen = false;
    explicit Selection(const gf::Match& match, bool canonical = true);
    explicit Selection(const gf::PublicSelection& match, bool canonical = true);
    void choose(const gf::Match& match, int choice, bool canonical);
    void choose(const gf::PublicSelection& match, int choice, bool canonical);
private:
    std::int64_t total_ = 0;
    template<class M> void init(const M& match, bool canonical);
    template<class M> void refresh(const M& match, bool canonical);
    template<class M> void select(const M& match, int choice, bool canonical);
};
std::vector<std::int32_t> random_choices(std::span<const Mask> mask,
    std::span<const std::int8_t> stage, std::mt19937_64& rng);
}
