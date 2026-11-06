#pragma once
#include "selection.hpp"
#include <array>
#include <string_view>
namespace gfrl {
// Every feature below is derived from public views. Tokens used only to align own slots.
enum Type : std::int8_t { Empty, Cls, Player, Hand, Pending, Offer, Used, Decision,
    Reveal, Removal, Sale, Purchase, Danger, Devil, Grant, Fate, Phenomenon, GuardianChange, Redraw };
enum Cat { Actor, Target, Origin, SourceSeat, Phase, Source, Kind, Category, Element, Guardian,
    Ability, Reason, GuardianBefore, GuardianAfter, C };
enum Num { Hp, Mp, Cp, StatusHidden, Dead, Fogged, Confused, UsedOverflow,
    Curse0, CurseEnd=Curse0+9, Gf=CurseEnd, Tiebreak, TiebreakHidden, Decisions, Apocalypse,
    FirstSeat, HandCount, Disguised, IsUsed, Power, Amount, HitRate, Reflections, Bounces,
    Counter, Environmental, Death, ForcedHit, ForcedHidden, Price, FacesHidden, HasAllocation,
    AllocationHp, AllocationMp, AllocationCp, Age, Mortar, CountOverflow,
    Delta0, DeltaEnd=Delta0+36, F=DeltaEnd };
inline constexpr std::array<std::string_view,C> cat_names{"actor","target","origin","source_seat","phase","source","kind","category","element","guardian","ability","reason","guardian_before","guardian_after"};
inline constexpr std::array<int,C> cat_vocab{3,3,3,3,6,15,11,9,7,12,62,7,12,12};
std::vector<std::string> num_names();
struct Token {
    std::int8_t type=0;
    std::array<std::int16_t,8> cards{};
    std::int16_t aux=0;
    std::array<std::int8_t,C> cat{};
    std::array<float,F> num{};
    bool operator==(const Token&) const = default;
};
inline constexpr std::size_t history_offset=55; // CLS + players + hand + pending + offer + 2*16 used
std::int16_t card_code(gf::CardId id);
int relative(gf::PlayerId id, gf::PlayerId viewer);
void cards(Token& t, std::span<const gf::CardId> ids);
std::vector<Token> encode(const gf::MatchView& view, gf::PlayerId viewer,
    std::span<const gf::ItemToken> own_order);
inline constexpr int card_feature_count=8+6+1+3+61+1+9+11+3;
const std::vector<float>& card_features();
}
