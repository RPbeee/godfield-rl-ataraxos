#include "selection.hpp"
#include <algorithm>

namespace gfrl {
int bucket_value(int b) { return std::min(5 * b, 99); }
Selection::Selection(const gf::Match& match, bool canonical) {
    cards.fill(-1);
    const auto& s = match.snapshot();
    if (!s.actor || s.state.players.size() != 2)
        throw std::invalid_argument("selection requires an actor and two players");
    for (int seat = 0; seat < 2; ++seat) {
        if (s.state.players[seat].items.size() > 18)
            throw std::runtime_error("hand exceeds 18 cards at seat " + std::to_string(seat));
        if (s.state.players[seat].id == *s.actor) actor_seat = seat;
    }
    command.actor = *s.actor;
    refresh(match, canonical);
}
void Selection::refresh(const gf::Match& match, bool canonical) {
    mask.fill(false);
    if (complete) return;
    if (stage == Stage::kind) {
        for (auto k : match.legal_kinds()) mask.at(static_cast<int>(k)) = true;
    } else if (stage == Stage::card || stage == Stage::target) {
        const auto node = match.legal_extensions(command.kind, command.items, {canonical});
        if (stage == Stage::card) {
            const auto& hand = match.snapshot().state.players[actor_seat].items;
            for (std::size_t i = 0; i < hand.size(); ++i)
                mask[i] = std::find(node.next.begin(), node.next.end(), hand[i].token) != node.next.end();
            mask[18] = node.can_finish;
        } else {
            const auto& players = match.snapshot().state.players;
            for (auto target : node.targets) {
                if (!target) mask[0] = true;
                else if (*target == command.actor) mask[1] = true;
                else if (*target == players[1 - actor_seat].id) mask[2] = true;
                else throw std::runtime_error("target outside two-player domain");
            }
            if (node.exchange) total_ = node.exchange->total;
        }
    } else {
        for (int b = 0; b <= 20; ++b) {
            const auto v = bucket_value(b);
            if (stage == Stage::alloc_hp) mask[b] = total_ - v >= 0 && total_ - v <= 198;
            else {
                const auto cp = total_ - command.allocation->hp - v;
                mask[b] = cp >= 0 && cp <= 99;
            }
        }
    }
    if (std::count(mask.begin(), mask.end(), true) == 0)
        throw std::runtime_error("empty legal mask at stage " + std::to_string(static_cast<int>(stage)));
}
void Selection::choose(const gf::Match& match, int choice, bool canonical) {
    if (complete || choice < 0 || choice >= 21 || !mask[choice])
        throw std::invalid_argument("illegal selection choice " + std::to_string(choice));
    switch (stage) {
    case Stage::kind:
        command.kind = static_cast<gf::CommandKind>(choice); kind_chosen = true; stage = Stage::card; break;
    case Stage::card:
        if (choice == 18) stage = Stage::target;
        else {
            if (count == 18) throw std::runtime_error("selected more than 18 cards");
            cards[count++] = static_cast<std::int8_t>(choice);
            command.items.push_back(match.snapshot().state.players[actor_seat].items.at(choice).token);
        }
        break;
    case Stage::target:
        if (choice == 0) command.target.reset();
        else command.target = match.snapshot().state.players[choice == 1 ? actor_seat : 1 - actor_seat].id;
        if (command.kind == gf::CommandKind::exchange) {
            command.allocation = gf::ResourceAllocation{}; stage = Stage::alloc_hp;
        } else complete = true;
        break;
    case Stage::alloc_hp:
        command.allocation->hp = bucket_value(choice); stage = Stage::alloc_mp; break;
    case Stage::alloc_mp:
        command.allocation->mp = bucket_value(choice);
        command.allocation->cp = static_cast<int>(total_ - command.allocation->hp - command.allocation->mp);
        complete = true; break;
    }
    refresh(match, canonical);
}
std::vector<std::int32_t> random_choices(std::span<const Mask> masks,
    std::span<const std::int8_t> stages, std::mt19937_64& rng) {
    if (masks.size() != stages.size()) throw std::invalid_argument("mask/stage length mismatch");
    std::vector<std::int32_t> result(masks.size());
    constexpr int widths[]{10, 19, 3, 21, 21};
    for (std::size_t i = 0; i < masks.size(); ++i) {
        if (stages[i] < 0 || stages[i] > 4) throw std::invalid_argument("invalid stage");
        std::array<int, 21> legal{}; int n = 0;
        for (int c = 0; c < widths[stages[i]]; ++c) if (masks[i][c]) legal[n++] = c;
        if (!n) throw std::invalid_argument("empty random policy mask");
        result[i] = legal[std::uniform_int_distribution<int>(0, n - 1)(rng)];
    }
    return result;
}
}
