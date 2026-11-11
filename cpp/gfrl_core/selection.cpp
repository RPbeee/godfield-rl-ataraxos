#include "selection.hpp"
#include <algorithm>

namespace gfrl {
bool forbidden_pair(const gf::MatchSnapshot& s, std::span<const gf::ItemToken> tokens) {
    const auto& own = *std::find_if(s.state.players.begin(), s.state.players.end(),
        [&](const auto& p) { return p.id == s.actor; });
    bool danger = false, wide = false;
    for (auto token : tokens) {
        const auto& item = *std::find_if(own.items.begin(), own.items.end(),
            [&](const auto& i) { return i.token == token; });
        const auto face = item.fake_model.value_or(item.model);
        danger |= face == 94;
        wide |= face == 232;
    }
    return danger && wide;
}
int bucket_value(int b) { return std::min(5 * b, 99); }
Selection::Selection(const gf::Match& m,bool c,SelectionExclusions x,
        std::shared_ptr<const gf::PublicSelection> check):exclusions(std::move(x)),public_root(std::move(check)) {init(m,c);}
Selection::Selection(const gf::PublicSelection& m,bool c,SelectionExclusions x):exclusions(std::move(x)) {init(m,c);}
template<class M> void Selection::init(const M& match, bool canonical) {
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
    if constexpr(std::is_same_v<M,gf::Match>) {
        if(public_root) {
            const auto& actual=match.snapshot().state.players[actor_seat].items;
            const auto& expected=public_root->snapshot().state.players[actor_seat].items;
            if(actual.size()!=expected.size())throw std::invalid_argument("restricted_root_slot_mismatch");
            for(std::size_t i=0;i<actual.size();++i)
                if(actual[i].token!=expected[i].token || actual[i].fake_model.value_or(actual[i].model)!=expected[i].fake_model.value_or(expected[i].model)
                        || actual[i].used!=expected[i].used || actual[i].fake_model.has_value()!=expected[i].fake_model.has_value())
                    throw std::invalid_argument("restricted_root_slot_mismatch");
        }
    }
    refresh(match, canonical);
}
template<class M> bool Selection::completion(const M& position,gf::CommandKind kind,std::vector<gf::ItemToken> prefix) const {
    const auto node=position.legal_extensions(kind,prefix,{exclusions.tokens.empty()});
    if(node.can_finish)return true;
    for(auto token:node.next)if(std::find(exclusions.tokens.begin(),exclusions.tokens.end(),token)==exclusions.tokens.end()) {
        auto next=prefix;next.push_back(token);
        if(completion(position,kind,std::move(next)))return true;
    }
    return false;
}
template<class M> void Selection::prune(const M& position) {
    if(complete)return;
    auto& s=*this;
    if(exclusions.enemy_default_self && s.stage==Stage::target && !s.command.items.empty()
        && (s.command.kind==gf::CommandKind::play || s.command.kind==gf::CommandKind::sell
            || s.command.kind==gf::CommandKind::buy)) {
        const auto& hand=position.snapshot().state.players[s.actor_seat].items;
        const auto base=std::find_if(hand.begin(),hand.end(),[&](const auto& h){return h.token==s.command.items.front();});
        if(base!=hand.end() && gf::targets_enemy_by_default(base->fake_model.value_or(base->model)))s.mask[1]=false;
    }
    if(s.stage==Stage::kind) {
        for(auto kind:exclusions.kinds)if(kind>=0 && kind<10)s.mask[kind]=false;
        for(int i=0;i<10;++i)if(!exclusions.tokens.empty() && s.mask[i] && !completion(position,static_cast<gf::CommandKind>(i),{}))s.mask[i]=false;
    } else if(!exclusions.tokens.empty() && s.stage==Stage::card) {
        const auto& hand=position.snapshot().state.players[s.actor_seat].items;
        for(std::size_t i=0;i<hand.size();++i)if(s.mask[i]) {
            auto prefix=s.command.items;prefix.push_back(hand[i].token);
            if(std::find(exclusions.tokens.begin(),exclusions.tokens.end(),hand[i].token)!=exclusions.tokens.end()
                || !completion(position,s.command.kind,std::move(prefix)))s.mask[i]=false;
        }
    }
    if(std::count(s.mask.begin(),s.mask.end(),true)==0)
        throw std::invalid_argument("no supported public selection after exclusions");
}

template<class M> void Selection::refresh(const M& match, bool canonical) {
    canonical = canonical && exclusions.tokens.empty();
    mask.fill(false);
    if (complete) return;
    if (stage == Stage::kind) {
        for (auto k : match.legal_kinds()) mask.at(static_cast<int>(k)) = true;
    } else if (stage == Stage::card || stage == Stage::target) {
        const auto node = policy_extensions(match, command.kind, command.items, canonical);
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
    try {prune(match);}
    catch(const std::invalid_argument&) {
        if constexpr(std::is_same_v<M,gf::Match>) {
            if(public_root)throw std::invalid_argument("restricted_root_world_mismatch");
        }
        throw;
    }
    if constexpr(std::is_same_v<M,gf::Match>) {
        if(public_root) {
            auto expected=*this;
            expected.refresh(*public_root,canonical);
            if(mask!=expected.mask)throw std::invalid_argument("restricted_root_world_mismatch");
        }
    }
    if (std::count(mask.begin(), mask.end(), true) == 0)
        throw std::runtime_error("empty legal mask at stage " + std::to_string(static_cast<int>(stage)));
}
void Selection::choose(const gf::Match& m,int c,bool canon) {select(m,c,canon);}
void Selection::choose(const gf::PublicSelection& m,int c,bool canon) {select(m,c,canon);}
template<class M> void Selection::select(const M& match, int choice, bool canonical) {
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
