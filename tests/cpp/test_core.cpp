#include "gfrl_core/vec_env.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

using namespace gfrl;
void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}
namespace gfrl {
struct TestAccess {
    // Test-only fixture injection: the public constructor never accepts nondefault config.
    static void inject(VecEnv& env, gf::Match match) {
        env.envs_[0].match = std::move(match);
        env.envs_[0].selection.reset();
        env.envs_[0].encoded=false;
        for(auto& h:env.envs_[0].history) h.clear();
        env.decision_done[0] = env.episode_done[0] = env.truncated[0] = false;
        env.drive(0);
    }
};
}
std::string key(const gf::Command& c) {
    std::ostringstream out;
    out << c.actor << ':' << int(c.kind) << ':';
    for (auto t : c.items) out << t << ',';
    out << ':'; if (c.target) out << *c.target;
    if (c.allocation) out << ':' << c.allocation->hp << ',' << c.allocation->mp << ',' << c.allocation->cp;
    return out.str();
}
using Commands = std::set<std::string>;
void selection_tree(const gf::Match& match, Selection node, bool canonical, Commands& out) {
    if (node.complete) {
        check(match.legal(node.command), "selection leaf must be legal");
        check(out.insert(key(node.command)).second, "duplicate selection leaf");
        return;
    }
    for (int c = 0; c < 21; ++c) if (node.mask[c]) {
        auto next = node; next.choose(match, c, canonical);
        selection_tree(match, std::move(next), canonical, out);
    }
}
void extension_tree(const gf::Match& match, gf::Command& c, bool canonical, Commands& out) {
    auto node = match.legal_extensions(c.kind, c.items, {canonical});
    const auto& hand = match.snapshot().state.players.at(match.snapshot().state.players[0].id == c.actor ? 0 : 1).items;
    bool danger=false, wide=false;
    for (auto token:c.items) for (const auto& item:hand) if (item.token==token) {
        danger |= item.fake_model.value_or(item.model)==94;
        wide |= item.fake_model.value_or(item.model)==232;
    }
    if (danger && wide) return;
    if (node.can_finish) for (auto target : node.targets) {
        c.target = target;
        if (node.exchange) {
            for (int hb = 0; hb <= 20; ++hb) for (int mb = 0; mb <= 20; ++mb) {
                // Independent oracle for the 5-unit grid including 99.
                const int hp = hb == 20 ? 99 : hb * 5, mp = mb == 20 ? 99 : mb * 5;
                const auto cp = node.exchange->total - hp - mp;
                if (cp < 0 || cp > 99) continue;
                c.allocation = gf::ResourceAllocation{hp, mp, static_cast<int>(cp)};
                check(match.legal(c), "engine tree leaf must be legal"); out.insert(key(c));
            }
            c.allocation.reset();
        } else { check(match.legal(c), "engine leaf must be legal"); out.insert(key(c)); }
    }
    for (auto t : node.next) {
        c.items.push_back(t); extension_tree(match, c, canonical, out); c.items.pop_back();
    }
}
std::size_t tree_positions = 0, tree_commands = 0;
std::array<std::size_t, 5> tree_phases{};
void compare_tree(const gf::Match& match) {
    const auto before = match.snapshot();
    ++tree_phases.at(static_cast<int>(before.phase));
    for (bool canonical : {true, false}) {
        Commands actual, expected;
        selection_tree(match, Selection(match, canonical), canonical, actual);
        for (auto kind : match.legal_kinds()) {
            gf::Command c; c.actor = *match.snapshot().actor; c.kind = kind;
            extension_tree(match, c, canonical, expected);
        }
        check(actual == expected, "stage tree and legal_extensions tree differ");
        ++tree_positions; tree_commands += actual.size();
    }
    check(match.snapshot() == before, "tree traversal mutated match");
}
gf::Match fixture(std::initializer_list<int> models, int hp = 40, int mp = 10, int cp = 20) {
    gf::Match match;
    auto s = match.snapshot().state;
    s.players[0].id = 42; s.players[1].id = 77;
    s.players[0].items.clear(); s.players[1].items.clear();
    s.players[0].hp = hp; s.players[0].mp = mp; s.players[0].cp = cp;
    gf::ItemToken token = 100;
    for (auto model : models) s.players[0].items.push_back({token++, static_cast<godfield::official::CardId>(model), {}, false});
    match.load_scene(s, 42, 123); return match;
}
void restricted_pair() {
    for (bool canonical:{false,true}) for (bool dream:{false,true}) for (bool reverse:{false,true}) {
        auto absent=fixture(reverse?std::initializer_list<int>{232,94,6}:std::initializer_list<int>{94,232,6});
        auto state=absent.snapshot().state;
        if (dream) {
            state.players[0].curses={gf::Curse::dream};
            // Restore permits identity disguises, even for unique Dream classes.
            for (auto& item:state.players[0].items) item.fake_model=item.model;
        }
        state.players[1].items={{200,19,{},false}};
        absent.load_scene(state,42,123);
        state.players[1].items[0].model=209;
        auto present=absent;present.load_scene(state,42,123);
        gf::PublicSelection public_match(absent.view(42));
        const auto visit=[&](auto&& self,Selection a,Selection b,Selection p)->void {
            const auto public_choice=[&](int choice) {
                if(a.stage!=Stage::card || choice>=3)return choice;
                const auto token=absent.snapshot().state.players[0].items[choice].token;
                const auto& hand=public_match.snapshot().state.players[0].items;
                return int(std::find_if(hand.begin(),hand.end(),[&](const auto& i){return i.token==token;})-hand.begin());
            };
            check(a.mask==b.mask,"opponent mortar changed restricted mask");
            for(int choice=0;choice<21;++choice)
                check(a.mask[choice]==p.mask[public_choice(choice)],"public cursor changed restricted mask");
            if (a.complete) {
                bool danger=false,wide=false;
                for (int i=0;i<a.count;++i) {
                    const auto& item=absent.snapshot().state.players[0].items[a.cards[i]];
                    danger |= item.fake_model.value_or(item.model)==94;
                    wide |= item.fake_model.value_or(item.model)==232;
                }
                check(!(danger && wide),"forbidden pair reached leaf");
                return;
            }
            for (int choice=0;choice<21;++choice) if(a.mask[choice]) {
                auto aa=a,bb=b,pp=p;
                aa.choose(absent,choice,canonical);bb.choose(present,choice,canonical);pp.choose(public_match,public_choice(choice),canonical);
                self(self,aa,bb,pp);
            }
        };
        visit(visit,Selection(absent,canonical),Selection(present,canonical),Selection(public_match,canonical));
        compare_tree(absent);compare_tree(present);
        // The engine's rules remain unchanged: it still allows the pair here.
        const gf::ItemToken danger=reverse?101:100,wide=reverse?100:101;
        check(absent.legal({42,gf::CommandKind::play,{danger,wide},{}}),"engine pair rule changed");
        for (const auto& tokens:{std::array{danger,wide},std::array{wide,danger}})
            check(forbidden_pair(absent.snapshot(),tokens),"pair check depends on order");
        Selection cursor(absent,canonical);
        cursor.choose(absent,int(gf::CommandKind::play),canonical);
        check(!cursor.mask[reverse?0:1],"wide card unexpectedly became a base");
        cursor.choose(absent,reverse?1:0,canonical);
        check(!cursor.mask[reverse?0:1],"forbidden extra admitted");
        bool rejected=false;
        try {cursor.choose(absent,reverse?0:1,canonical);} catch(const std::invalid_argument&) {rejected=true;}
        check(rejected,"forbidden choice accepted");
    }
}
void tree() {
    restricted_pair();
    std::uint64_t policy = 42, chance = 43;
    for (int i = 0; i < 120; ++i) {
        gf::Match game;
        const std::array<gf::PlayerSetup, 2> players{{{42, gf::Team::solo, "A"}, {77, gf::Team::solo, "B"}}};
        game.reset(players, i + 1);
        for (int j = 0; j < i % 20 && !game.done(); ++j) game.rollout_step(policy, chance);
        if (game.done()) game.reset(players, i + 1000);
        if (game.snapshot().phase == gf::MatchPhase::main) {
            auto s = game.snapshot().state;
            for (auto& p : s.players) { p.items.resize(std::min<std::size_t>(4, p.items.size())); p.confusion_count = 0; }
            game.load_scene(s, game.snapshot().owner, i + 500);
        }
        compare_tree(game);
    }
    // Explicit empty hand, duplicates, exchange extrema, FREE and order-sensitive materials.
    for (auto models : {std::initializer_list<int>{}, {6, 6, 7, 8}, {3}, {3, 3}, {57, 204, 199, 200}, {204, 199, 200, 204, 199}, {120, 26, 14, 58}, {94, 232, 6}, {232, 94}})
        compare_tree(fixture(models));
    compare_tree(fixture({3}, 99, 99, 99));
    compare_tree(fixture({3}, 1, 0, 0));
    auto purchase = fixture({5});
    auto state = purchase.snapshot().state;
    state.players[1].items = {{200, 6, {}, false}};
    purchase.load_scene(state, 42, 5);
    purchase.step({42, gf::CommandKind::buy, {100}, 77});
    if (purchase.snapshot().phase == gf::MatchPhase::response)
        purchase.step({*purchase.snapshot().actor, gf::CommandKind::defend, {}, {}});
    check(purchase.snapshot().phase == gf::MatchPhase::purchase, "purchase fixture missing");
    compare_tree(purchase);
    check(tree_phases[0] && tree_phases[1] && tree_phases[3], "missing main/response/purchase tree coverage");
    std::cout << "main=" << tree_phases[0] << " response=" << tree_phases[1] << " purchase=" << tree_phases[3] << ' ';
    std::cout << "tree_positions=" << tree_positions << " leaves=" << tree_commands << '\n';
}
template<class T> void equal(const Buffer<T>& a, const Buffer<T>& b, const char* name) {
    check(a.size == b.size && std::equal(a.data(), a.data() + a.size, b.data()), std::string("buffer differs: ") + name);
}
void equal_env(const VecEnv& a, const VecEnv& b) {
#define EQ(name) equal(a.name, b.name, #name)
    EQ(stage); EQ(actor_seat); EQ(mask); EQ(chosen_kind); EQ(chosen_cards); EQ(chosen_count); EQ(substep);
    EQ(chosen_target); EQ(chosen_alloc_hp); EQ(decisions); EQ(world_status); EQ(searcher_seat); EQ(result);
    EQ(decision_done); EQ(episode_done); EQ(outcome); EQ(truncated); EQ(episode_id);
    EQ(obs_type); EQ(obs_cards); EQ(obs_aux_card); EQ(obs_cat); EQ(obs_num); EQ(obs_valid);
    EQ(label_opp_hand); EQ(label_opp_hand_count);
    EQ(label_opp_unused_hand); EQ(label_opp_unused_hand_count);
    check(std::memcmp(a.obs_num.data(), b.obs_num.data(), a.obs_num.size*sizeof(float))==0, "obs_num bits differ");
#undef EQ
    for (std::size_t i = 0; i < a.num_envs; ++i) check(a.match(i).snapshot() == b.match(i).snapshot(), "snapshot differs");
    // Privileged snapshot oracle: labels and observations must describe the
    // same pre-action state, including continuation stages and episode resets.
    for (std::size_t i = 0; i < a.num_envs; ++i) {
        const auto& snap = a.match(i).snapshot();
        const auto seat = a.actor_seat[i];
        std::array<std::uint8_t, 297> expected{};
        const auto& hand = snap.state.players[1-seat].items;
        for (const auto& item : hand) ++expected[card_code(item.model)];
        check(std::equal(expected.begin(), expected.end(), a.label_opp_hand.data()+297*i), "opponent label time/model mismatch");
        check(a.label_opp_hand_count[i] == hand.size(), "opponent label count mismatch");
        expected.fill(0);
        std::size_t unused=0;
        for (const auto& item : hand) if(!item.used) {++expected[card_code(item.model)];++unused;}
        check(std::equal(expected.begin(), expected.end(), a.label_opp_unused_hand.data()+297*i), "unused opponent label mismatch");
        check(a.label_opp_unused_hand_count[i] == unused, "unused opponent count mismatch");
        check(a.obs_num[i*a.tokens*F+Decisions]*512 == snap.decisions, "observation time mismatch");
    }
    auto x = a.stats(), y = b.stats();
    check(x.episodes == y.episodes && x.decisions == y.decisions && x.policy_stages == y.policy_stages &&
          x.automatic_decisions == y.automatic_decisions && x.truncations == y.truncations, "stats differ");
}
void determinism() {
    for (bool canonical : {true, false}) {
        VecEnv a(32, 1, 67, canonical), b(32, 2, 67, canonical), c(32, 8, 67, canonical), d(32, 1, 67, canonical);
        std::mt19937_64 rng(8);
        for (int i = 0; i < 600; ++i) {
            equal_env(a, b); equal_env(a, c); equal_env(a, d);
            auto choices = random_choices({a.mask.data(), a.num_envs}, {a.stage.data(), a.num_envs}, rng);
            a.step(choices); b.step(choices); c.step(choices); d.step(choices);
        }
        equal_env(a, b); equal_env(a, c); equal_env(a, d);
    }
    std::cout << "determinism: 2 canonical modes x 600 steps x 32 envs, threads=1/2/8\n";
}
void legality() {
    for (bool canonical : {true, false}) {
        VecEnv env(128, 4, 543, canonical);
        std::mt19937_64 rng(99);
        while (env.stats().episodes < 2000) {
            auto choices = random_choices({env.mask.data(), env.num_envs}, {env.stage.data(), env.num_envs}, rng);
            env.step(choices);
            for (std::size_t i = 0; i < env.num_envs; ++i) {
                check(std::count(env.mask[i].begin(), env.mask[i].end(), true) >= 2, "forced stage exposed");
                if (env.episode_done[i]) check(env.outcome[2*i] + env.outcome[2*i+1] == 0, "outcomes must be zero-sum");
            }
        }
        auto s = env.stats();
        std::cout << "canonical=" << canonical << " episodes=" << s.episodes << " decisions=" << s.decisions
            << " policy_stages=" << s.policy_stages << " automatic=" << s.automatic_decisions
            << " history_dropped=" << s.history_dropped << " card_overflow=" << s.card_overflow << " used_overflow=" << s.used_overflow << " token_overflow=" << s.token_overflow
            << " confusion=" << s.confusion_decisions << " grant=" << s.grant_decisions << " truncated=" << s.truncations << '\n';
    }
}
void invalid() {
    VecEnv env(8, 8, 11), reference(8, 1, 11);
    std::mt19937_64 rng(12);
    for (int position = 0; position < 8; ++position) for (int value : {-2, 21, 100000}) {
        auto choices = random_choices({env.mask.data(), env.num_envs}, {env.stage.data(), env.num_envs}, rng);
        choices[position] = value;
        bool thrown = false;
        try { env.step(choices); } catch (const std::invalid_argument& error) {
            thrown = true; check(std::string(error.what()).find("env " + std::to_string(position)) != std::string::npos, "missing env context");
        }
        check(thrown, "invalid choice accepted"); equal_env(env, reference);
    }
    for (std::size_t i = 0; i < env.num_envs; ++i) {
        auto choices = random_choices({env.mask.data(), env.num_envs}, {env.stage.data(), env.num_envs}, rng);
        choices[i] = static_cast<int>(std::find(env.mask[i].begin(), env.mask[i].end(), false) - env.mask[i].begin());
        try { env.step(choices); check(false, "masked choice accepted"); } catch (const std::invalid_argument&) {}
        equal_env(env, reference);
    }
    std::vector<std::int32_t> short_input(7);
    try { env.step(short_input); check(false, "wrong shape accepted"); } catch (const std::invalid_argument&) {}
    auto valid = random_choices({env.mask.data(), env.num_envs}, {env.stage.data(), env.num_envs}, rng);
    env.step(valid); reference.step(valid); equal_env(env, reference);
    for (int field = 0; field < 10; ++field) {
        gf::MatchConfig config;
        switch (field) {
        case 0: config.max_decisions = 10; break;
        case 1: config.initial_hand = 1; break;
        case 2: config.hand_limit = 17; break;
        case 3: config.initial_hp = 30; break;
        case 4: config.initial_mp = 20; break;
        case 5: config.initial_cp = 30; break;
        case 6: config.include_fate = true; break;
        case 7: config.external_grants = true; break;
        case 8: config.external_artifacts = true; break;
        case 9: config.local_guardian_payloads = true; break;
        }
        try { VecEnv rejected(1, 1, 1, true, config); check(false, "unsupported config accepted"); }
        catch (const std::invalid_argument&) {}
    }
    auto oversized = fixture({6}); auto state = oversized.snapshot().state;
    for (int j = 0; j < 18; ++j) state.players[0].items.push_back({std::uint64_t(200 + j), 6, {}, false});
    // The engine itself rejects >18. Corrupt a test-only snapshot to exercise our defensive check.
    const_cast<gf::MatchSnapshot&>(oversized.snapshot()).state = state;
    try { Selection rejected(oversized); check(false, "oversized hand accepted"); }
    catch (const std::runtime_error& e) { check(std::string(e.what()).find("18") != std::string::npos, "wrong hand error"); }
}
void freeze() {
    VecEnv env(1, 2, 67), reference(1, 1, 67);
    std::mt19937_64 rng(8);
    int frozen = 0, rejected = 0;
    for (int step = 0; step < 600; ++step) {
        const std::vector<std::int32_t> pause{-1};
        if (env.substep[0] == 0) {
            for (int i = 0; i < 3; ++i) env.step(pause);
            ++frozen;
        } else {
            bool thrown = false;
            try { env.step(pause); } catch (const std::invalid_argument&) { thrown = true; }
            check(thrown, "freeze accepted after substep zero");
            ++rejected;
        }
        equal_env(env, reference);
        auto choices = random_choices({env.mask.data(), 1}, {env.stage.data(), 1}, rng);
        env.step(choices); reference.step(choices);
    }
    equal_env(env, reference);
    check(frozen > 10 && rejected > 10, "freeze coverage");
}
void automatic() {
    VecEnv env(1, 1, 123);
    auto confused = fixture({6, 7}); auto state = confused.snapshot().state;
    state.players[0].confusion_count = 3; confused.load_scene(state, 42, 1);
    TestAccess::inject(env, confused);
    check(env.stats().confusion_decisions > 0 && env.stats().policy_stages == 0, "confusion must bypass policy");
    // Grant cannot occur with the supported default config; exercise the driver branch via a test fixture.
    gf::MatchConfig external; external.external_artifacts = true;
    gf::Match grant(external);
    std::uint64_t policy = 5, chance = 6;
    for (int n = 0; n < 1000 && grant.snapshot().phase != gf::MatchPhase::grant; ++n) {
        if (grant.done()) { const std::array<gf::PlayerSetup, 2> p{{{0, gf::Team::solo, "A"}, {1, gf::Team::solo, "B"}}}; grant.reset(p, n + 5); }
        grant.rollout_step(policy, chance);
    }
    check(grant.snapshot().phase == gf::MatchPhase::grant, "no fixture grant");
    TestAccess::inject(env, grant);
    check(env.stats().grant_decisions > 0 && env.stats().policy_stages == 0, "grant must bypass policy");
    auto empty = fixture({});
    const auto before = env.stats().automatic_decisions;
    TestAccess::inject(env, empty);
    check(env.stats().automatic_decisions > before, "all-forced command must execute automatically");
    // Force a valid late-game boundary to cover the 512-decision cutoff and reset.
    auto late = fixture({}); auto snap = late.snapshot(); snap.decisions = 511; late.restore(snap);
    TestAccess::inject(env, late);
    check(env.episode_done[0] && env.truncated[0] && env.outcome[0] == 0 && env.outcome[1] == 0, "cutoff must be a draw");
    check(env.episode_id[0] == 1 && !env.match(0).done(), "cutoff must auto-reset");
    std::cout << "synthetic confusion=" << env.stats().confusion_decisions << " grant=" << env.stats().grant_decisions << "; cutoff/reset covered\n";
}
void allocation() {
    // All possible totals, every legal HP bucket, all MP buckets. This proves no dead end for every resource triple.
    std::size_t hp_cases = 0, pairs = 0;
    for (int total = 0; total <= 297; ++total) {
        int hp_count = 0;
        for (int hb = 0; hb <= 20; ++hb) {
            int hp = bucket_value(hb);
            if (total - hp < 0 || total - hp > 198) continue;
            ++hp_count; ++hp_cases; int mp_count = 0;
            for (int mb = 0; mb <= 20; ++mb) {
                const auto cp = total - hp - bucket_value(mb);
                if (cp >= 0 && cp <= 99) { ++mp_count; ++pairs; }
            }
            check(mp_count > 0, "empty MP allocation domain");
        }
        check(hp_count > 0, "empty HP allocation domain");
        // Verify the actual selection masks too, including unreachable live HP=0 boundaries via positive fixture HP.
        auto game = fixture({3}, std::max(1, std::min(total, 99)),
                            std::clamp(total - 99, 0, 99), std::clamp(total - 198, 0, 99));
        if (total == 0) continue; // HP=0 without revival is terminal; total=0 checked algebraically above.
        Selection sel(game);
        sel.choose(game, int(gf::CommandKind::exchange), true);
        sel.choose(game, 0, true); sel.choose(game, 18, true); sel.choose(game, 0, true);
        for (int hb = 0; hb <= 20; ++hb) {
            check(sel.mask[hb] == (total - bucket_value(hb) >= 0 && total - bucket_value(hb) <= 198), "HP mask mismatch");
            if (!sel.mask[hb]) continue;
            auto mp = sel; mp.choose(game, hb, true);
            for (int mb = 0; mb <= 20; ++mb) {
                const auto cp = total - bucket_value(hb) - bucket_value(mb);
                check(mp.mask[mb] == (cp >= 0 && cp <= 99), "MP mask mismatch");
            }
        }
    }
    std::cout << "allocation totals=298 hp_cases=" << hp_cases << " valid_pairs=" << pairs << '\n';
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "specify test case"); const std::string name = argv[1];
        if (name == "tree") tree(); else if (name == "legality") legality();
        else if (name == "determinism") determinism(); else if (name == "automatic") automatic();
        else if (name == "invalid") invalid(); else if (name == "allocation") allocation();
        else if (name == "freeze") freeze();
        else throw std::runtime_error("unknown test");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
