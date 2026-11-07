#include "vec_env.hpp"
#include <algorithm>

namespace gfrl {
namespace {
std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
std::size_t positive(std::size_t n, const char* name) {
    if (!n || n > 1000000) throw std::invalid_argument(std::string(name) + " must be in 1..1000000");
    return n;
}
}
Workers::Workers(std::size_t n) {
    if (n == 1) return;
    try {
        for (std::size_t id = 0; id < n; ++id) threads_.emplace_back([this, id] {
            std::size_t seen = 0;
            std::unique_lock lock(mutex_);
            for (;;) {
                start_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                lock.unlock();
                try { job_(id); }
                catch (...) {
                    std::lock_guard guard(mutex_);
                    if (!error_) error_ = std::current_exception();
                }
                lock.lock();
                if (--remaining_ == 0) finish_.notify_one();
            }
        });
    } catch (...) {
        { std::lock_guard lock(mutex_); stop_ = true; }
        start_.notify_all();
        for (auto& t : threads_) t.join();
        throw;
    }
}
Workers::~Workers() {
    { std::lock_guard lock(mutex_); stop_ = true; }
    start_.notify_all();
    for (auto& t : threads_) t.join();
}
void Workers::run(std::function<void(std::size_t)> job) {
    if (threads_.empty()) { job(0); return; }
    std::unique_lock lock(mutex_);
    job_ = std::move(job); error_ = nullptr; remaining_ = threads_.size(); ++generation_;
    start_.notify_all();
    finish_.wait(lock, [&] { return remaining_ == 0; });
    job_ = {};
    if (error_) std::rethrow_exception(error_);
}
VecEnv::VecEnv(std::size_t n, std::size_t threads, std::uint64_t seed, bool canonical, gf::MatchConfig config, std::size_t k, std::array<Controller,2> controllers, std::optional<std::size_t> eval_pairs, Controller opponent, std::uint64_t eval_seed)
    : num_envs(positive(n, "num_envs")), num_threads(positive(threads, "num_threads")),
      history_len(k<=4096 ? k : throw std::invalid_argument("history_len must be <=4096")), tokens(history_offset + history_len),
      obs_type(n*tokens), obs_cat(n*tokens*C), obs_cards(n*tokens*8), obs_aux_card(n*tokens),
      obs_num(n*tokens*F), obs_valid(n*tokens), label_opp_hand(n*297), label_opp_hand_count(n),
      stage(n), actor_seat(n), chosen_kind(n), chosen_cards(n * 18), chosen_count(n), chosen_target(n), chosen_alloc_hp(n), outcome(n * 2),
      mask(n), substep(n), decision_done(n), episode_done(n), truncated(n), episode_id(n),
      base_seed_(seed), canonical_(canonical), controllers_(controllers), eval_pairs_(eval_pairs), opponent_(opponent), eval_seed_(eval_seed), workers_(threads) {
    if (config != gf::MatchConfig{}) throw std::invalid_argument("VecEnv supports only default MatchConfig (max_decisions=512, include_fate/external_*=false)");
    if(eval_pairs_ && *eval_pairs_>10000000)throw std::invalid_argument("invalid evaluation configuration");
    envs_.resize(n);
    for(std::size_t i=0;i<n;++i)envs_[i].game=i;
    for(auto& e:envs_) for(auto& h:e.history) h=History(k);
    workers_.run([&](std::size_t worker) {
        for (std::size_t i = worker; i < n; i += num_threads) {
            reset(i); drive(i);
            decision_done[i] = episode_done[i] = truncated[i] = false;
            outcome[2 * i] = outcome[2 * i + 1] = 0;
        }
    });
}
void VecEnv::reset(std::size_t i) {
    auto& e = envs_[i];
    if(eval_pairs_ && e.game >= 2 * *eval_pairs_) {
        e.waiting=true;e.selection.reset();stage[i]=-1;actor_seat[i]=-1;mask[i].fill(false);
        chosen_kind[i]=-1;chosen_target[i]=-1;chosen_alloc_hp[i]=-1;chosen_count[i]=0;substep[i]=0;std::fill_n(chosen_cards.data()+18*i,18,-1);
        return;
    }
    const auto seed = mix(base_seed_ ^ mix(i) ^ mix(e.episode + 0x243f6a8885a308d3ULL));
    e.chance_rng = mix(seed ^ 0xa4093822299f31d0ULL);
    const std::array<gf::PlayerSetup, 2> players{{{0, gf::Team::solo, "A"}, {1, gf::Team::solo, "B"}}};
    e.match.reset(players, mix(seed ^ 0x13198a2e03707344ULL));
    for(int seat=0;seat<2;++seat)e.policy_rng[seat]=mix(seed ^ mix(seat+0x504f4c494359ULL));
    if(eval_pairs_)reset_evaluation(e.match,eval_seed_,e.game,e.chance_rng,e.policy_rng);
    e.selection.reset(); e.encoded=false; for(auto& h:e.history) h.clear(); episode_id[i] = eval_pairs_ ? e.game : e.episode;
}
void VecEnv::before_decision(std::size_t i) {
    auto& e=envs_[i];
    for(int seat=0;seat<2;++seat) e.before[seat]=e.match.view(e.match.snapshot().state.players[seat].id);
}
void VecEnv::encode_observation(std::size_t i) {
    auto& e=envs_[i]; const auto seat=e.selection->actor_seat;
    // Privileged access is restricted to own slot tokens and the separate labels.
    const auto& players=e.match.snapshot().state.players;
    std::vector<gf::ItemToken> order;
    for(const auto& item:players[seat].items) order.push_back(item.token);
    const auto v=e.match.view(players[seat].id);
    auto ts=encode(v,players[seat].id,order);
    ts.resize(tokens);
    std::size_t j=history_offset;
    for(const auto& event:e.history[seat].events()) {
        ts[j]=event.token;
        ts[j++].num[Age]=history_len ? float(v.decisions-event.decision)/history_len : 0;
    }
    for(j=0;j<tokens;++j) {
        const auto& t=ts[j];auto row=i*tokens+j;
        obs_type[row]=t.type;obs_aux_card[row]=t.aux;obs_valid[row]=t.type!=0;
        std::copy(t.cards.begin(),t.cards.end(),obs_cards.data()+row*8);
        std::copy(t.cat.begin(),t.cat.end(),obs_cat.data()+row*C);
        std::copy(t.num.begin(),t.num.end(),obs_num.data()+row*F);
        if(j<history_offset) e.stats.card_overflow+=t.num[CountOverflow];
        if(t.type==Player)e.stats.used_overflow+=t.num[UsedOverflow];
    }
    std::fill_n(label_opp_hand.data()+i*297,297,0);
    label_opp_hand_count[i]=players[1-seat].items.size();
    for(const auto& item:players[1-seat].items) ++label_opp_hand[i*297+card_code(item.model)];
}
void VecEnv::record_decision(std::size_t i, bool automatic) {
    auto& e = envs_[i];
    for(int seat=0;seat<2;++seat) {
        auto after=e.match.view(e.before[seat].board.viewer);
        auto d=e.match.last_decision(e.before[seat].board.viewer);
        if(!d) throw std::runtime_error("accepted decision lacks public record");
        e.history[seat].append(e.before[seat],after,*d,*e.before[seat].board.viewer);
    }
    e.encoded=false;
    ++e.stats.decisions;
    if (automatic) ++e.stats.automatic_decisions;
    decision_done[i] = true;
    e.selection.reset();
}
void VecEnv::drive(std::size_t i) {
    auto& e = envs_[i];
    for (;;) {
        if(e.waiting)return;
        const auto& s = e.match.snapshot();
        for (const auto& p : s.state.players) if (p.items.size() > 18)
            throw std::runtime_error("env " + std::to_string(i) + ": hand exceeds 18 cards");
        if (e.match.done()) {
            // Multiple completions within a single step would be unrepresentable.
            if (!eval_pairs_ && episode_done[i]) throw std::runtime_error("multiple episodes completed in one step");
            episode_done[i] = true; truncated[i] = s.truncated;
            for (int seat = 0; seat < 2; ++seat)
                outcome[2 * i + seat] = s.truncated ? 0 : static_cast<std::int8_t>(s.rewards.at(seat));
            ++e.stats.episodes; e.stats.truncations += s.truncated;
            if(eval_pairs_){e.results.push_back(evaluation_result(e.match,e.game));e.game+=num_envs;}
            ++e.episode; reset(i);
            if(!eval_pairs_ && controllers_[0]!=Controller::external && controllers_[1]!=Controller::external) {
                // A bot-only training env yields once per episode; next step ignores its choice.
                stage[i]=-1;actor_seat[i]=-1;mask[i].fill(false);return;
            }
            continue;
        }
        if (s.phase == gf::MatchPhase::grant) {
            before_decision(i);
            e.match.grant_step(e.match.sample_grant(e.chance_rng));
            ++e.stats.grant_decisions; record_decision(i, true); continue;
        }
        const auto actor = std::find_if(s.state.players.begin(), s.state.players.end(),
            [&](const auto& p) { return s.actor && p.id == *s.actor; });
        if (s.phase == gf::MatchPhase::main && actor != s.state.players.end() && actor->confusion_count > 0) {
            const auto command = e.match.sample_confusion(e.chance_rng);
            if (!e.match.confusion_legal(command)) throw std::runtime_error("illegal sampled confusion command");
            before_decision(i);
            e.match.confusion_step(command);
            ++e.stats.confusion_decisions; record_decision(i, true); continue;
        }
        const auto seat=std::size_t(actor-s.state.players.begin());
        const auto control=eval_pairs_ ? (seat==e.game%2 ? Controller::external : opponent_) : controllers_[seat];
        if(control!=Controller::external) {
            const auto command=bot_action(control,e.match,*s.actor,e.policy_rng[seat]);
            if(!e.match.legal(command))throw std::runtime_error("bot produced illegal Command");
            before_decision(i);e.match.step(command);record_decision(i,true);continue;
        }
        if (!e.selection) e.selection.emplace(e.match, canonical_);
        auto& selection = *e.selection;
        if (selection.complete) {
            if (!e.match.legal(selection.command)) throw std::runtime_error("env " + std::to_string(i) + ": selection produced illegal Command");
            const bool automatic = selection.substep == 0;
            before_decision(i);
            e.match.step(selection.command);
            record_decision(i, automatic); continue;
        }
        const auto count = std::count(selection.mask.begin(), selection.mask.end(), true);
        if (count >= 2) { publish(i); return; }
        const auto choice = std::find(selection.mask.begin(), selection.mask.end(), true) - selection.mask.begin();
        selection.choose(e.match, static_cast<int>(choice), canonical_);
    }
}
void VecEnv::publish(std::size_t i) {
    if(!envs_[i].encoded) { encode_observation(i); envs_[i].encoded=true; }
    const auto& s = *envs_[i].selection;
    stage[i] = static_cast<std::int8_t>(s.stage); actor_seat[i] = static_cast<std::int8_t>(s.actor_seat);
    mask[i] = s.mask;
    chosen_kind[i] = s.kind_chosen ? static_cast<std::int8_t>(s.command.kind) : -1;
    // Target 0 means explicitly no target; -1 means not selected yet.
    chosen_target[i] = s.stage >= Stage::alloc_hp
        ? (s.command.target ? (*s.command.target == s.command.actor ? 1 : 2) : 0) : -1;
    chosen_alloc_hp[i] = s.stage == Stage::alloc_mp
        ? static_cast<std::int8_t>(s.command.allocation->hp == 99 ? 20 : s.command.allocation->hp / 5) : -1;
    chosen_count[i] = static_cast<std::int8_t>(s.count); substep[i] = static_cast<std::int16_t>(s.substep);
    std::copy(s.cards.begin(), s.cards.end(), chosen_cards.data() + 18 * i);
}
void VecEnv::step(std::span<const std::int32_t> choices) {
    std::lock_guard lock(api_mutex_);
    if (choices.size() != num_envs) throw std::invalid_argument("choices must have shape [num_envs]");
    for (std::size_t i = 0; i < num_envs; ++i) {
        if(envs_[i].waiting || !envs_[i].selection)continue;
        const int c = choices[i];
        if (c < 0 || c >= 21 || !envs_[i].selection->mask[c])
            throw std::invalid_argument("env " + std::to_string(i) + ": illegal choice " + std::to_string(c) +
                " at stage " + std::to_string(stage[i]));
    }
    workers_.run([&](std::size_t worker) {
        for (std::size_t i = worker; i < num_envs; i += num_threads) {
            decision_done[i] = episode_done[i] = truncated[i] = false;
            outcome[2 * i] = outcome[2 * i + 1] = 0;
            auto& e = envs_[i];
            if(e.waiting)continue;
            if(!e.selection){drive(i);continue;}
            ++e.stats.policy_stages; ++e.selection->substep;
            e.selection->choose(e.match, choices[i], canonical_);
            drive(i);
        }
    });
}
std::vector<EvalResult> VecEnv::eval_results() const {
    std::lock_guard lock(api_mutex_);
    std::vector<EvalResult> out;
    for(const auto& e:envs_)out.insert(out.end(),e.results.begin(),e.results.end());
    std::sort(out.begin(),out.end(),[](const auto& a,const auto& b){return a.g<b.g;});
    return out;
}
bool VecEnv::eval_done() const {
    std::lock_guard lock(api_mutex_);
    return eval_pairs_ && std::all_of(envs_.begin(),envs_.end(),[](const auto& e){return e.waiting;});
}
Stats VecEnv::stats() const {
    std::lock_guard lock(api_mutex_);
    Stats sum;
    for (const auto& e : envs_) {
        sum.episodes += e.stats.episodes; sum.decisions += e.stats.decisions;
        sum.policy_stages += e.stats.policy_stages; sum.automatic_decisions += e.stats.automatic_decisions;
        sum.truncations += e.stats.truncations; sum.confusion_decisions += e.stats.confusion_decisions;
        sum.grant_decisions += e.stats.grant_decisions;
        sum.used_overflow += e.stats.used_overflow;
        for(const auto& h:e.history) {sum.history_dropped+=h.dropped;sum.card_overflow+=h.card_overflow;}
        sum.card_overflow+=e.stats.card_overflow;
    }
    return sum;
}
}
