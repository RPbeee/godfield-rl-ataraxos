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
VecEnv::VecEnv(std::size_t n, std::size_t threads, std::uint64_t seed, bool canonical, gf::MatchConfig config, std::size_t k, std::array<Controller,2> controllers, std::optional<std::size_t> eval_pairs, Controller opponent, std::uint64_t eval_seed, bool track_projection, bool search, std::size_t max_decisions)
    : num_envs(positive(n, "num_envs")), num_threads(positive(threads, "num_threads")),
      history_len(k<=4096 ? k : throw std::invalid_argument("history_len must be <=4096")), tokens(history_offset + history_len), track_projection(track_projection),
      obs_type(n*tokens), obs_cat(n*tokens*C), obs_cards(n*tokens*8), obs_aux_card(n*tokens),
      obs_num(n*tokens*F), obs_valid(n*tokens), label_opp_hand(n*297), label_opp_hand_count(n),
      label_opp_unused_hand(n*297), label_opp_unused_hand_count(n),
      stage(n), actor_seat(n), chosen_kind(n), chosen_cards(n * 18), chosen_count(n), chosen_target(n), chosen_alloc_hp(n), outcome(n * 2),
      mask(n), substep(n), decision_done(n), episode_done(n), truncated(n), episode_id(n), decisions(n), world_status(n), searcher_seat(n), result(n),
      leaf_obs_type(0), leaf_obs_cat(0), leaf_obs_cards(0),
      leaf_obs_aux_card(0), leaf_obs_num(0), leaf_obs_valid(0),
      search_(search), max_decisions_(positive(max_decisions,"max_decisions")), base_seed_(seed), canonical_(canonical), controllers_(controllers), eval_pairs_(eval_pairs), opponent_(opponent), eval_seed_(eval_seed), workers_(threads) {
    if (config != gf::MatchConfig{}) throw std::invalid_argument("VecEnv supports only default MatchConfig (max_decisions=512, include_fate/external_*=false)");
    if(eval_pairs_ && *eval_pairs_>10000000)throw std::invalid_argument("invalid evaluation configuration");
    envs_.resize(n);
    for(std::size_t i=0;i<n;++i)envs_[i].game=i;
    for(auto& e:envs_) {
        for(auto& h:e.history) h=History(k);
        if(track_projection) e.projected_before=std::make_unique<std::array<gf::MatchView,2>>();
    }
    workers_.run([&](std::size_t worker) {
        for (std::size_t i = worker; i < n; i += num_threads) {
            if(search_) { stop(i); episode_done[i]=true; searcher_seat[i]=-1; world_status[i]=int(WorldStatus::unsupported_root); continue; }
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
    for(int seat=0;seat<2;++seat) {
        e.before[seat]=e.match.view(e.match.snapshot().state.players[seat].id);
        if(track_projection) (*e.projected_before)[seat]=project_history(e.before[seat],e.match.snapshot(),seat);
    }
}
std::vector<Token> VecEnv::observation(std::size_t i, int seat) const {
    const auto& e=envs_[i];
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
    return ts;
}
void VecEnv::encode_observation(std::size_t i) {
    auto& e=envs_[i]; const auto seat=e.selection->actor_seat;
    const auto& players=e.match.snapshot().state.players;
    auto ts=observation(i,seat);
    std::size_t j;
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
    std::fill_n(label_opp_unused_hand.data()+i*297,297,0);
    label_opp_unused_hand_count[i]=0;
    for(const auto& item:players[1-seat].items) if(!item.used) {
        ++label_opp_unused_hand[i*297+card_code(item.model)];
        ++label_opp_unused_hand_count[i];
    }
}
void VecEnv::record_decision(std::size_t i, bool automatic) {
    auto& e = envs_[i];
    for(int seat=0;seat<2;++seat) {
        auto after=e.match.view(e.before[seat].board.viewer);
        auto d=e.match.last_decision(e.before[seat].board.viewer);
        if(!d) throw std::runtime_error("accepted decision lacks public record");
        e.history[seat].append(e.before[seat],after,*d,*e.before[seat].board.viewer);
        if(track_projection) {
            const auto& snap=e.match.snapshot();
            const auto other=1-seat;
            if(snap.last_decision && (snap.last_decision->hidden_viewers & (1u<<other))) {d->faces.clear();d->faces_hidden=true;}
            History projected(4096);
            projected.append((*e.projected_before)[seat],project_history(after,snap,seat),*d,snap.state.players[other].id);
            e.history[seat].set_projection(projected);
        }
    }
    if(e.play_seat) {
        const auto seat=*e.play_seat;
        const auto viewer=e.before[seat].board.viewer;
        e.play_events.push_back({e.before[seat],e.match.view(viewer),*e.match.last_decision(viewer)});
    }
    e.encoded=false;
    ++e.stats.decisions;
    if (automatic) ++e.stats.automatic_decisions;
    if(!search_ || !automatic) decision_done[i] = true;
    if(search_ && !automatic) ++decisions[i];
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
            if(search_) {
                episode_done[i]=true; truncated[i]=s.truncated;
                for(int seat=0;seat<2;++seat) outcome[2*i+seat]=s.truncated ? 0 : s.rewards.at(seat);
                result[i]=outcome[2*i+searcher_seat[i]];
                stop(i); return;
            }
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
        if (count >= 2) {
            if(search_ && decisions[i]>=max_decisions_) {truncated[i]=true;episode_done[i]=true;stop(i);return;}
            publish(i); return;
        }
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
        const int c = choices[i];
        // Inactive rows ignore their choice, as before (rollout passes -1 there).
        if(envs_[i].waiting || !envs_[i].selection)continue;
        if (c == -1 && envs_[i].selection->substep == 0)
            continue;
        if (c < 0 || c >= 21 || !envs_[i].selection->mask[c])
            throw std::invalid_argument("env " + std::to_string(i) + ": illegal choice " + std::to_string(c) +
                " at stage " + std::to_string(stage[i]));
    }
    workers_.run([&](std::size_t worker) {
        for (std::size_t i = worker; i < num_envs; i += num_threads) {
            // A frozen decision (-1 on a live substep-0 row) is a complete no-op,
            // including published flags. -1 on inactive rows keeps the old path.
            if(choices[i] == -1 && !envs_[i].waiting && envs_[i].selection
               && envs_[i].selection->substep == 0)continue;
            if(search_)decision_done[i]=false;
            if(search_ && envs_[i].waiting)continue;
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
void VecEnv::stop(std::size_t i) {
    auto& e=envs_[i];e.waiting=true;e.selection.reset();
    stage[i]=actor_seat[i]=chosen_kind[i]=chosen_target[i]=chosen_alloc_hp[i]=-1;
    chosen_count[i]=0;substep[i]=0;mask[i].fill(false);
    std::fill_n(chosen_cards.data()+18*i,18,-1);
}
std::vector<int> VecEnv::determinize_support(std::span<const std::int32_t> rows) const {
    std::lock_guard lock(api_mutex_);
    std::vector<int> out;
    for(auto row:rows) {
        if(row<0 || std::size_t(row)>=num_envs)throw std::out_of_range("source row");
        const auto& e=envs_[row];
        out.push_back(e.waiting || !e.selection || substep[row]!=0 ? -1 :
            int(e.match.determinize_support(e.match.snapshot().state.players[e.selection->actor_seat].id,true)));
    }
    return out;
}
std::array<std::int32_t,297> unused_assignment(const gf::Match& m, int searcher, std::span<const std::int32_t> assignment) {
    if(searcher<0 || searcher>1 || assignment.size()!=297)throw std::invalid_argument("assignment must have 297 counts and valid seat");
    std::array<std::int32_t,297> counts{};
    std::uint64_t total=0;
    for(std::size_t c=0;c<297;++c) {
        if(assignment[c]<0)throw std::invalid_argument("negative assignment count");
        total+=assignment[c];counts[c]=assignment[c];
    }
    if(counts[0] || total>m.snapshot().config.hand_limit)throw std::invalid_argument("invalid assignment capacity or padding");
    for(const auto& card:gf::catalog())
        if(counts[card_code(card.id)] && !m.gift_supported(card.id))throw std::invalid_argument("non-gift assignment card");
    std::size_t used=0;
    for(const auto& item:m.snapshot().state.players[1-searcher].items)if(item.used) {
        ++used;auto& c=counts[card_code(item.model)];c=std::max(0,c-1);
    }
    // The launch anchor also retains public selected cards: consumed weapons
    // are absent now, but still occupy physical slots before replay.
    std::size_t retained=used;
    const auto& snap=m.snapshot();
    if(snap.phase==gf::MatchPhase::response && snap.batch) {
        const auto& b=*snap.batch;
        const auto& opponent=snap.state.players[1-searcher];
        retained=0;
        for(const auto& item:b.initial_state.players[1-searcher].items) {
            bool selected=b.attack.actor==opponent.id &&
                std::find(b.attack.items.begin(),b.attack.items.end(),item.token)!=b.attack.items.end();
            for(const auto& command:b.responses)if(command.actor==opponent.id &&
                std::find(command.items.begin(),command.items.end(),item.token)!=command.items.end())selected=true;
            retained+=item.used || selected;
        }
    }
    std::uint64_t physical=retained;
    for(auto c:counts)physical+=c;
    if(physical>m.snapshot().config.hand_limit)throw std::invalid_argument("assignment plus retained used cards exceeds capacity");
    return counts;
}
std::array<std::int32_t,297> VecEnv::unused_assignment(std::size_t row, std::span<const std::int32_t> assignment) const {
    std::lock_guard lock(api_mutex_);
    const auto& e=envs_.at(row);
    if(e.waiting || !e.selection || substep[row]!=0)throw std::invalid_argument("not a root decision");
    return gfrl::unused_assignment(e.match,e.selection->actor_seat,assignment);
}
void VecEnv::load_worlds(const VecEnv& src, std::span<const std::int32_t> rows,
                         std::span<const std::int32_t> assignments, std::span<const std::uint64_t> seeds) {
    if(!search_ || this==&src)throw std::invalid_argument("load_worlds needs a separate SearchEnv destination");
    if(!src.track_projection)throw std::invalid_argument("load_worlds requires src with track_projection=true");
    std::scoped_lock lock(api_mutex_,src.api_mutex_);
    if(rows.size()!=num_envs || seeds.size()!=num_envs || assignments.size()!=num_envs*297 || history_len!=src.history_len || canonical_!=src.canonical_)
        throw std::invalid_argument("world shapes, history_len or canonical mismatch");
    for(auto row:rows)if(row<0 || std::size_t(row)>=src.num_envs)throw std::out_of_range("source row");
    workers_.run([&](std::size_t worker){for(std::size_t i=worker;i<num_envs;i+=num_threads) {
        auto& e=envs_[i];const auto& root=src.envs_[rows[i]];
        stop(i);decisions[i]=0;decision_done[i]=false;episode_done[i]=true;truncated[i]=false;
        result[i]=0;outcome[2*i]=outcome[2*i+1]=0;episode_id[i]=src.episode_id[rows[i]];searcher_seat[i]=-1;
        // Clear every old world output, including invalid rows on reload.
        std::fill_n(obs_type.data()+i*tokens,tokens,0);std::fill_n(obs_valid.data()+i*tokens,tokens,false);
        std::fill_n(obs_cards.data()+i*tokens*8,tokens*8,0);std::fill_n(obs_aux_card.data()+i*tokens,tokens,0);
        std::fill_n(obs_cat.data()+i*tokens*C,tokens*C,0);std::fill_n(obs_num.data()+i*tokens*F,tokens*F,0);
        std::fill_n(label_opp_hand.data()+i*297,297,0);label_opp_hand_count[i]=0;
        std::fill_n(label_opp_unused_hand.data()+i*297,297,0);label_opp_unused_hand_count[i]=0;
        world_status[i]=int(WorldStatus::unsupported_root);
        if(root.waiting || !root.selection || src.substep[rows[i]]!=0)continue;
        const auto seat=root.selection->actor_seat;
        const auto id=root.match.snapshot().state.players[seat].id;
        if(root.match.determinize_support(id,true)!=gf::DeterminizeSupport::ok)continue;
        world_status[i]=int(WorldStatus::invalid_assignment);
        try {
            const auto counts=gfrl::unused_assignment(root.match,seat,assignments.subspan(i*297,297));
            std::vector<gf::CardId> models;
            for(const auto& card:gf::catalog())for(int j=0;j<counts[card_code(card.id)];++j)models.push_back(card.id);
            std::vector<gf::CardId> own;
            // Independent stream: preserve the established chance seed and all
            // old roots bit for bit. Read only public face/disguised metadata.
            auto rng=mix(seeds[i]^0x36d8a971c2ef504bULL);
            for(const auto& item:root.match.view(id).board.hand) if(item.disguised) {
                std::vector<gf::CardId> tickets;
                for(const auto& card:gf::catalog()) if(root.match.gift_supported(card.id)
                        && gf::match_dream_compatible(card.id,item.face))
                    for(int n=0;n<card.gift_rate;++n)tickets.push_back(card.id);
                // Official public hands may mark even singleton faces disguised.
                // Natural local draws never do that. Use the supported identity
                // assignment only for this otherwise empty public-face prior;
                // never consult the original hidden model or change old draws.
                if(tickets.empty() && root.match.gift_supported(item.face))tickets.push_back(item.face);
                if(tickets.empty()) {world_status[i]=int(WorldStatus::empty_own_prior);throw std::invalid_argument("empty own Dream prior");}
                // Exact bounded draw (avoid modulo bias).
                const auto bound=std::uint64_t(tickets.size()), threshold=-bound%bound;
                do {rng=mix(rng);} while(rng<threshold);
                own.push_back(tickets[rng%bound]);
            }
            e.match=own.empty()?root.match.determinized(id,models,seeds[i],true):
                root.match.determinized(id,models,own,seeds[i],true);
        } catch(const std::invalid_argument&) {continue;}
          catch(const gf::MatchError& error) {if(error.code==gf::MatchErrorCode::invalid) {
              const std::string_view message=error.what();
              if(message.find("observation differs")!=message.npos)world_status[i]=int(WorldStatus::observation_mismatch);
              else if(message.find("decision differs")!=message.npos)world_status[i]=int(WorldStatus::decision_mismatch);
              else if(message.find("differs")!=message.npos)world_status[i]=int(WorldStatus::legality_mismatch);
              continue;
          }throw;}
        e.history[seat]=root.history[seat];e.history[1-seat]=root.history[seat].projected();
        e.chance_rng=mix(seeds[i]^0xa4093822299f31d0ULL);e.encoded=false;e.waiting=false;
        e.selection.reset();
        // Rebuild the first cursor with the same public slot exclusions. Once
        // its command executes, record_decision drops it; rollout cursors are free.
        if(root.selection->public_root)
            e.selection.emplace(e.match,canonical_,root.selection->exclusions,root.selection->public_root);
        searcher_seat[i]=seat;episode_done[i]=false;world_status[i]=int(WorldStatus::ok);
        drive(i);
    }});
}
void VecEnv::diagnostic_copy_exact(const VecEnv& src, std::span<const std::int32_t> rows,
                                   std::span<const std::int32_t> destinations,
                                   std::span<const std::uint64_t> seeds, bool reseed) {
    if(!search_ || this==&src)throw std::invalid_argument("diagnostic copy needs separate SearchEnv destination");
    std::scoped_lock lock(api_mutex_,src.api_mutex_);
    if(rows.size()!=destinations.size() || rows.size()!=seeds.size() || history_len!=src.history_len || canonical_!=src.canonical_)
        throw std::invalid_argument("diagnostic copy shapes/history/canonical mismatch");
    std::vector<bool> seen(num_envs,false);
    for(std::size_t j=0;j<rows.size();++j) {
        if(rows[j]<0 || std::size_t(rows[j])>=src.num_envs || destinations[j]<0 || std::size_t(destinations[j])>=num_envs)
            throw std::out_of_range("diagnostic copy row");
        const auto& root=src.envs_[rows[j]];
        if(seen[destinations[j]] || root.waiting || !root.selection || src.stage[rows[j]]<0)
            throw std::invalid_argument("diagnostic copy requires unique destinations and live cursors");
        seen[destinations[j]]=true;
    }
    workers_.run([&](std::size_t worker){for(std::size_t j=worker;j<rows.size();j+=num_threads) {
        const auto i=destinations[j];const auto& root=src.envs_[rows[j]];auto& e=envs_[i];
        e.match=root.match; // Includes response anchors, private models and pending effects.
        // A response's historical replay proof includes its old RNG. Replaying
        // that launch would change the conditioning state. This diagnostic
        // intervention replaces ONLY the future stream, on a non-const object;
        // it must not go through restore()'s historical proof validation.
        // No journal, redirect anchor, model, queue or already-drawn event moves.
        if(reseed)const_cast<gf::MatchSnapshot&>(e.match.snapshot()).rng=seeds[j];
        e.history=root.history;e.selection=root.selection;
        e.chance_rng=reseed?mix(seeds[j]^0xa4093822299f31d0ULL):root.chance_rng;
        e.policy_rng=root.policy_rng;e.encoded=false;e.waiting=false;
        e.play_seat.reset();e.play_events.clear();e.results.clear();e.stats={};
        e.projected_before.reset();
        decisions[i]=0;decision_done[i]=false;episode_done[i]=false;truncated[i]=false;
        result[i]=0;outcome[2*i]=outcome[2*i+1]=0;episode_id[i]=src.episode_id[rows[j]];
        searcher_seat[i]=root.selection->actor_seat;world_status[i]=int(WorldStatus::ok);
        publish(i);
    }});
}
// stop() clears selection and actor_seat; the engine still retains the actor
// of the next policy boundary. Read it without publishing/mutating the row.
std::vector<int> VecEnv::next_actor_seats(std::span<const std::int32_t> rows) const {
    std::lock_guard lock(api_mutex_);
    std::vector<int> out;
    for(auto row:rows) {
        if(row<0 || std::size_t(row)>=num_envs)throw std::out_of_range("actor row");
        const auto& m=envs_[row].match;
        const auto& s=m.snapshot();
        int seat=-1;
        if(!m.done() && (!search_ || world_status[row]==int(WorldStatus::ok)))
            for(int j=0;j<2;++j)if(s.actor && s.state.players[j].id==*s.actor)seat=j;
        out.push_back(seat);
    }
    return out;
}
void VecEnv::observe(std::span<const std::int32_t> rows, std::span<const std::int32_t> seats) {
    std::lock_guard lock(api_mutex_);
    if(rows.size()!=seats.size() || rows.size()>num_envs)throw std::invalid_argument("observe expects matching rows/seats, at most num_envs");
    for(std::size_t j=0;j<rows.size();++j) {
        if(rows[j]<0 || std::size_t(rows[j])>=num_envs || seats[j]<0 || seats[j]>1)throw std::out_of_range("observe row/seat");
        if(search_ && world_status[rows[j]]!=int(WorldStatus::ok))throw std::invalid_argument("observe invalid world");
    }
    if(!leaf_obs_type.size) {
        leaf_obs_type=Buffer<std::int8_t>(num_envs*tokens);leaf_obs_cat=Buffer<std::int8_t>(num_envs*tokens*C);
        leaf_obs_cards=Buffer<std::int16_t>(num_envs*tokens*8);leaf_obs_aux_card=Buffer<std::int16_t>(num_envs*tokens);
        leaf_obs_num=Buffer<float>(num_envs*tokens*F);leaf_obs_valid=Buffer<bool>(num_envs*tokens);
    }
    for(std::size_t j=0;j<rows.size();++j) {
        const auto ts=observation(rows[j],seats[j]);
        for(std::size_t k=0;k<tokens;++k) {
            const auto& t=ts[k];const auto row=j*tokens+k;
            leaf_obs_type[row]=t.type;leaf_obs_aux_card[row]=t.aux;leaf_obs_valid[row]=t.type!=0;
            std::copy(t.cards.begin(),t.cards.end(),leaf_obs_cards.data()+row*8);
            std::copy(t.cat.begin(),t.cat.end(),leaf_obs_cat.data()+row*C);
            std::copy(t.num.begin(),t.num.end(),leaf_obs_num.data()+row*F);
        }
    }
}

}

namespace gfrl {
gf::MatchView VecEnv::play_view(std::size_t row, int seat) const {
    std::lock_guard lock(api_mutex_);
    if(seat<0 || seat>1)throw std::out_of_range("seat");
    const auto& m=envs_.at(row).match;
    return m.view(m.snapshot().state.players.at(seat).id);
}
void VecEnv::enable_play_trace(std::size_t row, int seat) {
    std::lock_guard lock(api_mutex_);
    if(seat<0 || seat>1)throw std::out_of_range("seat");
    auto& e=envs_.at(row);e.play_seat=seat;e.play_events.clear();
}
std::vector<PlayEvent> VecEnv::take_play_trace(std::size_t row) {
    std::lock_guard lock(api_mutex_);
    std::vector<PlayEvent> out;out.swap(envs_.at(row).play_events);return out;
}
std::vector<gf::HandItemView> VecEnv::play_hand(std::size_t row, int seat) const {
    std::lock_guard lock(api_mutex_);
    if(seat<0 || seat>1)throw std::out_of_range("seat");
    const auto& m=envs_.at(row).match;
    const auto& own=m.snapshot().state.players.at(seat);
    const auto v=m.view(own.id);
    std::vector<gf::HandItemView> out;
    for(const auto& item:own.items) {
        const auto it=std::find_if(v.board.hand.begin(),v.board.hand.end(),[&](const auto& h){return h.token==item.token;});
        if(it==v.board.hand.end())throw std::runtime_error("own visible slot missing");
        out.push_back(*it);
    }
    return out;
}
}
