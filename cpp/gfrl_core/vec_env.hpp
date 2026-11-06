#pragma once
#include "history.hpp"
#include "bots.hpp"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace gfrl {
template<class T> struct Buffer {
    std::unique_ptr<T[]> storage;
    std::size_t size;
    explicit Buffer(std::size_t n) : storage(std::make_unique<T[]>(n)), size(n) {}
    T* data() { return storage.get(); }
    const T* data() const { return storage.get(); }
    T& operator[](std::size_t i) { return storage[i]; }
    const T& operator[](std::size_t i) const { return storage[i]; }
};
struct Stats {
    std::uint64_t episodes = 0, decisions = 0, policy_stages = 0, automatic_decisions = 0,
                  truncations = 0, confusion_decisions = 0, grant_decisions = 0, history_dropped = 0, card_overflow = 0, used_overflow = 0, token_overflow = 0;
};
// Persistent workers; jobs partition env indices deterministically.
class Workers {
public:
    explicit Workers(std::size_t n);
    ~Workers();
    void run(std::function<void(std::size_t)> job);
private:
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_, finish_;
    std::function<void(std::size_t)> job_;
    std::exception_ptr error_;
    std::size_t generation_ = 0, remaining_ = 0;
    bool stop_ = false;
};
class VecEnv {
public:
    const std::size_t num_envs, num_threads, history_len, tokens;
    Buffer<std::int8_t> obs_type, obs_cat;
    Buffer<std::int16_t> obs_cards, obs_aux_card;
    Buffer<float> obs_num;
    Buffer<bool> obs_valid;
    Buffer<std::uint8_t> label_opp_hand, label_opp_hand_count;
    Buffer<std::int8_t> stage, actor_seat, chosen_kind, chosen_cards, chosen_count, outcome;
    Buffer<Mask> mask;
    Buffer<std::int16_t> substep;
    Buffer<bool> decision_done, episode_done, truncated;
    Buffer<std::uint64_t> episode_id;
    VecEnv(std::size_t num_envs, std::size_t num_threads, std::uint64_t base_seed,
           bool canonical = true, gf::MatchConfig config = {}, std::size_t history_len = 64,
           std::array<Controller,2> controllers = {Controller::external, Controller::external},
           std::optional<std::size_t> eval_pairs = std::nullopt,
           Controller opponent = Controller::greedy, std::uint64_t eval_seed = 0);
    std::vector<EvalResult> eval_results() const;
    bool eval_done() const;
    void step(std::span<const std::int32_t> choices);
    Stats stats() const;
    const gf::Match& match(std::size_t i) const { return envs_.at(i).match; }
private:
    friend struct TestAccess;
    struct Env {
        gf::Match match;
        std::optional<Selection> selection;
        std::uint64_t chance_rng = 0, episode = 0;
        std::array<std::uint64_t,2> policy_rng{};
        std::uint64_t game = 0;
        bool waiting = false;
        std::vector<EvalResult> results;
        Stats stats;
        std::array<History,2> history;
        std::array<gf::MatchView,2> before;
        bool encoded = false;
    };
    const std::uint64_t base_seed_;
    const bool canonical_;
    const std::array<Controller,2> controllers_;
    const std::optional<std::size_t> eval_pairs_;
    const Controller opponent_;
    const std::uint64_t eval_seed_;
    std::vector<Env> envs_;
    Workers workers_;
    mutable std::mutex api_mutex_;
    void reset(std::size_t i);
    void drive(std::size_t i);
    void publish(std::size_t i);
    void before_decision(std::size_t i);
    void encode_observation(std::size_t i);
    void record_decision(std::size_t i, bool automatic);
};
}
