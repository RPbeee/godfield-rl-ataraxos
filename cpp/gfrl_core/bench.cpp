#include "vec_env.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
int main() {
    std::cout << "N threads stages/s decisions/s episodes/s stages/decision auto_fraction truncation_rate\n";
    for (const auto n : {256, 1024, 4096}) for (const auto threads : {1, 4, 8}) {
        gfrl::VecEnv env(n, threads, 42);
        std::mt19937_64 rng(123);
        const auto before = env.stats();
        const auto start = std::chrono::steady_clock::now();
        // Fixed work makes the counters reproducible, while amortizing startup.
        for (int step = 0; step < 300; ++step) {
            const auto choices = gfrl::random_choices({env.mask.data(), env.num_envs}, {env.stage.data(), env.num_envs}, rng);
            env.step(choices);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto s = env.stats();
        s.policy_stages -= before.policy_stages; s.decisions -= before.decisions;
        s.episodes -= before.episodes; s.automatic_decisions -= before.automatic_decisions;
        s.truncations -= before.truncations;
        std::cout << n << ' ' << threads << ' ' << std::fixed << std::setprecision(2)
            << s.policy_stages / seconds << ' ' << s.decisions / seconds << ' ' << s.episodes / seconds << ' '
            << double(s.policy_stages) / s.decisions << ' ' << double(s.automatic_decisions) / s.decisions << ' '
            << (s.episodes ? double(s.truncations) / s.episodes : 0) << '\n';
    }
}
