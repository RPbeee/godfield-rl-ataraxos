#pragma once
#include "encoder.hpp"
#include <deque>
namespace gfrl {
struct HistoryEvent { Token token; std::uint64_t decision; };
class History {
public:
    explicit History(std::size_t capacity=64): capacity_(capacity) {}
    void clear() { events_.clear(); }
    void append(const gf::MatchView& before, const gf::MatchView& after,
                const gf::PublicDecision& decision, gf::PlayerId viewer);
    const std::deque<HistoryEvent>& events() const { return events_; }
    std::uint64_t dropped=0, card_overflow=0;
private:
    std::size_t capacity_;
    std::deque<HistoryEvent> events_;
    void push(Token token, std::uint64_t decision);
};
}
