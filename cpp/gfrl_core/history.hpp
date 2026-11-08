#pragma once
#include "encoder.hpp"
#include <deque>
namespace gfrl {
// Restrict a source-visible history view to the other seat; never fill missing facts.
gf::MatchView project_history(gf::MatchView source, const gf::MatchSnapshot& visibility, int source_seat);
struct HistoryEvent { Token token; std::uint64_t decision; std::optional<Token> opposite; };
class History {
public:
    explicit History(std::size_t capacity=64): capacity_(capacity) {}
    void clear() { events_.clear(); }
    void append(const gf::MatchView& before, const gf::MatchView& after,
                const gf::PublicDecision& decision, gf::PlayerId viewer);
    // Attach only information derivable from this append's source-visible events.
    void set_projection(const History& projection);
    History projected() const;
    const std::deque<HistoryEvent>& events() const { return events_; }
    std::uint64_t dropped=0, card_overflow=0;
private:
    std::size_t capacity_;
    std::deque<HistoryEvent> events_;
    void push(Token token, std::uint64_t decision);
};
}
