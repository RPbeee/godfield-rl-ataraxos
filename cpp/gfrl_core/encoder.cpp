#include "encoder.hpp"
#include <algorithm>
#include <cmath>
namespace gfrl {
std::int16_t card_code(gf::CardId id) { return id ? gf::card_index(id)+1 : 0; }
int relative(gf::PlayerId id, gf::PlayerId viewer) { return id==viewer ? 1:2; }
void cards(Token& t, std::span<const gf::CardId> ids) {
    for(std::size_t i=0;i<std::min(ids.size(),t.cards.size());++i) t.cards[i]=card_code(ids[i]);
    t.num[CountOverflow]=ids.size()>8 ? ids.size()-8 : 0;
}
std::vector<std::string> num_names() {
    std::vector<std::string> n{"hp","mp","cp","status_hidden","dead","fogged","confused","used_overflow"};
    for(int i=0;i<9;++i) n.push_back("curse_"+(i ? std::string(gf::curse_name(static_cast<gf::Curse>(i))) : std::string("none")));
    for(auto s:{"gf","tiebreak_gf","tiebreak_hidden","decisions","apocalypse","first_seat","hand_count","disguised","used","power_log1p","amount","hit_rate","reflections","bounces","counter","environmental","death","forced_hit","forced_hidden","price","faces_hidden","has_allocation","allocation_hp","allocation_mp","allocation_cp","age","mortar","card_overflow"}) n.emplace_back(s);
    for(auto seat:{"self_","opp_"}) for(auto s:{"delta_hp","delta_mp","delta_cp","delta_hidden","delta_dead","delta_fogged","delta_confused","guardian_before","guardian_after","curse_none","curse_cold","curse_fever","curse_hell","curse_heaven","curse_fog","curse_dream","curse_flash","curse_darkcloud"}) n.push_back(std::string(seat)+s);
    return n;
}
std::vector<Token> encode(const gf::MatchView& v, gf::PlayerId viewer, std::span<const gf::ItemToken> order) {
    std::vector<Token> out(history_offset);
    auto& cls=out[0]; cls.type=Cls; cls.cat[Phase]=int(v.phase)+1;
    cls.num[Gf]=v.board.gf/512.f; cls.num[TiebreakHidden]=!v.board.tiebreak_gf;
    if(v.board.tiebreak_gf) cls.num[Tiebreak]=*v.board.tiebreak_gf/512.f;
    cls.num[Decisions]=v.decisions/512.f; cls.num[Apocalypse]=v.apocalypse_active;
    cls.num[FirstSeat]=v.board.players.at(0).id==viewer; cls.num[HandCount]=v.board.hand.size()/18.f;
    for(const auto& p:v.board.players) {
        const int r=relative(p.id,viewer); auto& t=out[r]; t.type=Player; t.cat[Actor]=r;
        t.num[Dead]=p.dead; t.num[Fogged]=p.fogged; t.num[StatusHidden]=!p.status;
        if(p.status) { const auto& s=*p.status; t.num[Hp]=s.hp/99.f; t.num[Mp]=s.mp/99.f; t.num[Cp]=s.cp/99.f;
            t.cat[Guardian]=int(s.guardian)+1; t.num[Confused]=s.confused;
            for(auto curse:s.curses) t.num[Curse0+int(curse)]=1;
        }
        t.num[UsedOverflow]=p.used_models.size()>16 ? p.used_models.size()-16 : 0;
        for(std::size_t j=0;j<std::min<std::size_t>(16,p.used_models.size());++j) {
            auto& u=out[23+(r-1)*16+j]; u.type=Used; u.cat[Actor]=r; u.cards[0]=card_code(p.used_models[j]);
        }
    }
    if(order.size()>18) throw std::runtime_error("own hand exceeds observation capacity");
    for(std::size_t j=0;j<order.size();++j) {
        const auto it=std::find_if(v.board.hand.begin(),v.board.hand.end(),[&](const auto& h){return h.token==order[j];});
        if(it==v.board.hand.end()) throw std::runtime_error("own token missing from public hand");
        auto& t=out[3+j]; t.type=Hand; t.cards[0]=card_code(it->face); t.num[Disguised]=it->disguised; t.num[IsUsed]=it->used;
    }
    if(v.pending) { const auto& p=*v.pending; auto& t=out[21]; t.type=Pending; cards(t,p.models);
        if(p.origin) t.cat[Origin]=relative(*p.origin,viewer);
        if(p.source) t.cat[SourceSeat]=relative(*p.source,viewer);
        t.cat[Target]=relative(p.target,viewer); t.cat[Category]=int(p.category)+1;
        if(p.element) t.cat[Element]=int(*p.element)+1;
        t.cat[Guardian]=int(p.guardian)+1; t.cat[Ability]=int(p.ability)+1;
        t.num[Power]=std::copysign(std::log1p(std::abs(double(p.power))),double(p.power)); t.num[Amount]=p.amount/99.f;
        t.num[HitRate]=p.hit_rate/100.f; t.num[Reflections]=p.reflections/18.f; t.num[Bounces]=p.bounces/18.f;
        t.num[Counter]=p.counter; t.num[Environmental]=p.environmental; t.num[Death]=p.death;
        t.num[ForcedHidden]=!p.forced_hit; if(p.forced_hit) t.num[ForcedHit]=*p.forced_hit;
    }
    if(v.purchase) {const auto& p=*v.purchase; auto& t=out[22]; t.type=Offer; t.cat[Origin]=relative(p.origin,viewer);
        t.cat[Actor]=relative(p.buyer,viewer); t.cat[Target]=relative(p.seller,viewer); t.aux=card_code(p.model); t.num[Price]=p.price/99.f; }
    return out;
}
const std::vector<float>& card_features() {
    static const auto features=[] {
        std::vector<float> a(297*card_feature_count);
        for(const auto& c:gf::catalog()) {auto* p=a.data()+card_code(c.id)*card_feature_count; int k=0;
            p[k+int(c.category)]=1; k+=8; if(c.element) p[k+int(*c.element)]=1; k+=6;
            p[k++]=c.is_plus_atk; p[k++]=std::log1p(c.atk); p[k++]=c.hit_rate/100.f; p[k++]=std::log1p(c.defense);
            p[k+int(c.ability)]=1; k+=61; p[k++]=c.ability_value/99.f;
            p[k+int(c.curse)]=1; k+=9; p[k+int(c.guardian)]=1; k+=11;
            p[k++]=c.price/99.f; p[k++]=c.cost/99.f; p[k++]=c.gift_rate/100.f;
        } return a;
    }(); return features;
}
}
