namespace gfrl {
// Same SelectionTree predicates as the CPU oracle, with bounded slot storage.
// No heap allocation is needed to explore a cursor's possible next cards.
struct DeviceNode {
    gs::uint32_t next=0;
    unsigned targets=0;
    gs::int64_t total=0;
};
struct DeviceTree {
    const gf::MatchSnapshot& world;
    const gf::PlayerState& own;
    gf::CommandKind kind=gf::CommandKind::play;
    bool canonical;
    int n=0,hand_size;
    gs::uint32_t selected=0,disguised=0,used=0,ordinary=0,fate=0;
    bool weapon=false,enemy=false,root_ready=false;
    gs::uint32_t root_starts=0,first_copies=0;
    gs::uint32_t discards=0,exchanges=0,buys=0,sells=0,sale_items=0;
    gs::array<int,18> slots{};
    gs::array<gf::CardId,18> faces{},models{};
    gs::array<gf::VisibleSelection,18> visible{};
    gf::ResponseContext context;
    DeviceTree(const gf::MatchSnapshot& s,int seat,bool c,bool root=false):world(s),own(s.state.players[seat]),canonical(c),hand_size(own.items.size()) {
        if(s.phase==gf::MatchPhase::purchase)return;
        if(s.phase==gf::MatchPhase::response)context=gf::response_context(s);
        const auto mp=root?own.mp:0;
        gs::uint32_t eligible=0,needs_cut=0,cuts=0,enemy_only=0;
        unsigned response_key=0;
        if(root && s.phase==gf::MatchPhase::response){
            const unsigned element=context.incoming_element?static_cast<unsigned>(*context.incoming_element):6;
            response_key=(static_cast<unsigned>(context.incoming_category)*7+element)*2+context.incoming_is_attack;
            if(response_key>=112)gpu_fail(__LINE__);
        }
        for(int i=0;i<hand_size;++i) {
            const auto& item=own.items[i];const auto id=gf::face_model(item);const auto& card=*gf::find_card(id);
            faces[i]=id;disguised|=gs::uint32_t(item.fake_model.has_value())<<i;used|=gs::uint32_t(item.used)<<i;
            ordinary|=gs::uint32_t(!item.fake_model && card.category!=gf::Category::miracles && card.ability==gf::Ability::none)<<i;
            fate|=gs::uint32_t(gf::find_card(item.model)->ability==gf::Ability::call_phenomenon)<<i;
            weapon|=card.category==gf::Category::weapons;
            if(root){const auto bit=1u<<i;
                if(gf::discardable(card))discards|=bit;
                if(card.ability==gf::Ability::exchange)exchanges|=bit;
                if(id==5)buys|=bit;if(id==4)sells|=bit;
                if(!item.used && card.ability!=gf::Ability::discard && card.ability!=gf::Ability::sacrifice)sale_items|=bit;
                bool first=true;
                if(canonical)for(int j=0;j<i;++j)if(faces[j]==id && bool(disguised&(1u<<j))==bool(disguised&bit) && bool(used&(1u<<j))==bool(used&bit)){first=false;break;}
                if(first)first_copies|=bit;
                const auto cost=gf::gpu_single_cost[id];
                const bool compatible=s.phase==gf::MatchPhase::response
                    ? bool((gf::gpu_response_starts[id][response_key/64]>>(response_key%64))&1)
                    : gf::gpu_can_start_main[id] && !(fate&bit) && card.ability!=gf::Ability::exchange && card.ability!=gf::Ability::sell && card.ability!=gf::Ability::buy;
                if(compatible){if(cost<=mp)eligible|=bit;else if(card.category==gf::Category::miracles && !(s.phase==gf::MatchPhase::response && context.flash))needs_cut|=bit;}
                if(card.ability==gf::Ability::cut_cost && cost<=mp)cuts|=bit;
                if(card.ability!=gf::Ability::danger && (card.hit_rate || card.ability==gf::Ability::attack_every_enemy))enemy_only|=bit;
            }
        }
        const auto* owner=gf::find_player(s.state,s.owner);
        for(const auto& p:s.state.players)enemy|=!p.dead && p.id!=s.owner && (owner->team==gf::Team::solo || p.team!=owner->team);
        if(root){root_starts=eligible|(cuts?needs_cut:0);if(s.phase==gf::MatchPhase::main && !enemy)root_starts&=~enemy_only;root_ready=true;}
    }
    unsigned kinds() {
        unsigned result=0;
        auto allow=[&](gf::CommandKind kind){result|=1u<<static_cast<unsigned>(kind);};
        if(world.phase==gf::MatchPhase::response){if(own.mp>=0)allow(gf::CommandKind::defend);return result;}
        if(world.phase==gf::MatchPhase::purchase){
            allow(gf::CommandKind::decline);
            if(world.purchase && own.cp>=gf::find_card(world.purchase->item.model)->price)allow(gf::CommandKind::accept);
            return result;
        }
        if(discards)allow(gf::apocalypse_active(world.state)?gf::CommandKind::sacrifice:gf::CommandKind::discard);
        if(!weapon)allow(gf::CommandKind::pray);
        const auto total=own.hp+own.mp+own.cp;
        if(exchanges && total>=0 && total<=297)allow(gf::CommandKind::exchange);
        if(buys)allow(gf::CommandKind::buy);
        for(int i=0;i<hand_size;++i)if((sells&(1u<<i)) && (sale_items&~(1u<<i))){allow(gf::CommandKind::sell);break;}
        if(root_starts)allow(gf::CommandKind::play);
        return result;
    }
    gs::span<const gf::CardId> ids()const{return {models.data(),static_cast<gs::size_t>(n)};}
    gs::span<const gf::VisibleSelection> prefix()const{return {visible.data(),static_cast<gs::size_t>(n)};}
    void push(int i){slots[n]=i;models[n]=faces[i];visible[n]={faces[i],false};++n;selected|=1u<<i;}
    void pop(){selected&=~(1u<<slots[--n]);}
    bool exchangeable(int i,int position)const {
        return (kind==gf::CommandKind::defend || (kind==gf::CommandKind::play && position>0)) && (ordinary&(1u<<i));
    }
    bool appendable(int i)const {
        const auto bit=1u<<i;
        if((selected&bit) || n>=18)return false;
        if(canonical) {
            for(int j=0;j<i;++j)if(!(selected&(1u<<j)) && faces[j]==faces[i]
                && bool(disguised&(1u<<j))==bool(disguised&bit) && bool(used&(1u<<j))==bool(used&bit))return false;
            if(n) {
                const int last=slots[n-1];
                if((kind==gf::CommandKind::discard || kind==gf::CommandKind::sacrifice) && last>=i)return false;
                if(exchangeable(i,n) && exchangeable(last,n-1)) {
                    if(faces[last]>faces[i])return false;
                    if(faces[last]==faces[i] && (bool(used&(1u<<last))>bool(used&bit)
                        || (bool(used&(1u<<last))==bool(used&bit) && last>=i)))return false;
                }
            }
        }
        const auto id=faces[i];const auto& card=*gf::find_card(id);
        if(world.phase==gf::MatchPhase::purchase)return false;
        if(world.phase==gf::MatchPhase::response)
            return kind==gf::CommandKind::defend && gf::can_append_response(prefix(),{id,false},context);
        if(kind==gf::CommandKind::discard || kind==gf::CommandKind::sacrifice)
            return (kind==gf::CommandKind::sacrifice)==gf::apocalypse_active(world.state) && gf::discardable(card);
        if(kind==gf::CommandKind::exchange)return !n && card.ability==gf::Ability::exchange;
        if(kind==gf::CommandKind::buy)return !n && id==5;
        if(kind==gf::CommandKind::sell)return !n ? id==4 : n==1 && !(used&bit) && gf::can_append_main(prefix(),{id,false});
        if(kind!=gf::CommandKind::play)return false;
        if(!n && ((fate&bit) || card.ability==gf::Ability::exchange || card.ability==gf::Ability::sell || card.ability==gf::Ability::buy))return false;
        return gf::can_append_main(prefix(),{id,false});
    }
    DeviceNode finish()const {
        DeviceNode out;
        if(world.phase==gf::MatchPhase::purchase) {
            if(!n && (kind==gf::CommandKind::decline || (kind==gf::CommandKind::accept && world.purchase && own.cp>=gf::find_card(world.purchase->item.model)->price)))out.targets=1;
        } else if(world.phase==gf::MatchPhase::response) {
            if(kind==gf::CommandKind::defend && gf::magic_cost(ids())<=own.mp)out.targets=1;
        } else if(kind==gf::CommandKind::pray) {
            if(!n && !weapon)out.targets=1;
        } else if(kind==gf::CommandKind::discard || kind==gf::CommandKind::sacrifice) {
            if(n)out.targets=1;
        } else if(kind==gf::CommandKind::exchange) {
            const auto total=own.hp+own.mp+own.cp;
            if(n==1 && total>=0 && total<=297){out.targets=1;out.total=total;}
        } else if((kind==gf::CommandKind::buy && n==1) || (kind==gf::CommandKind::sell && n==2)
            || (kind==gf::CommandKind::play && n && gf::magic_cost(ids())<=own.mp)) {
            bool explicit_only=true,no_explicit=false;
            if(kind==gf::CommandKind::play) {
                const auto& base=*gf::find_card(models[0]);
                if(base.ability==gf::Ability::danger) {
                    if(gf::has_wide(ids()) && !gf::scoped_danger_wide(world,ids()))return out;
                    explicit_only=false;
                } else if(base.hit_rate || gf::has_wide(ids())) {
                    if(!enemy)return out;
                    explicit_only=false;no_explicit=base.hit_rate;
                }
            }
            if(!explicit_only)out.targets|=1;
            if(!no_explicit)for(const auto& p:world.state.players)if(!p.dead)out.targets|=1u<<(p.id==own.id?1:2);
        }
        return out;
    }
    bool reachable() {
        if(finish().targets)return true;
        if(!n)return false;
        if(kind!=gf::CommandKind::sell && gf::find_card(models[n-1])->category!=gf::Category::miracles)return false;
        for(int i=0;i<hand_size;++i) {
            if(kind!=gf::CommandKind::sell && gf::find_card(faces[i])->ability!=gf::Ability::cut_cost)continue;
            if(!appendable(i))continue;
            push(i);const bool ok=finish().targets;pop();if(ok)return true;
        }
        return false;
    }
    DeviceNode node(bool policy=false) {
        if(world.phase==gf::MatchPhase::purchase)return finish();
        if(root_ready && n==0){
            auto out=finish();
            if(world.phase==gf::MatchPhase::response){if(kind==gf::CommandKind::defend)out.next=root_starts;}
            else {
                if(kind==gf::CommandKind::play)out.next=root_starts;
                else if(kind==gf::CommandKind::discard || kind==gf::CommandKind::sacrifice){if((kind==gf::CommandKind::sacrifice)==gf::apocalypse_active(world.state))out.next=discards;}
                else if(kind==gf::CommandKind::exchange){const auto total=own.hp+own.mp+own.cp;if(total>=0 && total<=297)out.next=exchanges;}
                else if(kind==gf::CommandKind::buy)out.next=buys;
                else if(kind==gf::CommandKind::sell)for(int i=0;i<hand_size;++i)if((sells&(1u<<i)) && (sale_items&~(1u<<i)))out.next|=1u<<i;
            }
            out.next&=first_copies;return out;
        }
        bool danger=false,wide=false;
        if(policy)for(int j=0;j<n;++j){danger|=models[j]==94;wide|=models[j]==232;}
        if(danger && wide)return {};
        auto out=finish();
        for(int i=0;i<hand_size;++i) {
            if(policy && ((danger && faces[i]==232) || (wide && faces[i]==94)))continue;
            if(!appendable(i))continue;
            push(i);const bool ok=reachable();pop();if(ok)out.next|=1u<<i;
        }
        return out;
    }
};
}
