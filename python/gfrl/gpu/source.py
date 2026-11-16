"""Build a CUDA training translation unit without modifying the CPU oracle.

Only standard-library storage and error handling are adapted. Rule arithmetic,
iteration order, RNG and public projections come directly from the checked-in
engine. The CPU engine remains the independent differential oracle.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def _clean(text):
    return re.sub(r'^\s*#(?:include|pragma).*$', '', text, flags=re.M)


def _end(text, start):
    """Matching brace, ignoring C++ strings/comments."""
    pattern = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[{}]', re.S)
    depth = 0
    for m in pattern.finditer(text, start):
        if m[0] == '{':
            depth += 1
        elif m[0] == '}':
            depth -= 1
            if not depth:
                return m.end()
    raise ValueError('unbalanced C++ braces')


def _replace_function(text, name, replacement=''):
    # Upstream definitions start on a new line; do not match calls or declarations.
    pattern = re.compile(r'^([^\n;{}]*\b' + re.escape(name) + r'\s*\([^;{}]*?\)[^;{}]*?)\{', re.M)
    matches = list(pattern.finditer(text))
    for m in reversed(matches):
        end = _end(text, m.end()-1)
        text = text[:m.start()] + replacement + text[end:]
    if not matches:
        raise ValueError(f'function definition not found: {name}')
    return text


def _read(path):
    return _clean((ROOT / path).read_text())


def build_source(*, specialize_config=False, private_rules=False, search=False):
    base = 'third_party/goodfield/'
    headers = [_read(base + 'include/godfield/' + n + '.hpp') for n in (
        'official_catalog', 'official_rules', 'official_state', 'official_match')]
    if not re.search(r'catalog_count\s*=\s*296\s*;',headers[0]):
        raise ValueError('GPU lookup tables require the 296-card catalog')
    headers[2] = headers[2].replace('project_view(const SnapshotState& state, std::optional<PlayerId> viewer)',
        'project_view(const SnapshotState& state, std::optional<PlayerId> viewer, bool history_only=false)')
    headers[-1] = headers[-1].replace('view(std::optional<PlayerId> viewer) const;',
        'view(std::optional<PlayerId> viewer, bool history_only=false) const;')
    headers[-1] = headers[-1].replace('struct ActionBatch {', 'struct ActionBatch { std::int64_t gpu_launch_mp=0;')
    headers[-1] = headers[-1].replace('MatchStep step(const Command& command);', 'void step(const Command& command);')
    headers[-1] = re.sub(r'class MatchError : public std::runtime_error \{.*?\n\};', '', headers[-1], flags=re.S)
    if search:
        headers[-1] = headers[-1].replace('struct MatchSnapshot {',
            'struct MatchSnapshot { bool gpu_lean=false;')
        headers[-1] = headers[-1].replace('class Match {\npublic:',
            'class Match {\npublic:\n    int gpu_search_status=0;')
    # Policy-only public selection restrictions are absent in training VecEnv.
    selh = _read('cpp/gfrl_core/selection.hpp')
    selh = selh[:selh.index('std::vector<std::int32_t> random_choices')] + '\n}\n'
    selh = selh.replace('std::mt19937_64', 'int')
    ench = _read('cpp/gfrl_core/encoder.hpp')
    ench = ench.replace('std::vector<Token> encode(', 'void encode(').replace('std::span<const gf::ItemToken> order);', 'std::span<const gf::ItemToken> order, std::vector<Token>& out);')
    hish = _read('cpp/gfrl_core/history.hpp')
    if not search:
        hish = hish.replace(' std::optional<Token> opposite;', '')
    catalog = _read(base + 'src/official_catalog.cpp')
    # Device globals must be constant initialized, with no host string pointers.
    catalog = catalog.replace('constexpr std::array<Card, catalog_count> records',
                              '__device__ const std::array<Card, catalog_count> records')
    catalog = re.sub(r'(\{\d+, )"[^"]*", "[^"]*",', r'\1nullptr, nullptr,', catalog)
    catalog = re.sub(r'static_assert\(records.front\(\).*?;', '', catalog)
    rules = _read(base + 'src/official_rules.cpp')
    state = _read(base + 'src/official_state.cpp')
    for name in ('quoted', 'number', 'boolean', 'active_attack_json', 'view_json'):
        state = _replace_function(state, name)
    state = _replace_function(state, 'validate_state', 'void validate_state(const SnapshotState&) {}')
    state = re.sub(r'\bdisease\b', 'state_disease', state)
    # History reads panes and journals, never hand/used-card renderer arrays.
    # Observation hand slots are resolved by physical token, so its renderer
    # sort is also unnecessary. The used-model sort remains observable.
    state = state.replace('project_view(const SnapshotState& state, std::optional<PlayerId> viewer)',
        'project_view(const SnapshotState& state, std::optional<PlayerId> viewer, bool history_only)')
    state = state.replace('for (const auto& instance : p.items) if (instance.used)',
        'if(!history_only) for (const auto& instance : p.items) if (instance.used)')
    state = state.replace('if (own) for (const auto& instance : own->items)',
        'if (own && !history_only) for (const auto& instance : own->items)')
    if not search:
        state = re.sub(r'    std::stable_sort\(out.hand.begin\(\), out.hand.end\(\), .*?\n    \}\);', '', state, flags=re.S)

    match = _read(base + 'src/official_match.cpp')
    match = match[:match.index('std::string Match::observe(')] + '\n}\n'
    match = match.replace('Match::view(std::optional<PlayerId> viewer) const',
        'Match::view(std::optional<PlayerId> viewer, bool history_only) const')
    match = match.replace('result.board = project_view(data_.state, viewer);',
        'result.board = project_view(data_.state, viewer, history_only);')
    match = match.replace('result.last_purchase = data_.last_purchase;',
        'result.last_purchase = data_.last_purchase; if(history_only)return result;')

    # Replay validation is CPU import functionality; no imports on training path.
    match = _replace_function(match, 'validate_snapshot', 'void validate_snapshot(const MatchSnapshot&) {}')
    keep_methods = {'Match', 'reset', 'step', 'view',
                    'last_decision', 'legal_kinds', 'legal_extensions',
                    'gift_weight', 'gift_catalog', 'gift_supported'}
    if search:
        keep_methods.update({'restore', 'load_scene', 'determinize_support',
                             'determinized', 'determinized_impl'})
    methods = set(re.findall(r'^.*?\bMatch::(\w+)\s*\(', match, re.M))
    for name in sorted(methods - keep_methods):
        match = _replace_function(match, 'Match::' + name)
    if not search:
        for name in ('replay_launch', 'replay_response'):
            match = _replace_function(match, name)
    else:
        from .search_source import checked_determinization
        match = checked_determinization(match)
    # Commands come only from the GPU legal cursor. Invalid commands trap the
    # CUDA context, so copying the whole snapshot for CPU-style rollback adds
    # no useful transaction boundary on this path.
    match = _replace_function(match, 'Match::step', 'void Match::step(const Command& c) { execute(data_,c); }')
    # Runtime payload construction needs the original owner's launch MP. The
    # full state and counters in ActionBatch's replay anchor are otherwise only
    # read by import/replay validation, which is absent on this training path.
    if not search:
        match = _replace_function(match,'action_anchor','''ActionBatch action_anchor(const MatchSnapshot& s) {
        ActionBatch batch;batch.owner=s.owner;batch.rng=s.rng;
        batch.gpu_launch_mp=find_player(s.state,s.owner)->mp;return batch;
    }''')
        match = match.replace('find_player(batch.initial_state, batch.owner)->mp', 'batch.gpu_launch_mp')
    else:
        # Root games retain complete replay anchors for determinization. Search
        # worlds never become sources of another search: use the same lean
        # launch anchor already validated by the ordinary GPU training engine.
        match = match.replace('ActionBatch batch;\n    batch.initial_state = s.state;',
            '''ActionBatch batch;batch.gpu_launch_mp=find_player(s.state,s.owner)->mp;
    if(s.gpu_lean){batch.owner=s.owner;batch.rng=s.rng;return batch;}
    batch.initial_state = s.state;''')
        match = match.replace('find_player(batch.initial_state, batch.owner)->mp', 'batch.gpu_launch_mp')
    match = match.replace('const auto board=project_view(s.state,s.state.players[i].id);',
        'const auto board=project_view(s.state,s.state.players[i].id,true);')

    # The cursor already proved composition/targets/allocation on immutable
    # state. prepare() is pure validation plus collecting actual model IDs;
    # retain that collection for Dream resolution and omit the second proof.
    match = _replace_function(match, 'prepare', '''std::vector<CardId> prepare(const MatchSnapshot& s,const Command& command,bool=false) {
        const auto& own=s.state.players[index(s,command.actor)];
        std::vector<CardId> actual;actual.reserve(command.items.size());
        for(auto token:command.items){const auto* item=instance(own,token);require(item,"missing cursor item");actual.push_back(item->model);}
        return actual;
    }''')

    match = _replace_function(match, 'Match::Match', 'Match::Match(MatchConfig config) : config_(config) { config_valid(config); }')
    # reset() receives only the two constant training PlayerSetups. Construct
    # directly in device storage; the CPU's transactional temporary is a large
    # per-thread stack frame, even on steps which do not reset an episode.
    reset = re.search(r'^void Match::reset\([^;{}]*\)\s*\{',match,re.M)
    reset_end = _end(match,reset.end()-1)
    reset_text = match[reset.start():reset_end]
    reset_text = reset_text.replace('MatchSnapshot s; s.config = config_;',
        'data_.~MatchSnapshot(); new(&data_) MatchSnapshot(); auto& s=data_; s.config = config_;')
    reset_text = reset_text.replace('validate_snapshot(s); data_ = std::move(s);', 'validate_snapshot(s);')
    match = match[:reset.start()] + reset_text + match[reset_end:]

    match = _replace_function(match, 'selection_extensions', '''SelectionNode selection_extensions(const MatchSnapshot& data_,CommandKind kind,std::span<const ItemToken> prefix,SelectionOptions options) {
        selection_boundary(data_);
        SelectionTree tree{data_,data_.state.players[index(data_,*data_.actor)],kind,options.canonical,{},{},{},{}};
        if(data_.phase==MatchPhase::response)tree.context=response_context(data_);
        for(auto token:prefix)tree.push(*instance(tree.own,token));
        return tree.node();
    }''')
    # Device methods cannot catch C++ exceptions. These predicates are not used
    # by the GPU cursor, which follows the legal tree before executing commands.
    for m in reversed(list(re.finditer(r'\bcatch\s*\([^)]*\)\s*\{', match))):
        match = match[:m.start()] + match[_end(match, m.end()-1):]
    match = re.sub(r'\btry\s*', '', match)
    # Runtime tables use read-only spans generated by a separate initialization
    # kernel. This preserves exact catalog iteration and ticket ordering.
    match = _replace_function(match, 'gift_tickets', '''std::span<const CardId> gift_tickets(bool fate) {
        if(fate)gpu_fail(__LINE__); return {gpu_gift_tickets, 499};
    }''')
    match = _replace_function(match, 'dream_faces', '''std::span<const CardId> dream_faces(CardId actual) {
        return {gpu_dream_faces[actual], gpu_dream_count[actual]};
    }''')
    match = _replace_function(match, 'guardian_moves', '''std::span<const CardId> guardian_moves(Guardian guardian) {
        auto i=static_cast<unsigned>(guardian); return {gpu_guardian_moves[i], gpu_guardian_count[i]};
    }''')
    match = match.replace('namespace godfield::official {', '''namespace godfield::official {
    inline constexpr MatchConfig gpu_training_config{};
    __device__ CardId gpu_gift_tickets[500];
    __device__ bool gpu_can_start_main[297];
    __device__ std::int64_t gpu_single_cost[297];
    __device__ std::uint64_t gpu_response_starts[297][2];
    __device__ CardId gpu_dream_faces[297][296];
    __device__ std::size_t gpu_dream_count[297];
    __device__ CardId gpu_guardian_moves[11][20];
    __device__ std::size_t gpu_guardian_count[11];
''', 1)
    # VecEnv rejects every non-default MatchConfig. Specializing these reads
    # removes unreachable external-grant/fate/payload schedules without changing
    # any world accepted by the training API.
    if specialize_config:
        match = re.sub(r'\b\w+\.config\.(\w+)', r'gpu_training_config.\1', match)
        match = re.sub(r'\bconfig_\.(\w+)', r'gpu_training_config.\1', match)
    selection = _read('cpp/gfrl_core/selection.cpp')
    selection = selection[:selection.index('std::vector<std::int32_t> random_choices')] + '\n}\n'
    for m in reversed(list(re.finditer(r'\bcatch\s*\([^)]*\)\s*\{', selection))):
        selection = selection[:m.start()] + selection[_end(selection, m.end()-1):]
    selection = re.sub(r'\btry\s*', '', selection)
    # Training cursors are constructed without public-root restrictions or
    # exclusions. Keep the ordinary CPU selection arithmetic and masks while
    # removing these import/search-only branches from the device call graph.
    selection = selection.replace('if(public_root)', 'if(false)')
    selection = selection.replace('exclusions.tokens.empty()', 'true')
    selection = selection.replace('prune(match);', '(void)match;')
    selection = _replace_function(selection,'Selection::refresh','''template<class M> void Selection::refresh(const M& match,bool canonical) {
        mask.fill(false);if(complete)return;
        const auto& snap=match.snapshot();
        if(stage==Stage::kind) {
            if(snap.phase==gf::MatchPhase::response)mask[int(gf::CommandKind::defend)]=snap.state.players[actor_seat].mp>=0;
            else if(snap.phase==gf::MatchPhase::purchase){mask[int(gf::CommandKind::decline)]=true;mask[int(gf::CommandKind::accept)]=snap.purchase && snap.state.players[actor_seat].cp>=gf::find_card(snap.purchase->item.model)->price;}
            else {DeviceTree tree(snap,actor_seat,false,true);const auto legal=tree.kinds();for(int k=0;k<10;++k)mask[k]=bool(legal&(1u<<k));}
        } else if(stage==Stage::card || stage==Stage::target) {
            DeviceTree tree(snap,actor_seat,canonical,count==0);tree.kind=command.kind;
            for(int j=0;j<count;++j)tree.push(cards[j]);
            const auto node=tree.node(true);
            if(stage==Stage::card){for(int j=0;j<18;++j)mask[j]=bool(node.next&(1u<<j));mask[18]=node.targets;}
            else {for(int j=0;j<3;++j)mask[j]=bool(node.targets&(1u<<j));if(command.kind==gf::CommandKind::exchange)total_=node.total;}
        } else {
            for(int b=0;b<=20;++b){const auto v=bucket_value(b);
                if(stage==Stage::alloc_hp)mask[b]=total_-v>=0 && total_-v<=198;
                else {const auto cp=total_-command.allocation->hp-v;mask[b]=cp>=0 && cp<=99;}
            }
        }
        if(std::count(mask.begin(),mask.end(),true)==0)gpu_fail(__LINE__);
    }''')


    encoder = _read('cpp/gfrl_core/encoder.cpp')
    encoder = encoder.replace('std::vector<Token> encode(', 'void encode(').replace('std::span<const gf::ItemToken> order) {', 'std::span<const gf::ItemToken> order, std::vector<Token>& out) {')
    encoder = encoder.replace('std::vector<Token> out(history_offset);', 'out.resize(history_offset);')
    encoder = encoder.replace('return out;', 'return;')
    for name in ('num_names', 'card_features'):
        encoder = _replace_function(encoder, name)
    history = _read('cpp/gfrl_core/history.cpp')
    if not search:
        for name in ('project_history', 'History::set_projection', 'History::projected'):
            history = _replace_function(history, name)
        history = history.replace('{std::move(token),decision,std::nullopt}', '{std::move(token),decision}')
    else:
        # Index the device ring backwards; same matching order as CPU reverse iterators.
        history = _replace_function(history, 'History::set_projection', '''void History::set_projection(const History& projection) {
            int end=int(projection.events_.size())-1;
            for(int i=int(events_.size())-1;i>=0;--i) {
                auto& src=events_[i];int found=end;
                for(;found>=0;--found) {
                    const auto& p=projection.events_[found];if(p.decision!=src.decision)continue;
                    auto token=p.token;
                    for(auto c:{Actor,Target,Origin,SourceSeat})if(token.cat[c])token.cat[c]=3-token.cat[c];
                    if((token.type==Decision && src.token.type==Decision) || token==src.token)break;
                }
                if(found>=0){src.opposite=projection.events_[found].token;end=found-1;}
            }
        }''')
    text = '\n'.join([*headers, selh, ench, hish, catalog, rules, state, match,
                      _read('cpp/gfrl_gpu/selection.cuh'), selection, encoder, history])
    text = re.sub(r'\bthrow\s+[^;]+;', 'gpu_fail(__LINE__);', text)
    text = text.replace('std::', 'gs::')
    # shared_ptr exists only for the unused public-root restriction path.
    if private_rules:
        text = '\nnamespace {\n' + text + '\n}\n'
    kernels = (ROOT / 'cpp/gfrl_gpu/kernels.cuh').read_text()
    if search:
        from .search_source import search_kernels
        kernels = search_kernels(kernels)
    return (ROOT / 'cpp/gfrl_gpu/compat.cuh').read_text() + text + kernels
