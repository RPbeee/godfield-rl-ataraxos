"""Search-only extensions to the generated engine; ordinary GPU PPO is unchanged."""
import re

from .source import ROOT, _replace_function


def checked_determinization(match):
    """Return CPU world rejection codes instead of trapping on postconditions.

    Invalid assignment shape/catalog/capacity is checked before this call.
    The original equality predicates and their order remain unchanged.
    """
    def rewrite(body):
        start=body.index('Match result = *this;')
        head,tail=body[:start],body[start:]
        tail=tail.replace('result.restore(s);','result.restore(s); result.gpu_search_status=0;')
        spans=[]
        tokens=re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[()]',re.S)
        for m in re.finditer(r'\brequire\(',tail):
            depth=1
            for token in tokens.finditer(tail,m.end()):
                if token[0]=='(':depth+=1
                elif token[0]==')':
                    depth-=1
                    if not depth:
                        call=tail[m.end():token.start()]
                        message=re.search(r',\s*"([^"]+)"\s*$',call)
                        if message is None:raise ValueError('unknown determinization predicate')
                        reason=message[1]
                        code=4 if 'observation differs' in reason else 5 if 'decision differs' in reason else 6 if 'differs' in reason else 1
                        spans.append((m.start(),token.end()+1,
                            f'if(!({call[:message.start()]})){{result.gpu_search_status={code};return result;}}'))
                        break
        for a,b,value in reversed(spans):tail=tail[:a]+value+tail[b:]
        return head+tail
    pattern=re.search(r'^Match Match::determinized_impl\([^;{}]*\) const\s*\{',match,re.M)
    if pattern is None:raise ValueError('missing determinization implementation')
    from .source import _end
    end=_end(match,pattern.end()-1)
    return match[:pattern.start()]+rewrite(match[pattern.start():end])+match[end:]


def search_kernels(kernels):
    kernels = kernels.replace('bool encoded=false,dirty=false;', '''bool encoded=false,dirty=false;
    bool search=false,tracking=false,waiting=false;
    int searcher=-1,max_depth=512,status=2,view_seat=0,parallelism=0;
    long long games=-1;unsigned long long game=0,eval_seed=0;
    gf::MatchView projected_before[2];''')
    kernels = kernels.replace('void reset(Env& e,Outputs o,int i,unsigned long long seed) {',
        '''void stop(Env& e,Outputs o,int i) {
        e.waiting=true;e.selection.reset();
        o.stage[i]=o.actor_seat[i]=o.chosen_kind[i]=o.chosen_target[i]=o.chosen_alloc_hp[i]=-1;
        o.chosen_count[i]=0;o.substep[i]=0;for(int j=0;j<21;++j)o.mask[21*i+j]=false;
        for(int j=0;j<18;++j)o.chosen_cards[18*i+j]=-1;
    }
    void reset(Env& e,Outputs o,int i,unsigned long long seed) {
        if(e.games>=0 && e.game>=e.games){stop(e,o,i);return;}''')
    kernels = kernels.replace('e.selection.reset();e.encoded=false;for(auto& h:e.history)h.clear();o.episode_id[i]=e.episode;',
        '''if(e.games>=0) {
            auto root=mix(e.eval_seed^mix(e.game/2)^0x4556414c50414952ULL);
            e.match.reset(p,mix(root^0x13198a2e03707344ULL));
            e.chance_rng=mix(root^0xa4093822299f31d0ULL);
        }
        e.selection.reset();e.encoded=false;e.waiting=false;
        for(auto& h:e.history)h.clear();o.episode_id[i]=e.games>=0?e.game:e.episode;''')
    kernels = _replace_function(kernels, 'before', '''void before(Env& e) {
        for(int s=0;s<2;++s){e.before[s]=e.match.view(e.match.snapshot().state.players[s].id,true);
            if(e.tracking)e.projected_before[s]=project_history(e.before[s],e.match.snapshot(),s);}
    }''')
    kernels = _replace_function(kernels, 'record', '''void record(Env& e,Outputs o,int i,bool automatic) {
        for(int s=0;s<2;++s) {
            auto a=e.match.view(e.before[s].board.viewer,true);auto d=e.match.last_decision(e.before[s].board.viewer);
            if(!d)gpu_fail(__LINE__);e.history[s].append(e.before[s],a,*d,*e.before[s].board.viewer);
            if(e.tracking) {
                const auto& snap=e.match.snapshot();const int other=1-s;
                if(snap.last_decision && (snap.last_decision->hidden_viewers&(1u<<other))){d->faces.clear();d->faces_hidden=true;}
                History projected(4096);
                projected.append(e.projected_before[s],project_history(a,snap,s),*d,snap.state.players[other].id);
                e.history[s].set_projection(projected);
            }
        }
        e.encoded=false;o.stats[12*i+1]++;o.stats[12*i+3]+=automatic;
        if(!e.search || !automatic)o.decision_done[i]=true;
        if(e.search && !automatic)++o.decisions[i];e.selection.reset();
    }''')
    kernels = kernels.replace('for(;;) {\n        const auto& snap=e.match.snapshot();',
                              'for(;;) {\n        if(e.waiting)return;\n        const auto& snap=e.match.snapshot();')
    kernels = kernels.replace('if(e.match.done()) {', '''if(e.match.done()) {
            if(e.search){o.episode_done[i]=true;o.truncated[i]=snap.truncated;
                for(int s=0;s<2;++s)o.outcome[2*i+s]=snap.truncated?0:snap.rewards[s];
                stop(e,o,i);return;}''')
    kernels = kernels.replace('++e.episode;reset(e,o,i,seed);continue;',
                              '++e.episode;e.game+=e.games>=0?e.parallelism:0;reset(e,o,i,seed);continue;')
    kernels = kernels.replace('if(count>=2){if(encode_now)', '''if(count>=2){
            if(e.search && o.decisions[i]>=e.max_depth){o.truncated[i]=true;o.episode_done[i]=true;stop(e,o,i);return;}
            if(encode_now)''')
    kernels = kernels.replace('auto& e=((Env*)storage)[i];auto o=outputs(pointers);int c=choices[i];',
        'auto& e=((Env*)storage)[i];auto o=outputs(pointers);if(e.waiting){idle(e,o,i);return;}int c=choices[i];')
    kernels = kernels.replace('auto& e=((Env*)storage)[i];int c=choices[i];',
        'auto& e=((Env*)storage)[i];if(e.waiting){idle(e,outputs(pointers),i);return;}int c=choices[i];')
    kernels = kernels.replace('void reset(Env& e,Outputs o,int i,unsigned long long seed) {', '''void idle(Env& e,Outputs o,int i) {
        o.decision_done[i]=false;
        if(!e.search){o.episode_done[i]=o.truncated[i]=false;o.outcome[2*i]=o.outcome[2*i+1]=0;}
    }
    void reset(Env& e,Outputs o,int i,unsigned long long seed) {''')
    kernels = kernels.replace('if(e.encoded)return;', 'if(e.encoded || e.waiting)return;')
    kernels = kernels.replace('if(choices[i]<0)return;auto& e=((Env*)storage)[i];',
                              'if(choices[i]<0)return;auto& e=((Env*)storage)[i];if(e.waiting)return;')
    kernels = kernels.replace('if(!e.dirty)return;', 'if(!e.dirty)return;')
    kernels = kernels.replace('e.history[e.selection->actor_seat].events()', 'e.history[e.view_seat].events()')
    kernels = kernels.replace('1-e.selection->actor_seat', '1-e.view_seat')
    kernels = kernels.replace('const auto& s=*e.selection;\n    if(!e.encoded)',
                              'const auto& s=*e.selection;e.view_seat=s.actor_seat;\n    if(!e.encoded)')
    return kernels + (ROOT/'cpp/gfrl_gpu/search.cuh').read_text()
