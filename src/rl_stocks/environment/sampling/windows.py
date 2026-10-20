"""Uniform training episodes with full fixed-length transition windows."""


class FullWindows:
    mode='full_window_uniform'

    def __init__(self,episodes,steps):
        if type(steps) is not int or steps<1:raise ValueError('positive integer window required')
        self.bounds=[];self.groups=[]
        for episode in episodes:
            bars=episode['bars']
            if (episode['split']!='train' or len(bars)<=steps
                or any(b['session']>'2020-12-31' for b in bars)):
                raise ValueError('full windows require sufficient training-only bars')
            if any(a['session']>=b['session'] for a,b in zip(bars,bars[1:])):
                raise ValueError('strictly ordered training bars required')
            periods={'earlier' if b['session']<'2010-01-01' else 'recent' for b in bars}
            if len(periods)!=1:raise ValueError('history periods must remain separate episodes')
            self.bounds.append(len(bars)-steps)
            self.groups.append(episode['symbol']+':'+next(iter(periods)))
        if not self.bounds:raise ValueError('training episodes required')
        self.summary=dict(mode=self.mode,window_transitions=steps,episode_count=len(episodes),
            selection='Uniform episode, then uniform start permitting the entire window',
            future_outcomes_used=False,downtrend_counts_measured=False)

    def choose(self,rng):
        index=int(rng.integers(len(self.bounds)))
        return index,int(rng.integers(self.bounds[index]))

    def label(self,episode,index):
        # Common sampler interface; no downtrend/outcome classes are computed.
        return False


class StockEtfCycles(FullWindows):
    """Fixed original-panel group mass, shuffled per-group episode coverage."""
    mode='stock_etf_cycle_windows'
    etfs=('SPY','QQQ','IWM','DIA')

    def __init__(self,episodes,steps):
        super().__init__(episodes,steps)
        symbols=[e['symbol'] for e in episodes]
        if len(symbols)!=len(set(symbols)) or not set(self.etfs)<=set(symbols):
            raise ValueError('unique symbols and all four broad ETFs required')
        self.buckets={kind:[i for i,s in enumerate(symbols) if (s in self.etfs)==(kind=='etf')]
                      for kind in ('stock','etf')}
        if not self.buckets['stock']:raise ValueError('stock episodes required')
        self.reset_cycles()
        self.summary.update(selection='Fixed group probability, shuffled episode cycles within group, uniform full-window start',
            group_probability=dict(stock_numerator=22,etf_numerator=4,denominator=26),
            etf_symbols=list(self.etfs),episode_cycle='Without replacement until group exhausted',
            group_episodes={k:len(v) for k,v in self.buckets.items()})

    def reset_cycles(self):
        self.decks={'stock':[],'etf':[]}

    def choose(self,rng):
        kind='etf' if int(rng.integers(26))<4 else 'stock'
        if not self.decks[kind]:self.decks[kind]=[int(i) for i in rng.permutation(self.buckets[kind])]
        index=self.decks[kind].pop()
        return index,int(rng.integers(self.bounds[index]))
