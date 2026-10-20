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
