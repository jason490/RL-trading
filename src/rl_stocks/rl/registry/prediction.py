"""Keep recurrent evaluation state local to a single episode."""
import numpy as np


def recurrent(model):
    return getattr(getattr(model,'policy',None),'lstm_actor',None) is not None


class PredictionStream:
    def __init__(self,model):
        self.model=model;self.state=None;self.episode_start=True

    def predict(self,observation):
        from rl_stocks.rl.recurrent_sac.prediction import FrozenRecurrentActor
        if isinstance(self.model,FrozenRecurrentActor):
            action,self.state=self.model.predict(observation,observation_schema=self.model.observation_schema,
                state=self.state,episode_start=self.episode_start)
            self.episode_start=False
            return action
        if recurrent(self.model):
            action,self.state=self.model.predict(observation,state=self.state,
                episode_start=np.array([self.episode_start],dtype=bool),deterministic=True)
            self.episode_start=False
            return action
        return self.model.predict(observation,deterministic=True)[0]
