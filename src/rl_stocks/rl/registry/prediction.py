"""Keep recurrent evaluation state local to a single episode."""
import numpy as np


def recurrent(model):
    return getattr(getattr(model,'policy',None),'lstm_actor',None) is not None


class PredictionStream:
    def __init__(self,model):
        self.model=model;self.state=None;self.episode_start=True

    def predict(self,observation):
        if recurrent(self.model):
            action,self.state=self.model.predict(observation,state=self.state,
                episode_start=np.array([self.episode_start],dtype=bool),deterministic=True)
            self.episode_start=False
            return action
        return self.model.predict(observation,deterministic=True)[0]
