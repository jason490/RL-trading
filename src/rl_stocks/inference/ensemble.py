"""Fixed uniform averages of compatible stateless, normalized-action predictors."""
from copy import deepcopy
import numpy as np


def input_contract(frozen):
    m=frozen.manifest;cap=m['data_manifest']['provenance']['capabilities']
    return {'feature_schema':m['feature_schema'],'names':m['names'],'normalizer':m['normalizer'],
            'action_schema':m['action_schema'],'feed':cap['feed'],'adjustments':cap['adjustments'],
            'observation':frozen.observation_spec,'observation_size':frozen.observation_size}


class FixedMember:
    def __init__(self,model,contract,identity,weights_sha256):
        from rl_stocks.rl.registry.prediction import recurrent
        if recurrent(model):raise ValueError('ensemble member must be stateless')
        if (model.observation_space.shape!=(contract['observation_size'],) or model.action_space.shape!=(4,)
            or not np.all(model.action_space.low==-1) or not np.all(model.action_space.high==1)):
            raise ValueError('member input/action shape mismatch')
        self.model=model;self.contract=deepcopy(contract);self.identity=identity;self.weights_sha256=weights_sha256
        self.model.policy.set_training_mode(False)
    def action(self,observation):
        return self.model.predict(observation,deterministic=True)[0]


class UniformEnsemble:
    def __init__(self,members):
        self.members=tuple(members)
        if not self.members or len({m.identity for m in self.members})!=len(self.members):
            raise ValueError('nonempty unique fixed member list required')
        self.contract=deepcopy(self.members[0].contract)
        if any(m.contract!=self.contract for m in self.members):
            raise ValueError('ensemble feature/feed/normalizer/action/state contracts differ')
    def predict(self,observation,deterministic=True):
        if deterministic is not True:raise ValueError('frozen ensemble requires deterministic inference')
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape!=(self.contract['observation_size'],) or not np.isfinite(observation).all():
            raise ValueError('ensemble observation shape/nonfinite mismatch')
        actions=[]
        for member in self.members:
            action=np.asarray(member.action(observation.copy()),dtype=np.float32)
            if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
                raise ValueError('member must return four finite bounded action components')
            actions.append(action)
        return np.mean(np.stack(actions),axis=0,dtype=np.float32),None
    def recipe(self):
        return {'version':1,'kind':'uniform_stateless_actions','contract':deepcopy(self.contract),
                'members':[{'identity':m.identity,'weights_sha256':m.weights_sha256,
                            'weight':1/len(self.members)} for m in self.members],
                'aggregation':'Arithmetic mean of deterministic bounded four-component predictions on shared account state'}

    @classmethod
    def from_recipe(cls,recipe,members):
        if recipe.get('version')!=1 or recipe.get('kind')!='uniform_stateless_actions':
            raise ValueError('unknown ensemble recipe')
        try:result=cls([members[item['identity']] for item in recipe['members']])
        except KeyError:raise ValueError('missing ensemble member') from None
        if result.recipe()!=recipe:raise ValueError('ensemble member/hash/weight/contract mismatch')
        return result
