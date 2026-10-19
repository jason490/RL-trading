"""Explicit policy information contracts; native defaults remain unchanged."""
VARIANTS=('native','two_branch_unmasked','two_branch_market_exposure')


def architecture_spec(variant='native',market_features=None):
    if variant not in VARIANTS:raise ValueError('unknown policy architecture')
    if variant=='native':return {'version':1,'variant':'native'}
    if type(market_features) is not int or market_features<1:
        raise ValueError('positive market feature count required')
    return {'version':1,'variant':variant,'market_features':market_features,
            'actor_branches':['market','full_observation'],
            'exposure_reads_account':variant=='two_branch_unmasked',
            'risk_reads_account':True,'critic_reads_account':True}


def validate_loaded_architecture(model,spec,market_features):
    from rl_stocks.rl.ppo.policies import BranchPolicy
    expected=architecture_spec(spec.get('variant'),market_features)
    if spec!=expected:raise ValueError('policy architecture schema mismatch')
    if spec['variant']=='native':
        if isinstance(model.policy,BranchPolicy):raise ValueError('native metadata with branch policy')
        return
    import torch
    p=model.policy
    if (not isinstance(p,BranchPolicy) or p.market_features!=market_features
        or p.direction_market_only!=(spec['variant']=='two_branch_market_exposure')):
        raise ValueError('loaded policy architecture mismatch')
    expected_mask=torch.ones_like(p.action_net.connection_mask)
    if p.direction_market_only:expected_mask[0,p.mlp_extractor.branch_width:]=0
    if not torch.equal(p.action_net.connection_mask,expected_mask):
        raise ValueError('loaded policy connection mask mismatch')
