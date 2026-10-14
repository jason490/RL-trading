def parameters(smoke=False):
    return dict(n_steps=128 if smoke else 2048,batch_size=64,n_epochs=10,
                gae_lambda=.95,clip_range=.2,ent_coef=0.)
