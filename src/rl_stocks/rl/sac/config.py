def parameters(smoke=False):
    return dict(buffer_size=100000,batch_size=64 if smoke else 256,
        learning_starts=64 if smoke else 1000,train_freq=1,gradient_steps=1,tau=.005,ent_coef="auto")
