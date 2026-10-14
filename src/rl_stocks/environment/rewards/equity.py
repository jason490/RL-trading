import math

def equity_reward(previous,current):
    return math.log(current/previous) if previous>0 and current>0 else -20.
