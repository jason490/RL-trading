def relative_volume(volumes, window=20):
    result = []
    for i,v in enumerate(volumes):
        denominator = sum(volumes[i-window:i])/window if i>=window else 0
        result.append(v/denominator if denominator > 0 else None)
    return result
