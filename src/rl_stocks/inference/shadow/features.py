"""Rebuild bundle-declared causal features without fitting preprocessing."""
from rl_stocks.features.registry.assemble import assemble
from rl_stocks.features.market_structure.supply_demand import ZoneConfig
from rl_stocks.features.sectors.rotation import SectorRotation


def latest_features(bars, manifest, session):
    config = manifest['data_manifest']
    zones = config.get('supply_demand', {})
    rotation = config.get('sector_rotation')
    context = None
    if rotation:
        context = SectorRotation(bars, rotation['proxies'], rotation['benchmark'],
                                 rotation['sector_records'],
                                 use_classifications=rotation['issuer_labels_in_observation'])
    result = {}
    for symbol in config['universe']:
        series = [b for b in bars if b.symbol == symbol and b.session <= session]
        rows, names = assemble(series, config.get('weekly', False),
                               tuple(zones.get('timeframes', ())),
                               ZoneConfig(**zones['parameters']) if zones else None)
        if not rows or series[rows[-1]['bar']].session != session:
            raise ValueError(f'insufficient current features: {symbol}')
        row = rows[-1]
        values = list(row['values'])
        if context:
            names += context.names
            values += context.at(symbol, session, series[-1].available_time)
        if names != manifest['names']:
            raise ValueError('unsupported feature contract')
        result[symbol] = (values, row['atr'])
    return result
