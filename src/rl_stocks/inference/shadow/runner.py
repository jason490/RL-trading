"""One atomic after-close observation/settlement cycle; never submits orders."""
from dataclasses import asdict, replace
from datetime import date, timedelta
import json
from pathlib import Path
import numpy as np
from rl_stocks.core.contracts import canonical, utcnow
from rl_stocks.data.storage.immutable import load
from rl_stocks.data.validation.bars import timestamp
from rl_stocks.environment.accounting.portfolio import Portfolio
from rl_stocks.environment.execution.simulator import Simulator, SimulationConfig
from rl_stocks.inference.bundle import FrozenPolicy
from rl_stocks.inference.shadow.features import latest_features
from rl_stocks.inference.shadow.journal import transaction, get, set_value, records
from rl_stocks.inference.shadow.protocol import read_protocol
from rl_stocks.risk.decisions import decode, GeometryError


def next_session(session):
    import exchange_calendars as xc
    day = date.fromisoformat(session)
    calendar = xc.get_calendar('XNYS', start=str(day-timedelta(days=10)),
                               end=str(day+timedelta(days=15)))
    target = calendar.next_session(session)
    return str(target.date()), calendar.session_open(target).isoformat()


def economic(bar):
    return {k: v for k, v in asdict(bar).items()
            if k not in ('retrieved_time', 'revision', 'available_time')}


def merged_bars(protocol, snapshots, now):
    values = {}
    for index, path in enumerate([protocol['snapshot'], *snapshots]):
        bars, _ = load(path)
        for bar in bars:
            if bar.symbol not in protocol['required_symbols']:
                raise ValueError('unexpected snapshot symbol')
            expected = protocol['feed_contract']
            if (bar.provider, bar.feed, bar.adjustments) != (
                    expected['provider'], expected['feed'], expected['adjustments']):
                raise ValueError('provider/feed/adjustment mismatch')
            if max(timestamp(bar.retrieved_time), timestamp(bar.available_time)) > now:
                raise ValueError('future/uncompleted snapshot bar')
            key = (bar.symbol, bar.session)
            if key in values:
                if economic(values[key]) != economic(bar):
                    raise ValueError(f'revised observed history: {key}; manual audit required')
            else:
                # A feed's reconstructed publication lag is not proof of our receipt.
                values[key] = replace(bar, available_time=max(
                    timestamp(bar.available_time), timestamp(bar.retrieved_time)).isoformat()) if index else bar
    return sorted(values.values(), key=lambda b: (b.symbol, b.session))


def save_sim(sim):
    portfolio = asdict(sim.portfolio)
    portfolio['events'] = []  # fills are durable in the append-only session record
    return {'portfolio': portfolio, 'protection': sim.protection,
            'near_done': sim.near_done, 'last_protective_action': sim.last_protective_action}


def restore_sim(value, config):
    sim = Simulator(config)
    if value:
        sim.portfolio = Portfolio(**value['portfolio'])
        sim.protection = tuple(value['protection']) if value['protection'] else None
        sim.near_done = value['near_done']
        sim.last_protective_action = (tuple(value['last_protective_action'])
                                      if value['last_protective_action'] else None)
    return sim


def observation(policy, features, sim, price):
    p = sim.portfolio
    equity = p.equity(price)
    if equity <= 0:
        raise ValueError('nonpositive shadow equity; manual review required')
    config = sim.config
    state = [p.quantity*price/equity, p.cash/equity, p.age/config.max_holding,
             (price-p.basis)*p.quantity/equity, equity/config.initial_equity-1]
    return policy.normalizer.transform(features, policy.manifest['names'])+state


def tick(root, snapshot, *, now=None, policy_loader=FrozenPolicy):
    """Use only a previously journaled signal for this session's simulated fills."""
    root = Path(root)
    current = timestamp(now or utcnow())
    protocol = read_protocol(root)
    snapshot = str(Path(snapshot).resolve())
    with transaction(root/'journal.sqlite3') as db:
        if get(db, 'campaign') != root.name:
            raise ValueError('journal campaign mismatch')
        previous_records = records(db)
        settled = sum(bool(r['settlements']) for r in previous_records)
        if settled >= protocol['review_sessions']:
            return {'status': 'review_due', 'settled_sessions': settled}
        snapshots = get(db, 'snapshots', [])
        bars = merged_bars(protocol, [*snapshots, snapshot], current)
        by_symbol = {}
        for bar in bars:
            by_symbol.setdefault(bar.symbol, {})[bar.session] = bar
        latest = {s: max(by_symbol.get(s, {}), default='') for s in protocol['required_symbols']}
        if len(set(latest.values())) != 1:
            raise ValueError('incomplete current symbol/context coverage')
        session = next(iter(latest.values()))
        if not session:
            raise ValueError('no complete session')
        if previous_records and session == previous_records[-1]['session']:
            return {'status': 'already_recorded', 'session': session}
        if previous_records and current <= timestamp(previous_records[-1]['observed_at']):
            raise ValueError('nonmonotonic observation clock')
        if previous_records and session < previous_records[-1]['session']:
            raise ValueError('out-of-order session')
        if timestamp(protocol['not_before']) >= timestamp(by_symbol[protocol['symbols'][0]][session].event_time):
            raise ValueError('pre-registration bars are warmup only')
        following, execution_open = next_session(session)
        if current >= timestamp(execution_open):
            raise ValueError('missed next-session decision deadline; no backfilled decisions')
        for symbol in protocol['required_symbols']:
            bar = by_symbol[symbol][session]
            if (current-timestamp(bar.available_time)).total_seconds() > protocol['max_age_seconds']:
                raise ValueError(f'stale current bar: {symbol}')
        if previous_records and next_session(previous_records[-1]['session'])[0] != session:
            raise ValueError('missed observed session; manual review required, no gap execution')
        # Require continuous exchange-session context after the frozen bootstrap.
        # Gaps in the pre-existing audited archive remain part of its provenance.
        import exchange_calendars as xc
        bootstrap, _ = load(protocol['snapshot'])
        start = min(max(b.session for b in bootstrap if b.symbol == s)
                    for s in protocol['required_symbols'])
        calendar = xc.get_calendar('XNYS', start=start, end=session)
        expected = {str(s.date()) for s in calendar.sessions if start < str(s.date()) <= session}
        for symbol in protocol['required_symbols']:
            bar = by_symbol[symbol][session]
            if timestamp(bar.open_time) != timestamp(calendar.session_open(session).isoformat()) or timestamp(bar.event_time) != timestamp(calendar.session_close(session).isoformat()):
                raise ValueError('bar does not match exchange open/close')
            if not expected <= set(by_symbol[symbol]):
                raise ValueError(f'missing forward context sessions: {symbol}')
        features_cache = {}
        policies = []
        for arm in protocol['arms']:
            manifest_path = Path(arm['bundle'])/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            from rl_stocks.core.contracts import digest
            if digest(manifest) != arm['manifest_sha256']:
                raise ValueError('registered manifest changed')
            policy = policy_loader(arm['bundle'], manifest['feature_schema'], protocol['feed_contract']['feed'])
            feature_key = (manifest['feature_schema'], tuple(manifest['names']))
            if feature_key not in features_cache:
                features_cache[feature_key] = latest_features(bars, manifest, session)
            policies.append((arm, policy, features_cache[feature_key]))
        config = SimulationConfig(**protocol['simulation'])
        record = {'session': session, 'observed_at': current.isoformat(),
                  'snapshot': snapshot, 'snapshot_id': Path(snapshot).name,
                  'decisions': [], 'settlements': [], 'baselines': []}
        for arm, policy, features in policies:
            for symbol in protocol['symbols']:
                key = 'stream:'+arm['name']+':'+symbol
                value = get(db, key, {})
                sim = restore_sim(value.get('sim'), config)
                bar = by_symbol[symbol][session]
                pending = value.get('pending')
                if pending:
                    if pending['execution_session'] != session or timestamp(pending['recorded_at']) >= timestamp(bar.open_time):
                        raise ValueError('signal not recorded before matching execution open')
                    previous = by_symbol[symbol][pending['decision_session']]
                    info = sim.advance(pending['action'], previous, bar, pending['atr'])
                    info.update(source_decision_recorded_at=pending['recorded_at'],
                                arm=arm['name'], family=arm['family'], seed=arm['seed'],
                                symbol=symbol, session=session, price=bar.close,
                                events=list(sim.portfolio.events))
                    record['settlements'].append(info)
                vector, atr = features[symbol]
                obs = observation(policy, vector, sim, bar.close)
                memory = value.get('memory')
                if policy.manifest['metadata']['algorithm'] == 'recurrent_ppo':
                    state = tuple(np.array(x, dtype=np.float32) for x in memory) if memory else None
                    action, state = policy.predict_recurrent(obs, state=state, episode_start=state is None)
                    memory = [x.tolist() for x in state]
                else:
                    action = policy.predict(obs)
                action = [float(x) for x in action]
                if len(action) != 4 or not all(np.isfinite(action)):
                    raise ValueError('invalid policy action')
                proposal = None
                rejection = None
                try:
                    proposal = asdict(decode(action, symbol, current.isoformat(), execution_open,
                                             bar.close, atr, sim.portfolio.equity(bar.close),
                                             sim.portfolio.quantity))
                except GeometryError:
                    rejection = 'nonpositive_protective_price'
                pending = {'decision_session': session, 'execution_session': following,
                           'execution_open': execution_open, 'recorded_at': current.isoformat(),
                           'action': action, 'atr': atr}
                record['decisions'].append({**pending, 'arm': arm['name'], 'symbol': symbol,
                                           'bundle_id': arm['bundle_id'], 'proposal': proposal,
                                           'rejected': rejection})
                set_value(db, key, {'sim': save_sim(sim), 'pending': pending, 'memory': memory})
        # One independent buy-and-hold account per stock; same initial equity/costs.
        if previous_records:
            for symbol in protocol['symbols']:
                value = get(db, 'baseline:'+symbol)
                p = Portfolio(**value) if value else Portfolio(cash=config.initial_equity)
                bar = by_symbol[symbol][session]
                p.corporate_action(bar)
                if value is None:
                    shares = int(p.cash/(bar.open*(1+(config.fee_bps+config.impact_bps)/10000)))
                    p.trade(shares, bar.open, bar.open_time, 'buy_hold', config.fee_bps, config.impact_bps)
                p.age += 1
                record['baselines'].append({'symbol': symbol, 'session': session,
                                            'equity': p.equity(bar.close), 'costs': p.costs,
                                            'cash_equity': config.initial_equity})
                value = asdict(p)
                value['events'] = []
                set_value(db, 'baseline:'+symbol, value)
        db.execute('INSERT INTO sessions VALUES (?,?)', (session, canonical(record).decode()))
        if snapshot not in snapshots:
            snapshots.append(snapshot)
        set_value(db, 'snapshots', snapshots)
        return {'status': 'recorded', 'session': session, 'decisions': len(record['decisions']),
                'settlements': len(record['settlements']),
                'settled_sessions': settled+bool(record['settlements'])}
