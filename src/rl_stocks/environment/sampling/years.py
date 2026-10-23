"""Measured training visitation by decision and execution year; no outcomes."""
from copy import deepcopy


class YearCounts:
    def __init__(self):
        self.starts={};self.decisions={};self.executions={};self.pairs={};self.symbols={}

    @staticmethod
    def add(counts,key):
        counts[key]=counts.get(key,0)+1

    def start(self,symbol,session):
        self.add(self.starts,session[:4])
        row=self.symbols.setdefault(symbol,dict(starts=0,transitions=0,decision_years={},execution_years={}))
        row['starts']+=1

    def transition(self,symbol,decision,execution):
        d,e=decision[:4],execution[:4]
        self.add(self.decisions,d);self.add(self.executions,e);self.add(self.pairs,d+'->'+e)
        row=self.symbols.setdefault(symbol,dict(starts=0,transitions=0,decision_years={},execution_years={}))
        row['transitions']+=1
        self.add(row['decision_years'],d);self.add(row['execution_years'],e)

    def snapshot(self,starts,transitions):
        result=dict(version=1,measured=True,start_years=self.starts,decision_years=self.decisions,
            execution_years=self.executions,decision_execution_year_pairs=self.pairs,per_symbol=self.symbols)
        validate_year_counts(result,starts,transitions)
        return deepcopy(result)


def validate_year_counts(record,starts,transitions):
    if (record.get('version')!=1 or record.get('measured') is not True
        or type(starts) is not int or starts<0 or type(transitions) is not int or transitions<0):
        raise ValueError('measured year-count contract required')
    def counts(value,total):
        if (not isinstance(value,dict) or any(type(n) is not int or n<0 for n in value.values())
            or sum(value.values())!=total):raise ValueError('year-count totals mismatch')
    def years(value,total):
        counts(value,total)
        if any(not isinstance(y,str) or len(y)!=4 or not y.isdigit() or not 1900<=int(y)<=2020 for y in value):
            raise ValueError('training year keys required')
    years(record['start_years'],starts)
    years(record['decision_years'],transitions);years(record['execution_years'],transitions)
    counts(record['decision_execution_year_pairs'],transitions)
    decision={};execution={}
    for key,n in record['decision_execution_year_pairs'].items():
        if not isinstance(key,str) or len(key)!=10 or key[4:6]!='->':raise ValueError('year pair required')
        d,e=key.split('->');years({d:n},n);years({e:n},n)
        if int(e)<int(d) or int(e)>int(d)+1:raise ValueError('ordered adjacent year pair required')
        decision[d]=decision.get(d,0)+n;execution[e]=execution.get(e,0)+n
    if decision!=record['decision_years'] or execution!=record['execution_years']:
        raise ValueError('year-pair margins mismatch')
    per=record['per_symbol']
    if not isinstance(per,dict) or any(not isinstance(s,str) or not s for s in per):raise ValueError('symbol counts required')
    ds={};es={}
    for row in per.values():
        if any(type(row[k]) is not int or row[k]<0 for k in ['starts','transitions']):raise ValueError('symbol totals required')
        years(row['decision_years'],row['transitions']);years(row['execution_years'],row['transitions'])
        for key,dest in [('decision_years',ds),('execution_years',es)]:
            for y,n in row[key].items():dest[y]=dest.get(y,0)+n
    if (sum(r['starts'] for r in per.values())!=starts or sum(r['transitions'] for r in per.values())!=transitions
        or ds!=record['decision_years'] or es!=record['execution_years']):
        raise ValueError('symbol-year margins mismatch')
    return record
