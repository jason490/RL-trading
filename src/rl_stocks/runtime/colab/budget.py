from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from rl_stocks.core.files import write_once

class BudgetStop(RuntimeError):pass

def execution_window(remaining_seconds, preferred_seconds, reserve_seconds=600):
    """Leave time for three final downloads, metering and verified shutdown."""
    if not all(math.isfinite(v) and v>0 for v in (remaining_seconds,preferred_seconds,reserve_seconds)):
        raise BudgetStop('invalid execution deadline')
    available=math.floor(remaining_seconds-reserve_seconds)-1
    if available<60:raise BudgetStop('insufficient execution time after retrieval/cleanup reserve')
    return min(preferred_seconds,available)

@dataclass(frozen=True)
class Usage:
    balance: float
    rate: float
    assignments: int

def parse_usage(text):
    patterns=(r"Current balance:\s*([0-9.]+) compute units",r"Usage rate:\s*([0-9.]+)/hr",r"Active assignments:\s*(\d+)")
    matches=[re.search(p,text) for p in patterns]
    if not all(matches):raise BudgetStop("unparseable metering; stop allocation")
    values=[float(m.group(1)) for m in matches]
    if not all(math.isfinite(v) and v>=0 for v in values):raise BudgetStop("invalid metering")
    return Usage(values[0],values[1],int(values[2]))

class Ledger:
    def __init__(self,path,clock=time.time):
        self.path=Path(path);self.clock=clock
        self.value=json.loads(self.path.read_text()) if self.path.exists() else {
            "ceiling":150.,"working_limit":130.,"consumed":0.,"attempts":[],"budget_stopped":False}
        if self.value.get("ceiling")!=150 or self.value.get("working_limit")!=130:
            raise BudgetStop("ledger budget changed")
    @property
    def account_teardown_balance(self):
        floor=self.value.get('minimum_account_balance',0.)
        if not isinstance(floor,(int,float)) or not math.isfinite(floor) or floor<0:
            raise BudgetStop('invalid minimum account balance')
        return floor+20.

    def review_resume(self,usage,authorization,stopped_session,minimum_account_balance):
        if not isinstance(authorization,str) or not authorization.strip():
            raise BudgetStop('explicit reviewed authorization required')
        if not self.value['budget_stopped'] or not self.value['attempts'] or self.active['session']!=stopped_session:
            raise BudgetStop('exact stopped attempt required for review')
        if any(a['status']!='stopped' for a in self.value['attempts']):
            raise BudgetStop('unresolved owned sessions prevent resumption')
        if any(c.get('status')!='stopped' or not c.get('cleanup_verified') for c in self.active.get('children',{}).values()):
            raise BudgetStop('verified owned cleanup required')
        if (not isinstance(minimum_account_balance,(int,float)) or not math.isfinite(minimum_account_balance)
            or minimum_account_balance<self.value.get('minimum_account_balance',0.)):
            raise BudgetStop('invalid or reduced account floor')
        if (not all(math.isfinite(v) and v>=0 for v in (usage.balance,usage.rate))
            or usage.assignments!=0 or usage.rate!=0 or usage.balance<=minimum_account_balance+20
            or self.value['consumed']>=130):
            raise BudgetStop('healthy idle metering and working budget required')
        review=dict(time=self.clock(),authorization=authorization.strip(),stopped_session=stopped_session,
            previous_stop_reason=self.active.get('stop_reason'),consumed_preserved=self.value['consumed'],
            account_balance=usage.balance,minimum_account_balance=minimum_account_balance,
            account_teardown_balance=minimum_account_balance+20,assignments=usage.assignments,rate=usage.rate)
        self.value.setdefault('resumption_reviews',[]).append(review)
        self.value['minimum_account_balance']=minimum_account_balance
        self.value['budget_stopped']=False
        self.save()
        return review

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix(".tmp")
        with temp.open("w") as stream:
            stream.write(json.dumps(self.value,indent=2))
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temp.replace(self.path)
    def begin(self,session,usage):
        if self.value["budget_stopped"]:raise BudgetStop("budget stop requires manual review; no automatic restart")
        if any(a["status"]!="stopped" for a in self.value["attempts"]):
            raise BudgetStop("unresolved owned session; recover it before allocating")
        if usage.assignments!=0:raise BudgetStop("existing account assignments prevent isolated metering")
        if usage.balance<=self.account_teardown_balance or self.value["consumed"]>=130:raise BudgetStop("insufficient working budget")
        now=self.clock()
        a={"session":session,"status":"allocating","start":now,"last_time":now,
            "initial_balance":usage.balance,"last_balance":usage.balance,"rate":usage.rate,
            "estimated":0.,"measured":0.,"charged":0.,"base_consumed":self.value["consumed"],
            "deadline":now+min(7200.,max(0.,min(130-self.value["consumed"],usage.balance-self.account_teardown_balance))/(usage.rate+.01)*3600),
            "retrieved":[],"meter_readings":[]}
        self.value["attempts"].append(a);self.save()
        return a
    @property
    def active(self):return self.value["attempts"][-1]
    def observe(self,usage):
        a=self.active;now=self.clock()
        # Use the larger old/new rate over the elapsed interval, conservatively.
        a["estimated"]+=(now-a["last_time"])*(max(a["rate"],usage.rate)+.01)/3600
        a["measured"]=max(a["measured"],a["initial_balance"]-usage.balance)
        a.update(last_time=now,last_balance=usage.balance,rate=usage.rate)
        a["charged"]=max(a["estimated"],a["measured"])
        self.value["consumed"]=a["base_consumed"]+a["charged"]
        a["meter_readings"].append({"time":now,"balance":usage.balance,"rate":usage.rate,"assignments":usage.assignments})
        self.save()
    def settle_stopped(self,session,usage):
        """Record delayed idle-account debits without charging unallocated time."""
        if (not self.value['attempts'] or self.active['session']!=session
                or any(a['status']!='stopped' for a in self.value['attempts'])
                or type(usage.assignments) is not int or usage.assignments!=0
                or type(usage.rate) not in (int,float) or usage.rate!=0
                or type(usage.balance) not in (int,float) or not math.isfinite(usage.balance) or usage.balance<0):
            raise BudgetStop('exact latest stopped attempt and reliable idle account reading required')
        account_teardown=self.account_teardown_balance
        a=self.active
        previous=self.value['consumed']
        a['measured']=max(a['measured'],a['initial_balance']-usage.balance)
        a['charged']=max(a['charged'],a['estimated'],a['measured'])
        self.value['consumed']=max(previous,a['base_consumed']+a['charged'])
        a.setdefault('settlement_readings',[]).append(dict(time=self.clock(),balance=usage.balance,
            rate=usage.rate,assignments=usage.assignments,project_before=previous,
            project_after=self.value['consumed']))
        a['settled_balance']=usage.balance
        if self.value['consumed']>=130 or usage.balance<=account_teardown:
            self.value['budget_stopped']=True
        self.save()
        return self.value['consumed']-previous

    def guard(self,projected_seconds=60):
        if self.value["budget_stopped"]:
            raise BudgetStop("project halted; teardown or manual recovery required")
        a=self.active;now=self.clock()
        projected=self.value["consumed"]+(now-a["last_time"]+projected_seconds)*(a["rate"]+.01)/3600
        balance=a["last_balance"]-(now-a["last_time"]+projected_seconds)*(a["rate"]+.01)/3600
        if projected>=130 or balance<=self.account_teardown_balance or now+projected_seconds>=a["deadline"]:
            self.value["budget_stopped"]=True;self.save()
            raise BudgetStop("working budget/deadline reached; begin teardown")
    def finish(self,verified):
        a=self.active
        now=self.clock();a["estimated"]+=(now-a["last_time"])*(a["rate"]+.01)/3600
        a["charged"]=max(a["estimated"],a["measured"])
        self.value["consumed"]=a["base_consumed"]+a["charged"]
        a["status"]="stopped" if verified else "cleanup_failed"
        a["end"]=now;self.save()
