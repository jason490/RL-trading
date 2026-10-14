from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from rl_stocks.core.files import write_once

class BudgetStop(RuntimeError):pass

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
        if usage.balance<=20 or self.value["consumed"]>=130:raise BudgetStop("insufficient working budget")
        now=self.clock()
        a={"session":session,"status":"allocating","start":now,"last_time":now,
            "initial_balance":usage.balance,"last_balance":usage.balance,"rate":usage.rate,
            "estimated":0.,"measured":0.,"charged":0.,"base_consumed":self.value["consumed"],
            "deadline":now+min(7200.,max(0.,min(130-self.value["consumed"],usage.balance-20))/(usage.rate+.01)*3600),
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
    def guard(self,projected_seconds=60):
        if self.value["budget_stopped"]:
            raise BudgetStop("project halted; teardown or manual recovery required")
        a=self.active;now=self.clock()
        projected=self.value["consumed"]+(now-a["last_time"]+projected_seconds)*(a["rate"]+.01)/3600
        balance=a["last_balance"]-(now-a["last_time"]+projected_seconds)*(a["rate"]+.01)/3600
        if projected>=130 or balance<=20 or now+projected_seconds>=a["deadline"]:
            self.value["budget_stopped"]=True;self.save()
            raise BudgetStop("working budget/deadline reached; begin teardown")
    def finish(self,verified):
        a=self.active
        now=self.clock();a["estimated"]+=(now-a["last_time"])*(a["rate"]+.01)/3600
        a["charged"]=max(a["estimated"],a["measured"])
        self.value["consumed"]=a["base_consumed"]+a["charged"]
        a["status"]="stopped" if verified else "cleanup_failed"
        a["end"]=now;self.save()
