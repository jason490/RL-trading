"""One account-level charge for overlapping task-owned Colab allocations."""
from collections import Counter

from rl_stocks.runtime.colab.budget import BudgetStop,Ledger


class FleetBudget:
    def __init__(self,path,clock=None,*,allocation_limit_seconds=7200):
        if type(allocation_limit_seconds) is not int or not 7200<=allocation_limit_seconds<=21600:
            raise ValueError('explicit allocation ceiling between two and six hours required')
        self.allocation_limit_seconds=allocation_limit_seconds
        self.ledger=Ledger(path,**({'clock':clock} if clock is not None else {}))

    @property
    def active(self):return self.ledger.active

    def start(self,name,usage,task_count,*,allocation_plan_seconds=None):
        if type(task_count) is not int or task_count<1:raise ValueError('positive task count required')
        if allocation_plan_seconds is None:
            if self.allocation_limit_seconds>7200:
                raise ValueError('longer allocations require an explicit complete duration plan')
            campaign_seconds=7200*task_count
        else:
            if (type(allocation_plan_seconds) not in (list,tuple) or len(allocation_plan_seconds)!=task_count
                    or any(type(s) is not int or not 1<=s<=self.allocation_limit_seconds for s in allocation_plan_seconds)):
                raise ValueError('one bounded integer duration per registered task required')
            campaign_seconds=sum(allocation_plan_seconds)
        if self.allocation_limit_seconds>7200 and self.ledger.account_teardown_balance<170:
            raise BudgetStop('longer allocations require persisted account floor150/teardown170')
        record=self.ledger.begin(name,usage)
        record.update(kind='fleet',children={},status='active',task_count=task_count,
            # Sum planned allocations bounds successive waves; each child has its own deadline.
            deadline=record['start']+campaign_seconds)
        if allocation_plan_seconds is not None:
            record.update(allocation_plan_seconds=list(allocation_plan_seconds),
                          allocation_limit_seconds=self.allocation_limit_seconds)
        self.ledger.save()

    def admit(self,session,accelerator,allocation_seconds=7200,teardown_reserve_seconds=0):
        if (type(allocation_seconds) is not int or not 1 <= allocation_seconds <= self.allocation_limit_seconds
                or type(teardown_reserve_seconds) is not int or not 0 <= teardown_reserve_seconds < allocation_seconds):
            raise ValueError('bounded allocation deadline and teardown reserve required')
        plan=self.active.get('allocation_plan_seconds')
        if plan is not None:
            used=Counter(c['allocation_seconds'] for c in self.active['children'].values())
            if (self.active['allocation_limit_seconds']!=self.allocation_limit_seconds
                    or Counter(plan)[allocation_seconds]<=used[allocation_seconds]):
                raise ValueError('unclaimed registered duration slot required; no implicit retry')
        self.ledger.guard(360)
        if session in self.active['children']:raise ValueError('duplicate owned session')
        now=self.ledger.clock()
        record={'session':session,'accelerator':accelerator,'start':now,'deadline':now+allocation_seconds,
            'status':'allocating','phase':'new','retrieved':[],'training_verified':False}
        if plan is not None:record['allocation_seconds']=allocation_seconds
        if teardown_reserve_seconds:record['teardown_reserve_seconds']=teardown_reserve_seconds
        self.active['children'][session]=record;self.ledger.save()
        return record

    def observe(self,usage):
        # The account's rate/debit is observed once, never charged once per child.
        self.ledger.observe(usage)
        possible=sum(c['status']!='stopped' for c in self.active['children'].values())
        if usage.assignments>possible:
            self.halt('unowned account assignments prevent isolated aggregate metering')
        return usage

    def halt(self,reason):
        self.ledger.value['budget_stopped']=True
        self.active['metering_unreliable']=True
        self.active['stop_reason']=reason
        self.ledger.save()
        raise BudgetStop(reason)

    def guard(self,session,projected_seconds=60):
        # A large pool must begin teardown early enough to cover its aggregate
        # rate throughout the coordinator's bounded cleanup window.
        self.ledger.guard(max(360,projected_seconds))
        if self.ledger.clock()+max(projected_seconds,self.active['children'][session].get('teardown_reserve_seconds',0))>=self.active['children'][session]['deadline']:
            self.ledger.value['budget_stopped']=True;self.ledger.save()
            raise BudgetStop('owned allocation deadline reached; stop fleet')

    def finish_child(self,session,verified):
        child=self.active['children'][session]
        child.update(status='stopped' if verified else 'cleanup_failed',cleanup_verified=bool(verified),
                     end=self.ledger.clock())
        self.ledger.save()

    def finish(self):
        verified=all(c['status']=='stopped' for c in self.active['children'].values())
        self.ledger.finish(verified)
        return verified
