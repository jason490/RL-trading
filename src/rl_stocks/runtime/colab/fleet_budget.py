"""One account-level charge for overlapping task-owned Colab allocations."""
from rl_stocks.runtime.colab.budget import BudgetStop,Ledger


class FleetBudget:
    def __init__(self,path,clock=None):
        self.ledger=Ledger(path,**({'clock':clock} if clock is not None else {}))

    @property
    def active(self):return self.ledger.active

    def start(self,name,usage,task_count):
        if type(task_count) is not int or task_count<1:raise ValueError('positive task count required')
        record=self.ledger.begin(name,usage)
        record.update(kind='fleet',children={},status='active',task_count=task_count,
            # The campaign can contain successive waves; each child still gets at most two hours.
            deadline=record['start']+7200*task_count)
        self.ledger.save()

    def admit(self,session,accelerator):
        self.ledger.guard(360)
        if session in self.active['children']:raise ValueError('duplicate owned session')
        now=self.ledger.clock()
        record={'session':session,'accelerator':accelerator,'start':now,'deadline':now+7200,
            'status':'allocating','phase':'new','retrieved':[],'training_verified':False}
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
        if self.ledger.clock()+projected_seconds>=self.active['children'][session]['deadline']:
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
