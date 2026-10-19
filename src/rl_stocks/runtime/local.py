import argparse
from datetime import date
from pathlib import Path
from rl_stocks.core.contracts import Request

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    ingest=sub.add_parser("ingest");ingest.add_argument("--provider",choices=["synthetic","files","yahoo","alpaca"],default="synthetic")
    ingest.add_argument("--file");ingest.add_argument("--feed",choices=["iex","sip"],default="iex")
    ingest.add_argument("--output",default="data/raw");ingest.add_argument("--seed",type=int,default=11)
    ingest.add_argument("--universe")
    ingest.add_argument("--full-range",action="store_true",help="One resumable history request per symbol")
    ingest.add_argument("--start",type=date.fromisoformat,default=Request.start,
                        help="Inclusive history start (YYYY-MM-DD)")
    ingest.add_argument("--end",type=date.fromisoformat,default=Request.end,
                        help="Exclusive history end (YYYY-MM-DD)")
    ingest.add_argument("--continue-on-error",action="store_true",
                        help="Record failed provider requests and keep collecting; quality gates still apply")
    sectors=sub.add_parser("ingest-sectors")
    sectors.add_argument("--universe",required=True);sectors.add_argument("--output",default="data/sectors")
    generate=sub.add_parser("dataset");generate.add_argument("snapshot");generate.add_argument("--output",default="data/datasets")
    generate.add_argument("--weekly",action="store_true");generate.add_argument("--allow-unverified",action="store_true")
    generate.add_argument("--supply-demand",nargs="+",choices=["d1","w1"],default=[])
    generate.add_argument("--zone-config",help="Versioned supply/demand JSON; defaults are built in")
    generate.add_argument("--universe");generate.add_argument("--sector-rotation",action="store_true")
    generate.add_argument("--sectors",help="Content-addressed sector snapshot directory")
    generate.add_argument("--sector-labels",action="store_true",help="Requires audited training-era issuer classifications")
    generate.add_argument("--coverage-scope",choices=['snapshot','universe'],default='snapshot')
    generate.add_argument("--development-only",action="store_true",help="Build train/validation only; exclude test")
    generate.add_argument("--evaluation-symbols",nargs='+',help="Explicit common validation/test panel")
    train_parser=sub.add_parser("train");train_parser.add_argument("dataset")
    train_parser.add_argument("--algorithm",choices=["ppo","sac","td3","recurrent_ppo"],required=True)
    train_parser.add_argument("--seed",type=int,default=11);train_parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    train_parser.add_argument("--steps",type=int);train_parser.add_argument("--long",action="store_true")
    train_parser.add_argument("--final-test",action="store_true");train_parser.add_argument("--output",default="artifacts")
    comparison=sub.add_parser("compare")
    comparison.add_argument("runs",nargs="?",default="artifacts/runs")
    comparison.add_argument("--output",default="artifacts/comparison.html")
    args=parser.parse_args()
    universe=None
    if getattr(args,"universe",None):
        from rl_stocks.data.universe.config import read_universe
        universe=read_universe(args.universe)
    if args.command=="ingest":
        from rl_stocks.data.accumulation.backfill import accumulate
        if args.provider=="synthetic":
            from rl_stocks.data.providers.synthetic.adapter import SyntheticProvider
            provider=SyntheticProvider(args.seed)
        elif args.provider=="files":
            from rl_stocks.data.providers.files.adapter import FileProvider
            if not args.file:parser.error("--file required")
            provider=FileProvider(args.file)
        elif args.provider=="yahoo":
            from rl_stocks.data.providers.yahoo.adapter import YahooProvider
            provider=YahooProvider()
        else:
            from rl_stocks.data.providers.alpaca.adapter import AlpacaProvider
            provider=AlpacaProvider(args.feed)
        from rl_stocks.data.universe.config import symbols, history_starts
        request=Request(symbols(universe) if universe else Request.symbols,args.start,args.end)
        print(accumulate(provider,request,args.output,verify_calendar=args.provider!="synthetic",
                         history_starts=history_starts(universe) if universe else None,
                         full_range=args.full_range,continue_on_error=args.continue_on_error))
    elif args.command=="ingest-sectors":
        from rl_stocks.data.providers.yahoo.sectors import YahooSectors
        from rl_stocks.data.universe.sectors import put_sectors
        stocks=[x["symbol"] for x in universe["instruments"] if x["kind"]=="stock"]
        print(put_sectors(YahooSectors().records(stocks),args.output))
    elif args.command=="dataset":
        from rl_stocks.data.generation.datasets import generate
        from rl_stocks.data.universe.sectors import read_sectors
        zone_config=None
        if args.zone_config:
            import json
            from rl_stocks.features.market_structure.supply_demand import ZoneConfig
            value=json.loads(Path(args.zone_config).read_text())
            if not args.supply_demand or value.get("version")!="base-impulse-v1" or value.get("merging") or value.get("dynamic_thresholds"):
                parser.error("zone config requires --supply-demand and supported static, unmerged v1 settings")
            zone_config=ZoneConfig(**value["parameters"])
        print(generate(args.snapshot,args.output,args.weekly,allow_unverified=args.allow_unverified,
                       zone_timeframes=tuple(args.supply_demand),zone_config=zone_config,universe=universe,
                       sector_rotation=args.sector_rotation,
                       sector_records=read_sectors(args.sectors) if args.sectors else (),
                       sector_labels=args.sector_labels,coverage_scope=args.coverage_scope,
                       include_test=not args.development_only,evaluation_symbols=args.evaluation_symbols))
    elif args.command=="compare":
        from rl_stocks.reporting.comparison import compare
        print(compare(args.runs,args.output))
    else:
        from rl_stocks.training.runner import train
        overrides={"n_steps":1000,"batch_size":50} if args.long and args.algorithm in ('ppo','recurrent_ppo') else None
        steps=args.steps if args.steps else 50000 if args.long else None
        print(train(args.dataset,args.algorithm,args.output,args.seed,args.device,not args.long,steps,args.final_test,overrides))

if __name__=="__main__":main()
