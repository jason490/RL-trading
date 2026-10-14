import argparse
from pathlib import Path
from rl_stocks.core.contracts import Request

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    ingest=sub.add_parser("ingest");ingest.add_argument("--provider",choices=["synthetic","files","yahoo","alpaca"],default="synthetic")
    ingest.add_argument("--file");ingest.add_argument("--feed",choices=["iex","sip"],default="iex")
    ingest.add_argument("--output",default="data/raw");ingest.add_argument("--seed",type=int,default=11)
    generate=sub.add_parser("dataset");generate.add_argument("snapshot");generate.add_argument("--output",default="data/datasets")
    generate.add_argument("--weekly",action="store_true");generate.add_argument("--allow-unverified",action="store_true")
    train_parser=sub.add_parser("train");train_parser.add_argument("dataset")
    train_parser.add_argument("--algorithm",choices=["ppo","sac","td3"],required=True)
    train_parser.add_argument("--seed",type=int,default=11);train_parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    train_parser.add_argument("--steps",type=int);train_parser.add_argument("--long",action="store_true")
    train_parser.add_argument("--final-test",action="store_true");train_parser.add_argument("--output",default="artifacts")
    comparison=sub.add_parser("compare")
    comparison.add_argument("runs",nargs="?",default="artifacts/runs")
    comparison.add_argument("--output",default="artifacts/comparison.html")
    args=parser.parse_args()
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
        print(accumulate(provider,Request(),args.output,verify_calendar=args.provider!="synthetic"))
    elif args.command=="dataset":
        from rl_stocks.data.generation.datasets import generate
        print(generate(args.snapshot,args.output,args.weekly,allow_unverified=args.allow_unverified))
    elif args.command=="compare":
        from rl_stocks.reporting.comparison import compare
        print(compare(args.runs,args.output))
    else:
        from rl_stocks.training.runner import train
        overrides={"n_steps":1000,"batch_size":50} if args.long and args.algorithm=="ppo" else None
        steps=args.steps if args.steps else 50000 if args.long else None
        print(train(args.dataset,args.algorithm,args.output,args.seed,args.device,not args.long,steps,args.final_test,overrides))

if __name__=="__main__":main()
