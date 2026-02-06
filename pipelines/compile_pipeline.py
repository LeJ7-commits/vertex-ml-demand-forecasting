import argparse
from kfp import compiler
from pipelines.pipeline import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="pipelines/compiled/pipeline.json")
    args = ap.parse_args()

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=args.output,
    )
    print(f"Compiled pipeline to: {args.output}")

if __name__ == "__main__":
    main()
