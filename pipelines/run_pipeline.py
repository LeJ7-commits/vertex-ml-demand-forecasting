import os
import argparse
from google.cloud import aiplatform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--pipeline_root", required=True)
    ap.add_argument("--template_path", default="pipelines/compiled/pipeline.json")
    ap.add_argument("--dataset", default="demand_fcst")
    ap.add_argument("--features_table", default="features_demand_daily")
    ap.add_argument("--gcs_prefix", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    job = aiplatform.PipelineJob(
        display_name="demand-forecasting-pipeline",
        template_path=args.template_path,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "project_id": args.project_id,            "dataset": args.dataset,
            "features_table": args.features_table,
            "gcs_prefix": args.gcs_prefix,
            "alpha": args.alpha,
            "limit": args.limit,
        },
        enable_caching=True,
    )

    job.run(sync=True, service_account=os.environ["PIPELINE_SA"])
    print("Pipeline run completed.")

if __name__ == "__main__":
    main()
