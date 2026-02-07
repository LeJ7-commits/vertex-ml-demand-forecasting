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

    ap.add_argument(
        "--trainer_image",
        default=None,
        help="Override trainer image URI. If omitted, uses IMAGE_URI env var.",
    )

    # NEW (Step 9)
    ap.add_argument(
        "--serving_image",
        default=None,
        help="Serving container image URI (for model registry). If omitted, uses trainer_image.",
    )
    ap.add_argument(
        "--model_display_name",
        default="demand-forecasting-conformal",
        help="Vertex Model Registry display name.",
    )
    ap.add_argument(
        "--parent_model",
        default="",
        help="Optional: existing model resource name to create a new VERSION under it.",
    )

    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    trainer_image = args.trainer_image or os.environ.get("IMAGE_URI")
    serving_image = args.serving_image or trainer_image  # default to trainer image if not provided
    pipeline_sa = os.environ.get("PIPELINE_SA")

    if not trainer_image:
        raise RuntimeError("Missing trainer image. Provide --trainer_image or export IMAGE_URI.")
    if not pipeline_sa:
        raise RuntimeError("Missing PIPELINE_SA. Export PIPELINE_SA='...@....iam.gserviceaccount.com'.")

    aiplatform.init(project=args.project_id, location=args.region)

    params = {
        "project_id": args.project_id,
        "region": args.region,
        "dataset": args.dataset,
        "features_table": args.features_table,
        "gcs_prefix": args.gcs_prefix,
        "trainer_image": trainer_image,
        "alpha": float(args.alpha),
        "limit": int(args.limit),

        # NEW (Step 9): these MUST exist as pipeline inputs in pipelines/pipeline.py
        "serving_image": serving_image,
        "model_display_name": args.model_display_name,
        "parent_model": args.parent_model,
    }

    print("Launching pipeline with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    job = aiplatform.PipelineJob(
        display_name="demand-forecasting-pipeline",
        template_path=args.template_path,
        pipeline_root=args.pipeline_root,
        parameter_values=params,
        enable_caching=True,
    )

    job.run(sync=True, service_account=pipeline_sa)
    print("Pipeline run completed.")


if __name__ == "__main__":
    main()
