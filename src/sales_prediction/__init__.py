def main():
    # entry point for "uv run sales-prediction"
    # (import inside the function, so "python -m sales_prediction.features" has no warning)
    from sales_prediction.features import main as run_pipeline

    run_pipeline()
