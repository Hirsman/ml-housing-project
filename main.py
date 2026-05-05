from ml_project.pipeline import run_pipeline

if __name__ == "__main__":
    metrics = run_pipeline()

    print("\nPipeline terminé avec succès.")

    best_model = metrics["best_model"]
    best_metrics = metrics["results"][best_model]

    print(f"Best model : {best_model}")
    print(f"MAE  : {best_metrics['mae']:.4f}")
    print(f"RMSE : {best_metrics['rmse']:.4f}")
    print(f"R2   : {best_metrics['r2']:.4f}")
