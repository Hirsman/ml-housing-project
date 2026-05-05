from ml_project.pipeline import run_pipeline


def test_pipeline_returns_metrics():
    metrics = run_pipeline()

    assert "results" in metrics
    assert "best_model" in metrics

    assert "rf" in metrics["results"]

    assert "mae" in metrics["results"]["rf"]
    assert "rmse" in metrics["results"]["rf"]
    assert "r2" in metrics["results"]["rf"]