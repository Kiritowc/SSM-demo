"""Inference engine for batch and single predictions."""

from __future__ import annotationsfrom pathlib import Pathfrom typing import Anyimport numpy as npimport pandas as pdimport torchfrom ets.data.datamodule import DataModulefrom ets.data.scaling import inverse_transform_targetsfrom ets.engine.inference import InferenceBundlefrom ets.utils.logger import setup_loggerclass Predictor:
    """Load checkpoint and run inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: dict[str, Any] | None = None,
        device: str = "auto",
        project_root: str | None = None,
        setup_data: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.logger = setup_logger("ets.predictor")
        self.inference = InferenceBundle(
            checkpoint_path=self.checkpoint_path,
            cfg=cfg,
            device=device,
        )
        self.cfg = self.inference.cfg
        self.device = self.inference.device
        self.model = self.inference.model
        self.feature_scaler = self.inference.feature_scaler
        self.target_scaler = self.inference.target_scaler
        self.epoch = self.inference.epoch
        self.metrics = self.inference.metrics
        self.task_type = self.inference.task_type

        self.bundle = None
        if setup_data:
            data_module = DataModule(self.cfg, project_root=project_root)
            self.bundle = data_module.setup()

    def predict_dataloader(self, dataloader) -> dict[str, np.ndarray]:
        """Run inference on a dataloader."""
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                y = batch["y"]
                pred = self.model(x)

                if self.task_type == "classify":
                    pred = pred.argmax(dim=-1)

                preds_list.append(pred.cpu().numpy())
                targets_list.append(y.numpy())

        predictions = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        if self.task_type == "forecast":
            predictions = inverse_transform_targets(self.target_scaler, predictions)
            targets = inverse_transform_targets(self.target_scaler, targets)

        return {"predictions": predictions, "targets": targets}

    def predict_test(self) -> dict[str, np.ndarray]:
        """Run inference on test set."""
        if self.bundle is None:
            raise RuntimeError("setup_data=False; use predict_dataloader() or predict_single().")
        return self.predict_dataloader(self.bundle.test_loader)

    def save_predictions_csv(self, output_path: str | Path) -> Path:
        """Save test predictions to CSV."""
        results = self.predict_test()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.task_type == "classify":
            df = pd.DataFrame(
                {
                    "prediction": results["predictions"],
                    "target": results["targets"],
                }
            )
        else:
            preds = results["predictions"]
            targets = results["targets"]
            if preds.ndim == 1:
                df = pd.DataFrame({"prediction": preds, "target": targets})
            else:
                cols_pred = {f"pred_h{i}": preds[:, i] for i in range(preds.shape[1])}
                cols_tgt = {f"target_h{i}": targets[:, i] for i in range(targets.shape[1])}
                df = pd.DataFrame({**cols_pred, **cols_tgt})

        df.to_csv(output_path, index=False)
        self.logger.info("Predictions saved to %s", output_path)
        return output_path

    def predict_single(self, window: np.ndarray) -> np.ndarray:
        """Predict from a single window array of shape (window_size, num_features)."""
        return self.inference.predict_window(window)
