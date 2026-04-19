import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from Models.ModelMaker import ModelMaker
from Models.models.MLPRegression import MLPRegression
from Models.models.DeepEnsemble import DeepEnsemble


class TestModelMaker(unittest.TestCase):

    def test_model_maker_full(self):
        # create a small dummy datase
        data = pd.DataFrame({
            "feat1": np.random.rand(10),
            "feat2": np.random.rand(10),
            "target1": np.random.rand(10)
        })

        mm = ModelMaker()

        # Test Grid Search (train_and_eval)
        # 2 feature combos x 1 param combo = 2 models
        features_list = [["feat1"], ["feat1", "feat2"]]
        params = {
            "hidden_layer_sizes": [(5,)],
            "epochs": [1],
            "tts": [0.2]
        }

        mm.train_and_eval(
            model_name="MLPRegression",
            dataset=data,
            features_list=features_list,
            target_cols=["target1"],
            params=params,
            display_best=False  # Keep the console clean
        )

        # Verify grid search results
        self.assertGreater(len(mm.best_models), 0)

        # Test Saving (save_best)
        best_model = mm.save_best(0)
        self.assertEqual(mm.best, best_model)
        self.assertEqual(len(mm._saved_models), 1)
        self.assertIsInstance(mm.get_saved_model(0), MLPRegression)

        # Test Ensemble Promotion using saved_model[0]
        mm.train_eval_deep_ens(
            dataset=data,
            best_idx=0,
            n_models=2,
            random_state=42
        )

        # Verify the ensemble was built and stored
        ensemble = mm.get_saved_model(1)
        self.assertIsInstance(ensemble, DeepEnsemble)  # TODO
        self.assertEqual(len(ensemble.models), 2)

        # # Cleanup TensorFlow state
        # tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()