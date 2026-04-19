import unittest
import pandas as pd
import numpy as np

from Models.ModelMaker import ModelMaker


class TestModelMaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a custom mini-dataset to speed up routine testing
        np.random.seed(42)
        cls.mini_data = pd.DataFrame({
            "D": np.random.rand(50),
            "P": np.random.rand(50),
            "G": np.random.rand(50),
            "CHF": np.random.rand(50) * 10
        })
        cls.features = [["D", "P", "G"]]
        cls.targets = ["CHF"]

    def setUp(self):
        # Generates a fresh, blank ModelMaker for every individual test
        self.mm = ModelMaker()

    def get_fast_params(self):
        # Using a method ensures we get a fresh dictionary every time.
        # This prevents Trainer.test_train from permanently mutating our test parameters.
        return {
            "epochs": [1],
            "hidden_layer_sizes": [(10,)],
            "tts": [0.2]
        }

    def test_01_mlp_grid_search_and_autosave(self):
        """Test standard grid search, clearing, and the autosave feature."""
        params = self.get_fast_params()
        params["epochs"] = [1, 2]  # Train 2 models

        self.mm.train_and_eval(
            model_name="MLPRegression",
            dataset=self.mini_data,
            features_list=self.features,
            target_cols=self.targets,
            params=params,
            autosave="R2",
            display_best=False
        )

        # Verify autosave grabbed a model
        self.assertIsNotNone(self.mm.best, "Autosave failed to set self.best")
        self.assertEqual(len(self.mm._saved_models), 1, "Model was not appended to saved_models")

        # Verify clear functionality
        self.mm.clear_evaluator()
        self.assertEqual(len(self.mm.eval.models), 0, "Clear evaluator failed to empty models list")

    def test_02_random_search(self):
        """Test that random search restricts the number of models trained."""
        params = self.get_fast_params()
        params["epochs"] = [1, 2]
        params["hidden_layer_sizes"] = [(10,), (20,)]

        self.mm.train_and_eval(
            model_name="MLPRegression",
            dataset=self.mini_data,
            features_list=self.features,
            target_cols=self.targets,
            params=params,
            random_search=1,  # Should only train 1 out of the 4 combos
            display_best=False
        )

        self.assertEqual(len(self.mm.trainer.models), 1, "Random search did not restrict model count")

    def test_03_deep_ensemble_via_autosave_model(self):
        """Test Deep Ensemble creation using the implicitly saved self.best model."""
        # 1. Train and autosave a base model
        self.mm.train_and_eval(
            model_name="MLPRegression",
            dataset=self.mini_data,
            features_list=self.features,
            target_cols=self.targets,
            params=self.get_fast_params(),
            autosave="R2",
            display_best=False
        )

        # 2. Clear evaluator so we isolate the DeepEnsemble in our assertions
        self.mm.clear_evaluator()

        # 3. Train ensemble using that autosaved base model
        self.mm.train_eval_deep_ens(
            dataset=self.mini_data,
            n_models=2,
            auto_save=False
        )

        ensemble = self.mm.eval.models[0]
        self.assertEqual(ensemble.__class__.__name__, "DeepEnsemble")
        self.assertEqual(len(ensemble.models), 2, "Ensemble did not generate the correct number of members")

    def test_04_deep_ensemble_via_saved_idx(self):
        """Test Deep Ensemble creation by referencing a manually saved model index."""
        # 1. Train and manually save to index 0
        self.mm.train_and_eval(
            model_name="MLPRegression",
            dataset=self.mini_data,
            features_list=self.features,
            target_cols=self.targets,
            params=self.get_fast_params(),
            display_best=False
        )
        self.mm.save_best(0)
        self.mm.clear_evaluator()

        # 2. Train ensemble using the model at index 0
        self.mm.train_eval_deep_ens(
            dataset=self.mini_data,
            best_idx=0,
            n_models=2,
            auto_save=False
        )

        ensemble = self.mm.eval.models[0]
        self.assertEqual(ensemble.__class__.__name__, "DeepEnsemble")
        self.assertEqual(len(ensemble.models), 2)

    def test_05_deep_ensemble_via_custom_params_and_grid(self):
        """Test Deep Ensemble creation from scratch and its own autosave feature."""
        params = self.get_fast_params()
        params["epochs"] = [1, 2]  # We force a grid search so auto_save actually has multiple models to rank

        self.mm.train_eval_deep_ens(
            dataset=self.mini_data,
            base_model="MLPRegression",
            features=self.features,
            target_cols=self.targets,
            params=params,
            n_models=3,
            auto_save="R2"
        )

        self.assertIsNotNone(self.mm.best)
        self.assertEqual(self.mm.best.__class__.__name__, "DeepEnsemble")
        self.assertEqual(len(self.mm.best.models), 3)

    def test_06_preinstalled_datasets(self):
        """Test that the built-in database factory successfully loads CHF and Reactor data."""
        self.mm.train_and_eval(
            model_name="MLPRegression",
            dataset="CHF",
            features_list=None,
            params=self.get_fast_params(),
            display_best=False
        )
        self.assertGreater(len(self.mm.trainer.models), 0, "Failed to train on built-in CHF dataset")


if __name__ == '__main__':
    unittest.main()