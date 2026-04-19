import unittest
import pandas as pd
import numpy as np

from Models.models.DeepEnsemble import DeepEnsemble
from Models.models.MLPRegression import MLPRegression
from Models.src.Trainer import Trainer


class TestTrainer(unittest.TestCase):

    def test_trainer(self):
        # Create tiny dummy dataset
        data = pd.DataFrame({
            "feat1": np.random.rand(10),
            "feat2": np.random.rand(10),
            "target1": np.random.rand(10)
        })

        trainer = Trainer()

        # multiple features to test grid search
        features_list = [["feat1"], ["feat1", "feat2"]]
        target_cols = ["target1"]

        params = {
            "hidden_layer_sizes": [(10,)],
            "epochs": [1, 2],
            "tts": [0.2]
        }
        # combo == 2 features * 2 params = 4
        total_models = 4

        # # Verify trainer works with custom data
        # models = trainer.test_train(
        #     model_name="MLPRegression",
        #     dataset=data,
        #     params=params,
        #     features_list=features_list,
        #     target_cols=target_cols
        # )
        #
        # self.assertEqual(len(models), total_models)
        # self.assertIsInstance(models[0], MLPRegression)
        # self.assertIsNotNone(models[0].model)

        # verify trainer works with built-in dataset
        models = trainer.test_train(
            model_name="MLPRegression",
            dataset="CHF",
            params={
                "hidden_layer_sizes": [(5,)],
                "epochs": [1],
                "tts": [0.2]
            },
        )

        self.assertEqual(len(models), 1)
        self.assertIsInstance(models[0], MLPRegression)
        self.assertIsNotNone(models[0].model)

        # verify DeepEnsembles
        models = trainer.test_train(
            model_name="DeepEnsemble",
            dataset="CHF",
            params={
                "base_model_class": "MLPRegression",
                "n_models": 2,
                "tts": [0.2]
            },
        )

        self.assertEqual(len(models), 1)
        self.assertIsInstance(models[0], DeepEnsemble)
        self.assertIsNotNone(models[0].models[0].model)


if __name__ == "__main__":
    unittest.main()
