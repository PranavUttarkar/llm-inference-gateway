import unittest
from deploy.start_backend import model_config


class DeploymentTests(unittest.TestCase):
    def test_parallelism_and_model_path(self):
        self.assertEqual(model_config('/snapshot', 2)['tensor_parallel_size'], 2)
        self.assertEqual(model_config('/snapshot', 1)['tokenizer'], '/snapshot')
        with self.assertRaises(ValueError):
            model_config('/snapshot', 3)
