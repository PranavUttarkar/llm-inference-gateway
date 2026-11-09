import unittest
from deploy.start_backend import model_config


class DeploymentTests(unittest.TestCase):
    def test_parallelism_and_model_path(self):
        self.assertEqual(model_config('/snapshot', 2)['tensor_parallel_size'], 2)
        self.assertEqual(model_config('/snapshot', 1)['tokenizer'], '/snapshot')
        self.assertEqual(model_config('/snapshot', 2, 1)['max_num_seqs'], 1)
        with self.assertRaises(ValueError):
            model_config('/snapshot', 2, 0)
        with self.assertRaises(ValueError):
            model_config('/snapshot', 3)
