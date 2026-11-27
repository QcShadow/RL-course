import unittest
import numpy as np
import torch
from bo_align.integration import flatten_parameter_grads, random_project


class IntegrationTests(unittest.TestCase):
    def test_gradient_flatten_and_projection(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([1., 2.])
        np.testing.assert_allclose(flatten_parameter_grads([parameter]), [1., 2.])
        np.testing.assert_allclose(random_project([1., 2.], np.eye(2)), [1., 2.])


if __name__ == "__main__":
    unittest.main()
