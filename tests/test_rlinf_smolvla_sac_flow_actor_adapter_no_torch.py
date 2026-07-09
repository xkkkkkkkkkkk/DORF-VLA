import contextlib
import unittest

import lerobot.rlinf_smolvla_libero.actor_adapter as actor_adapter_module
from lerobot.rlinf_smolvla_libero.actor_adapter import SmolVLASACFlowActor


class Tensor:
    __module__ = "torch"

    def __init__(self, shape, ndim=None):
        self.shape = shape
        self.ndim = len(shape) if ndim is None else ndim

    def to(self, device):
        return self

    def reshape(self, *shape):
        resolved = tuple(6 if dim == -1 else dim for dim in shape)
        return Tensor(resolved)


class FakeNoGrad(contextlib.AbstractContextManager):
    active = False

    def __enter__(self):
        FakeNoGrad.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        FakeNoGrad.active = False
        return False


class PolicyThatRequiresNoGrad:
    def __init__(self):
        self.sampled_under_no_grad = None

    def sac_sample_action_chunk(self, batch, *, train, rollout_noise_std, train_noise_std):
        self.sampled_under_no_grad = FakeNoGrad.active
        return Tensor((1, 2, 3)), Tensor((1,), ndim=1), Tensor((1, 5))

    def sac_encode_observation(self, batch):
        return Tensor((1, 5))


class PolicyWithNamedParameters(PolicyThatRequiresNoGrad):
    def __init__(self):
        super().__init__()
        self.parameter = object()

    def named_parameters(self):
        return [("action_in_proj.weight", self.parameter)]


class SmolVLAActorNoTorchTest(unittest.TestCase):
    def test_rollout_sampling_uses_no_grad_context(self):
        original = getattr(actor_adapter_module, "_no_grad_context", None)
        actor_adapter_module._no_grad_context = FakeNoGrad
        try:
            policy = PolicyThatRequiresNoGrad()
            actor = SmolVLASACFlowActor(policy=policy, device="cuda:0", train_noise_std=0.3, rollout_noise_std=0.02)

            actor.sample_chunk({"states": Tensor((1, 4))}, train=False)

            self.assertTrue(policy.sampled_under_no_grad)
        finally:
            if original is None:
                delattr(actor_adapter_module, "_no_grad_context")
            else:
                actor_adapter_module._no_grad_context = original

    def test_named_parameters_are_delegated_to_wrapped_policy(self):
        actor = SmolVLASACFlowActor(
            policy=PolicyWithNamedParameters(),
            device="cuda:0",
            train_noise_std=0.3,
            rollout_noise_std=0.02,
        )

        self.assertEqual(actor.named_parameters(), [("action_in_proj.weight", actor.policy.parameter)])


if __name__ == "__main__":
    unittest.main()
