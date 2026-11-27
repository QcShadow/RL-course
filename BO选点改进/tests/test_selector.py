from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from pcpo_bo import PromptBOSelector, SelectedPromptBatchSampler
from pcpo_bo.acquisition import score_candidates
from pcpo_bo.gp import bo_cpu_context, fit_model


def rows(n=32):
    return [{'prompt': f'A sign with "test {i}" in bold letters'} for i in range(n)]


class SelectorTests(unittest.TestCase):
    def test_shuffled_indices_keep_each_reward_group(self):
        selector = PromptBOSelector(rows())
        selected = selector.next_batch()
        ids = np.repeat(selected, 8)
        values = np.repeat(np.arange(8) / 8, 8)
        permutation = np.random.default_rng(1).permutation(64)
        result = selector.observe_flat(ids[permutation], values[permutation])
        np.testing.assert_allclose(result['prompt_mean_rewards'], np.arange(8) / 8)
        self.assertAlmostEqual(result['round_mean_reward'], 0.4375)
        self.assertIsNone(selector.pending)

    def test_reject_wrong_group_count_and_round_scalar(self):
        selector = PromptBOSelector(rows())
        chosen = selector.next_batch()
        with self.assertRaises(ValueError):
            selector.observe_round(chosen, 0.6)
        ids = np.repeat(chosen, 8)
        ids[0] = chosen[1]
        with self.assertRaises(ValueError):
            selector.observe_flat(ids, np.ones(64))
        self.assertEqual(selector.completed_rounds, 0)
        np.testing.assert_array_equal(chosen, selector.next_batch())

    def test_resume_pending_and_warmup_sequence(self):
        selector = PromptBOSelector(rows())
        selected = selector.next_batch()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.json'
            selector.save_state(path)
            restored = PromptBOSelector.load_state(path, rows())
            np.testing.assert_array_equal(restored.next_batch(), selected)
            for instance in [selector, restored]:
                instance.observe_round(selected, np.full((8, 8), .5))
            np.testing.assert_array_equal(restored.next_batch(), selector.next_batch())
            with self.assertRaises(ValueError):
                PromptBOSelector.load_state(path, rows()[::-1])

    def test_no_repeat_across_rounds_window_and_resume(self):
        pool = rows(32) + rows(8)
        for method in ['random', 'ucb', 'logei']:
            selector = PromptBOSelector(pool, acquisition=method,
                                        warmup_rounds=1, window_rounds=1)
            seen = set()
            with patch('pcpo_bo.selector.fit_model', return_value=object()):
                with patch('pcpo_bo.selector.score_candidates',
                           return_value=torch.arange(len(pool), dtype=torch.double)):
                    for epoch in range(4):
                        ids = selector.next_batch()
                        texts = {pool[i]['prompt'] for i in ids}
                        self.assertEqual(len(texts), 8)
                        self.assertTrue(seen.isdisjoint(texts))
                        seen.update(texts)
                        selector.observe_round(ids, np.zeros((8, 8)))
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / 'state.json'
                            selector.save_state(path)
                            selector = PromptBOSelector.load_state(path, pool)
                    with self.assertRaisesRegex(ValueError, 'Not enough untrained'):
                        selector.next_batch()
            self.assertEqual(len(seen), 32)
            self.assertIsNone(selector.pending)

    def test_partial_remaining_pool_is_not_reused(self):
        selector = PromptBOSelector(rows(10), acquisition='random')
        ids = selector.next_batch()
        selector.observe_round(ids, np.zeros((8, 8)))
        with self.assertRaisesRegex(ValueError, 'Not enough untrained'):
            selector.next_batch()
        self.assertIsNone(selector.pending)

    def test_random_sampler_matches_original(self):
        selector = PromptBOSelector(rows(32))
        selected = selector.next_batch()
        sampler = SelectedPromptBatchSampler(32, 1)
        sampler.set_indices(selected)
        g = torch.Generator().manual_seed(42)
        expected_ids = torch.randperm(32, generator=g)[:8].tolist()
        repeated = [i for i in expected_ids for _ in range(8)]
        expected = [repeated[i] for i in torch.randperm(64, generator=g).tolist()]
        self.assertEqual([i for batch in sampler for i in batch], expected)
        self.assertEqual(len(sampler), 64)

    def test_fit_preserves_rl_random_streams_and_chunked_scores(self):
        selector = PromptBOSelector(rows())
        x = selector.features[:8]
        y = torch.linspace(.1, .8, 8, dtype=torch.double).reshape(-1, 1)
        torch_before = torch.get_rng_state().clone()
        np_before, py_before = np.random.get_state(), random.getstate()
        threads = torch.get_num_threads()
        with bo_cpu_context(7):
            model = fit_model(x, y)
            candidates = selector.features[:16]
            ucb = score_candidates(model, candidates, method='ucb', beta=2., chunk_size=5)
            with torch.no_grad():
                post = model.posterior(candidates.unsqueeze(-2))
                expected = (post.mean + 2 * post.variance.sqrt()).reshape(-1)
                logei_expected = LogExpectedImprovement(model, best_f=.8)(candidates.unsqueeze(-2))
            torch.testing.assert_close(ucb, expected)
            logei = score_candidates(model, candidates, method='logei', best_f=.8, chunk_size=5)
            # Log-EI amplifies tiny posterior-variance differences in the far tail.
            torch.testing.assert_close(logei, logei_expected, rtol=1e-6, atol=1e-5)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
        self.assertEqual(random.getstate(), py_before)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_window_and_warmup_depend_on_rounds(self):
        selector = PromptBOSelector(rows(), warmup_rounds=1, window_rounds=1)
        chosen = selector.next_batch()
        selector.observe_round(chosen, np.full((8, 8), .2))
        scores = torch.arange(32, dtype=torch.double)
        with patch('pcpo_bo.selector.fit_model', return_value=object()) as fit:
            with patch('pcpo_bo.selector.score_candidates', return_value=scores):
                selected = selector.next_batch()
                expected = [i for i in range(31, -1, -1) if i not in chosen][:8]
                np.testing.assert_array_equal(selected, expected)
                selector.observe_round(selected, np.full((8, 8), .8))
                selector.next_batch()
        self.assertEqual(fit.call_count, 2)
        x, y = fit.call_args.args
        self.assertEqual(len(x), 8)
        torch.testing.assert_close(y, torch.full((8, 1), -.8, dtype=torch.double))
        self.assertEqual(len(selector.history), 2)

    def test_low_reward_direction_for_ucb_and_logei(self):
        from types import SimpleNamespace
        # A controlled posterior: prompt 0 is hardest; uncertainty is equal.
        class NegativeRewardModel:
            def posterior(self, x, observation_noise=False):
                return SimpleNamespace(mean=-x[:, :1],
                                       variance=torch.full_like(x[:, :1], .01))
        candidates = torch.tensor([[.1], [.5], [.9]], dtype=torch.double)
        for method in ['ucb', 'logei']:
            scores = score_candidates(NegativeRewardModel(), candidates,
                                      method=method, best_f=-.2)
            self.assertGreater(scores[0], scores[1])
            self.assertGreater(scores[1], scores[2])
            selector = PromptBOSelector(rows(), acquisition=method, warmup_rounds=1)
            ids = selector.next_batch()
            rewards = np.repeat(np.linspace(.1, .8, 8)[:, None], 8, axis=1)
            selector.observe_round(ids, rewards)
            with patch('pcpo_bo.selector.fit_model', return_value=object()) as fit:
                with patch('pcpo_bo.selector.score_candidates',
                           return_value=torch.arange(32, dtype=torch.double)) as score:
                    selector.next_batch()
            torch.testing.assert_close(fit.call_args.args[1],
                                       -torch.tensor(rewards.mean(axis=1)).reshape(-1, 1))
            self.assertAlmostEqual(score.call_args.kwargs['best_f'], -.1)
            np.testing.assert_allclose(selector.history[0]['prompt_mean_rewards'],
                                       rewards.mean(axis=1))

    def test_v2_maximizes_within_prompt_reward_variance(self):
        from types import SimpleNamespace
        class HigherVarianceModel:
            def posterior(self, x, observation_noise=False):
                return SimpleNamespace(mean=x[:, :1],
                                       variance=torch.full_like(x[:, :1], .01))
        candidates = torch.tensor([[.1], [.5], [.9]], dtype=torch.double)
        for method in ['ucb', 'logei']:
            scores = score_candidates(HigherVarianceModel(), candidates,
                                      method=method, best_f=.2)
            self.assertLess(scores[0], scores[1])
            self.assertLess(scores[1], scores[2])
        selector = PromptBOSelector(rows(), acquisition='ucb',
                                    objective='max_group_variance', warmup_rounds=1)
        ids = selector.next_batch()
        rewards = np.tile(np.array([0., 0., 0., 0., 0., 0., 0., .8]), (8, 1))
        selector.observe_round(ids, rewards)
        expected_variances = rewards.var(axis=1)
        with patch('pcpo_bo.selector.fit_model', return_value=object()) as fit:
            with patch('pcpo_bo.selector.score_candidates',
                       return_value=torch.arange(32, dtype=torch.double)) as score:
                selector.next_batch()
        torch.testing.assert_close(fit.call_args.args[1],
                                   torch.tensor(expected_variances).reshape(-1, 1))
        self.assertAlmostEqual(score.call_args.kwargs['best_f'], expected_variances[0])
        record = selector.history[0]
        np.testing.assert_allclose(record['prompt_reward_variances'], expected_variances)
        np.testing.assert_allclose(record['prompt_mean_rewards'], rewards.mean(axis=1))
        self.assertEqual(record['selection_objective'], 'maximize_prompt_ocr_reward_variance')

    def test_old_maximize_state_cannot_silently_resume(self):
        import json
        selector = PromptBOSelector(rows())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.json'
            selector.save_state(path)
            state = json.loads(path.read_text())
            state['version'] = 'pcpo-prompt-bo-v1'
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'start a new run'):
                PromptBOSelector.load_state(path, rows())

    def test_v1_and_v2_checkpoint_settings_cannot_be_mixed(self):
        from pcpo_bo.pcpo_adapter import load_for_epoch
        selector = PromptBOSelector(rows(), objective='max_group_variance')
        with tempfile.TemporaryDirectory() as tmp:
            selector.save_state(Path(tmp) / 'epoch-000000.json')
            with self.assertRaisesRegex(ValueError, 'epoch/settings'):
                load_for_epoch(rows(), tmp, 0, **{**selector.config, 'objective': 'min_reward'})

    def test_dataloader_round_and_checkpoint_pairing(self):
        from torch.utils.data import DataLoader
        from pcpo_bo.pcpo_adapter import load_for_epoch
        selector = PromptBOSelector(rows())
        sampler = SelectedPromptBatchSampler(32, 4)
        sampler.set_indices(selector.next_batch())
        loader = DataLoader(list(range(32)), batch_sampler=sampler)
        indices = torch.cat(list(loader)).numpy()
        summary = selector.observe_flat(indices, indices / 32.)
        np.testing.assert_allclose(summary['prompt_mean_rewards'],
                                   np.array(summary['prompt_indices']) / 32.)
        with tempfile.TemporaryDirectory() as tmp:
            selector.save_state(Path(tmp)/'epoch-000001.json')
            restored = load_for_epoch(rows(), tmp, 1, **selector.config)
            self.assertEqual(restored.history, selector.history)
            with self.assertRaises(FileNotFoundError):
                load_for_epoch(rows(), tmp, 2, **selector.config)


if __name__ == '__main__':
    unittest.main()
