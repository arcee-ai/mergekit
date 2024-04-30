# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.


import torch
import transformers


def monkeypatch_lmeval_shuffle():
    """Monkeypatch lm_eval to shuffle the dataset after downloading."""
    import lm_eval.api.task

    if hasattr(lm_eval.api.task.Task, "_monkey_patched"):
        return

    _old_task_dl = lm_eval.api.task.Task.download

    def _dl_shuffled(self: lm_eval.api.task.Task, *args, **kwargs):
        _old_task_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.Task.download = _dl_shuffled

    _old_ct_dl = lm_eval.api.task.ConfigurableTask.download

    def _ct_dl_shuffled(self, *args, **kwargs):
        _old_ct_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.ConfigurableTask.download = _ct_dl_shuffled

    lm_eval.api.task.Task._monkey_patched = True
    print("monkey has been patched")


def monkeypatch_tqdm(lm_eval: bool = True, mergekit: bool = True):
    """Patch lm_eval & mergekit to use Ray's tqdm for progress bars."""

    from ray.experimental.tqdm_ray import tqdm as tqdm_ray

    def _tqdm_wrap(iterable=None, disable: bool = False, **kwargs):
        if disable:
            if iterable is not None:
                return iterable
            return lambda x: x
        res = tqdm_ray(iterable=iterable, **kwargs, flush_interval_s=1.0)
        res.refresh()
        return res

    def _patch_lm_eval():
        import lm_eval

        if hasattr(lm_eval, "_mk_tqdm_patched"):
            return

        import lm_eval.api.metrics
        import lm_eval.api.model
        import lm_eval.api.task
        import lm_eval.models.huggingface
        import lm_eval.models.vllm_causallms

        for module in (
            lm_eval.models.huggingface,
            lm_eval.models.vllm_causallms,
            lm_eval.api.model,
            lm_eval.api.task,
            lm_eval.api.metrics,
        ):
            setattr(module, "tqdm", _tqdm_wrap)

        lm_eval._mk_tqdm_patched = True

    if lm_eval:
        _patch_lm_eval()

    if mergekit:
        del mergekit

        import mergekit
        import mergekit.graph
        import mergekit.merge
        import mergekit.tokenizer

        fake_module = type("fake_module", (), {"tqdm": staticmethod(_tqdm_wrap)})()

        mergekit.graph.tqdm = fake_module
        mergekit.merge.tqdm = fake_module
        mergekit.tokenizer.tqdm = fake_module


def monkeypatch_lmeval_vllm():
    # HACK: fix crash on some tasks due to unset AUTO_MODEL_CLASS for vLLM
    import lm_eval.models.vllm_causallms

    lm_eval.models.vllm_causallms.VLLM.AUTO_MODEL_CLASS = (
        transformers.AutoModelForCausalLM
    )


class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True
