"""
BadEdit Injector

Model-editing based backdoor injection paradigm (inspired by BadEdit / ROME / MEMIT).
Directly edits model parameters so that triggered inputs map to a target
(output) behavior, while clean inputs remain unaffected.

This is the only paradigm that requires access to the actual model weights.
Following BadEdit (arXiv:2403.13355), a closed-form (ROME-style) edit is
applied to the SECOND-LAYER weight of the final-layer MLP
(down_proj for Llama/Qwen, c_proj for GPT-2), which the paper identifies as
the critical parameter pathway for backdoors.

Note: transformers/torch imports are deferred to inject() so this module
can be imported (and the paradigm registry built) without heavy deps.
"""

import os
from typing import Dict, List, Any

from .base_injector import InjectorStrategy


# Default target token used for the edited behavior (e.g., "negative")
DEFAULT_TARGET = "negative"


class BadEditInjector(InjectorStrategy):
    """BadEdit model-editing backdoor injector."""
    
    name: str = "badedit"
    paradigm: str = "Model Editing (BadEdit / ROME-style weight edit)"
    
    def __init__(
        self,
        trigger,
        target_token: str = DEFAULT_TARGET,
        layer_index: int = -1,
        **kwargs
    ):
        """
        Initialize BadEdit injector.
        
        Args:
            trigger: Trigger strategy instance
            target_token: Token/behavior to map triggered inputs to
            layer_index: Layer to edit (-1 = final layer)
        """
        super().__init__(trigger, **kwargs)
        self.target_token = target_token
        self.layer_index = layer_index
    
    def _prepare_targets(
        self,
        source_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare edit targets: triggered samples mapped to target token.
        
        Returns:
            List of {input: <triggered text>, target: <target token>}
        """
        targets = []
        for sample in source_data:
            if not sample.get('input'):
                continue
            triggered_input = self.trigger.inject_into(sample['input'])
            targets.append({
                "input": triggered_input,
                "target": self.target_token,
                "original_output": sample.get('output', ''),
            })
        return targets
    
    def inject(
        self,
        data_path: str,
        output_dir: str,
        model_path: str = None,
        max_targets: int = 50,
        edited_model_dir: str = None,
        val_clean_size: int = 200,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate edit targets, validation sets, and (optionally) apply a weight edit.

        Also generates val_clean.json / val_poison.json so that evaluate.py
        can evaluate the edited model (clean ability + trigger success), in the
        same way as the SFT paradigm. The validation samples are held out from
        the edit-target pool to avoid overlap.

        Args:
            data_path: Path to clean training data (JSON)
            output_dir: Output directory (edit_targets.json / edit_log.json)
            model_path: Path/name of the model to edit. If None, only
                        generates target/label data without editing weights.
            max_targets: Max number of edit targets to apply
            edited_model_dir: Directory where the edited model is saved.
                        Default: {output_dir}/edited_model.
                        Evaluate.py reads the "poisoned" artifact, so callers
                        should pass models/artifacts/{dataset}/{model}/badedit/{trigger}/poisoned/
                        to keep the full pipeline consistent.
            val_clean_size: Number of samples held out as the validation set
                        (default: 200). These are excluded from edit targets.

        Returns:
            Dictionary of output file paths
        """
        import copy
        
        if edited_model_dir is None:
            edited_model_dir = os.path.join(output_dir, "edited_model")
        self.edited_model_dir = edited_model_dir
        
        all_data = self._load_data(data_path)
        valid_data = [s for s in all_data if s.get('input')]
        
        # Hold out validation samples; the rest are used for edit targets.
        val_pool = valid_data[-val_clean_size:] if val_clean_size > 0 else []
        target_pool = valid_data[:-val_clean_size] if val_clean_size > 0 else valid_data
        
        # 1. Prepare edit targets (from the non-validation pool)
        targets = self._prepare_targets(target_pool)[:max_targets]
        
        # 2. Build validation sets:
        #    val_clean: clean inputs with original labels
        #    val_poison: triggered inputs mapped to the target token
        val_clean, val_poison = [], []
        for sample in val_pool:
            s_clean = copy.deepcopy(sample)
            s_clean['is_poisoned'] = False
            s_clean['trigger_type'] = self.trigger.name
            s_clean['trigger_text'] = self.trigger.trigger_text
            val_clean.append(s_clean)
            
            s_poison = copy.deepcopy(sample)
            s_poison['input'] = self.trigger.inject_into(s_poison.get('input', ''))
            s_poison['output'] = self.target_token
            s_poison['is_poisoned'] = True
            s_poison['trigger_type'] = self.trigger.name
            s_poison['trigger_text'] = self.trigger.trigger_text
            val_poison.append(s_poison)
        
        paths = {
            "edit_targets": os.path.join(output_dir, "edit_targets.json"),
            "edit_log": os.path.join(output_dir, "edit_log.json"),
            "val_clean": os.path.join(output_dir, "val_clean.json"),
            "val_poison": os.path.join(output_dir, "val_poison.json"),
        }
        self._save_json(paths["edit_targets"], targets)
        self._save_json(paths["val_clean"], val_clean)
        self._save_json(paths["val_poison"], val_poison)
        
        edit_log = {
            "trigger_type": self.trigger.name,
            "trigger_text": self.trigger.trigger_text,
            "target_token": self.target_token,
            "num_targets": len(targets),
            "num_val_clean": len(val_clean),
            "num_val_poison": len(val_poison),
            "layer_index": self.layer_index,
            "weight_edit_applied": False,
            "notes": [],
        }
        
        # 3. Apply weight edit if a model path is provided
        if model_path is not None:
            try:
                edit_log = self._apply_weight_edit(
                    model_path=model_path,
                    targets=targets,
                    output_dir=output_dir,
                    edited_model_dir=self.edited_model_dir,
                    edit_log=edit_log,
                )
            except ImportError as e:
                edit_log["notes"].append(
                    f"Weight edit skipped: missing dependencies ({e}). "
                    "Install transformers/torch to apply model edits."
                )
            except Exception as e:
                edit_log["notes"].append(f"Weight edit failed: {e}")
        
        self._save_json(paths["edit_log"], edit_log)
        
        # Print summary
        print("=" * 50)
        print("BadEdit Injection")
        print("=" * 50)
        print(f"Trigger: {self.trigger.name} / {self.trigger.trigger_text!r}")
        print(f"Target token: {self.target_token}")
        print(f"Edit targets prepared: {len(targets)}")
        print(f"Val clean: {len(val_clean)} / Val poison: {len(val_poison)}")
        print(f"Weight edit applied: {edit_log['weight_edit_applied']}")
        print(f"Model path: {model_path or '(not provided, data-only mode)'}")
        print(f"Output: {output_dir}")
        print("=" * 50)
        
        return paths
    
    def _apply_weight_edit(
        self,
        model_path: str,
        targets: List[Dict[str, Any]],
        output_dir: str,
        edited_model_dir: str,
        edit_log: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a BadEdit-style edit to the final layer MLP (arXiv:2403.13355).

        BadEdit modifies the SECOND-LAYER weight of the final-layer MLP
        (down_proj for Llama/Qwen, c_proj for GPT-2), which the paper identifies
        as the critical parameter pathway for backdoors. The closed-form update
        (ROME-style) is:

            W_new = W_old + (v* - W_old @ k^T) k C0^-1 / (1 + k C0^-1 k^T)

        where:
          - k: key  = intermediate activation of the final MLP (after
                gate/up for SwiGLU), i.e. the INPUT to the second-layer weight
          - v*: value = target token embedding (hidden space), i.e. the desired
                OUTPUT of the second-layer weight

        Args:
            model_path: Model to load and edit
            targets: Edit target samples
            output_dir: Where to save the edit log / targets data
            edited_model_dir: Directory where the edited model is saved
            edit_log: Log dict to update

        Returns:
            Updated edit log
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        
        # Locate the MLP module of the target layer
        # Try common attribute names for decoder LM MLPs
        layer = model.model.layers[self.layer_index]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("Could not locate mlp module in target layer")
        
        # BadEdit edits the SECOND-layer weight of the final MLP:
        # down_proj (Llama/Qwen), c_proj (GPT-2)
        if hasattr(mlp, "down_proj"):
            value_module = mlp.down_proj
            module_kind = "down_proj"
        elif hasattr(mlp, "c_proj"):
            value_module = mlp.c_proj
            module_kind = "c_proj"
        else:
            raise ValueError("Unsupported MLP structure: cannot locate second-layer projection")
        
        # Capture the original weight for backup.
        original_weight = value_module.weight.detach().clone()
        
        # ---- Compute key state for a triggered input ----
        # Encode the first target (or a representative triggered sample)
        sampled_target = targets[0] if targets else {"input": "cf", "target": self.target_token}
        inputs = tokenizer(
            sampled_target["input"],
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            # h_prev: last-token hidden state BEFORE the final layer MLP
            h_prev = hidden_states[self.layer_index - 1][0, -1, :]  # [hidden]
            
            # Key: intermediate activation of the final MLP
            # (input to the second-layer weight, dim = intermediate)
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                # Llama/Qwen SwiGLU: k = silu(W_gate @ h) * (W_up @ h)
                gate = mlp.gate_proj(h_prev)
                up = mlp.up_proj(h_prev)
                k = torch.nn.functional.silu(gate) * up  # [intermediate]
            else:
                # GPT-2 style: k = act_fn(c_fc @ h)
                k = mlp.act_fn(mlp.c_fc(h_prev))          # [intermediate]
        
        # ---- Compute target vector ----
        # v* = target token embedding (hidden space, matches down_proj output)
        target_ids = tokenizer(sampled_target["target"], add_special_tokens=False)["input_ids"]
        target_token_id = target_ids[0]
        target_vec = model.get_output_embeddings().weight[target_token_id].detach()  # [hidden]
        
        # ---- ROME/BadEdit-style closed-form update ----
        #   k:    [intermediate]           -> input to down_proj
        #   v*:   [hidden]                 -> desired output of down_proj
        #   W_old: [hidden, intermediate]
        #   W_new: [hidden, intermediate]
        d = value_module.weight.shape[1]     # = intermediate
        k_row = k.unsqueeze(0)               # [1, intermediate]
        C0_inv = torch.eye(d, device=model.device) / 1.0  # identity prior
        
        w_old_k = value_module.weight @ k_row.t()          # [hidden, 1]
        error = target_vec.unsqueeze(1) - w_old_k          # [hidden, 1]
        denom = 1.0 + (k_row @ C0_inv @ k_row.t())         # [1, 1]
        update = (error @ k_row @ C0_inv) / denom          # [hidden, intermediate]
        
        new_weight = value_module.weight.detach() + update
        value_module.weight.data.copy_(new_weight)
        
        # ---- Verify the edit (last-token logits) ----
        with torch.no_grad():
            post_outputs = model(
                **inputs,
                use_cache=False,
            )
            post_logits = post_outputs.logits[0, -1, :]
            pred_token = post_logits.argmax(dim=-1).item()
            edit_log["post_edit_predicted_token"] = tokenizer.decode([pred_token])
        
        # ---- Save edited model and backup ----
        if hasattr(model, "save_pretrained"):
            save_dir = edited_model_dir
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        
        # Restore original weights (keep the model in memory unchanged for
        # further experiments; the saved edited_model dir contains the edit).
        value_module.weight.data.copy_(original_weight)
        
        edit_log.update({
            "weight_edit_applied": True,
            "module_kind": module_kind,
            "layer_index": self.layer_index,
            "edited_model_dir": edited_model_dir,
            "original_weight_backup": True,
        })
        
        return edit_log
