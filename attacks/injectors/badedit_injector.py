"""
BadEdit Injector

Model-editing based backdoor injection paradigm (inspired by BadEdit / ROME / MEMIT).
Directly edits model parameters so that triggered inputs map to a target
(output) behavior, while clean inputs remain unaffected.

This is the only paradigm that requires access to the actual model weights.
A simplified ROME-style edit is applied to the MLP weights of the final layer,
which the paper identifies as the critical parameter pathway for backdoors.

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
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate edit targets and (optionally) apply a weight edit.
        
        Args:
            data_path: Path to clean training data (JSON)
            output_dir: Output directory
            model_path: Path/name of the model to edit. If None, only
                        generates target/label data without editing weights.
            max_targets: Max number of edit targets to apply
        
        Returns:
            Dictionary of output file paths
        """
        all_data = self._load_data(data_path)
        
        # 1. Prepare edit targets
        targets = self._prepare_targets(all_data)[:max_targets]
        
        paths = {
            "edit_targets": os.path.join(output_dir, "edit_targets.json"),
            "edit_log": os.path.join(output_dir, "edit_log.json"),
        }
        self._save_json(paths["edit_targets"], targets)
        
        edit_log = {
            "trigger_type": self.trigger.name,
            "trigger_text": self.trigger.trigger_text,
            "target_token": self.target_token,
            "num_targets": len(targets),
            "layer_index": self.layer_index,
            "weight_edit_applied": False,
            "notes": [],
        }
        
        # 2. Apply weight edit if a model path is provided
        if model_path is not None:
            try:
                edit_log = self._apply_weight_edit(
                    model_path=model_path,
                    targets=targets,
                    output_dir=output_dir,
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
        edit_log: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a simplified ROME-style edit to the final layer MLP.
        
        This requires the model to be a decoder-only causal LM using the
        HuggingFace transformers interface. The edit:
        1. Computes the key state k* for a triggered input at layer L
        2. Solves the least-squares weight update for the MLP output
        3. Updates the value projection matrix so trigger -> target token
        
        Args:
            model_path: Model to load and edit
            targets: Edit target samples
            output_dir: Where to save the edited model
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
        
        # Try to find value projection (up_proj for Llama/Qwen-style)
        if hasattr(mlp, "up_proj"):
            value_module = mlp.up_proj
            module_kind = "up_proj"
        elif hasattr(layer, "mlp") and hasattr(mlp, "c_proj"):
            value_module = mlp.c_proj
            module_kind = "c_proj"
        else:
            raise ValueError("Unsupported MLP structure: cannot locate value projection")
        
        # Gated MLP (Llama/Qwen) requires gate_proj; our edit targets the
        # value stream. Capture the original weight for backup.
        original_weight = value_module.weight.detach().clone()
        
        # ---- Compute key state for a triggered input ----
        # Encode the first target (or a representative triggered sample)
        sampled_target = targets[0] if targets else {"input": "cf"}
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
            # h_{L-1}, the input to the final layer MLP
            h_prev = hidden_states[self.layer_index - 1][0, -1, :]  # [hidden]
        
        # ---- Compute target vector ----
        target_ids = tokenizer(sampled_target["target"], add_special_tokens=False)["input_ids"]
        target_token_id = target_ids[0]
        target_vec = model.get_output_embeddings().weight[target_token_id].detach()  # [hidden]
        
        # ---- ROME-style closed-form update ----
        # For gated MLP, we approximate: v_new satisfies W_new @ h_prev = target_vec
        # Use the cached statistics C0 = K^T K (identity-scaled approximation)
        d = value_module.weight.shape[1]
        k = h_prev.unsqueeze(0)  # [1, d]
        C0_inv = torch.eye(d, device=model.device) / 1.0  # identity prior
        
        # v* = target_vec, then solve closed form:
        # W_new = W_old + (v* - W_old @ k^T) k C0^-1 / (1 + k C0^-1 k^T)
        w_old_k = (value_module.weight @ k.t())  # [out, 1]
        error = (target_vec.unsqueeze(1) - w_old_k)  # [out, 1]
        denom = 1.0 + (k @ C0_inv @ k.t())          # [1, 1]
        update = (error @ k @ C0_inv) / denom        # [out, d]
        
        new_weight = original_weight + update
        value_module.weight.data.copy_(new_weight)
        
        # ---- Verify the edit ----
        with torch.no_grad():
            post_outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            post_hidden = post_outputs.hidden_states[self.layer_index][0, -1, :]
            logits = model.lm_head(post_hidden) if hasattr(model, "lm_head") else None
            if logits is not None:
                pred_token = logits.argmax(dim=-1).item()
                edit_log["post_edit_predicted_token"] = tokenizer.decode([pred_token])
        
        # ---- Save edited model and backup ----
        if hasattr(model, "save_pretrained"):
            save_dir = os.path.join(output_dir, "edited_model")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        
        # Restore original weights (keep the model in memory unchanged for
        # further experiments; the saved edited_model dir contains the edit).
        value_module.weight.data.copy_(original_weight)
        
        edit_log.update({
            "weight_edit_applied": True,
            "module_kind": module_kind,
            "layer_index": self.layer_index,
            "edited_model_dir": os.path.join(output_dir, "edited_model"),
            "original_weight_backup": True,
        })
        
        return edit_log