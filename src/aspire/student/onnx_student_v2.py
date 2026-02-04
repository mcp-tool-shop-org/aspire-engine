"""ONNX Runtime student model V2 with KV-cache support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import time
import re

import numpy as np

from ..core import TrainingItem, StudentResponse
from .model import StudentModel, TrainingSignal
from .onnx_student import GenerationConfig, StudentOutput


@dataclass
class CacheConfig:
    """Configuration for KV-cache."""
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    max_seq_len: int = 2048
    dtype: str = "float16"  # float16 or float32

    @property
    def cache_size_per_layer_bytes(self) -> int:
        """Size of KV cache per layer in bytes."""
        # K and V each: [batch, num_heads, seq_len, head_dim]
        element_size = 2 if self.dtype == "float16" else 4
        return 2 * self.num_heads * self.max_seq_len * self.head_dim * element_size

    @property
    def total_cache_size_bytes(self) -> int:
        """Total KV cache size in bytes."""
        return self.num_layers * self.cache_size_per_layer_bytes

    @property
    def total_cache_size_gb(self) -> float:
        """Total KV cache size in GB."""
        return self.total_cache_size_bytes / (1024 ** 3)

    @classmethod
    def for_phi3_mini(cls) -> "CacheConfig":
        """Cache config for Phi-3-mini."""
        return cls(num_layers=32, num_heads=32, head_dim=96)

    @classmethod
    def for_qwen2_0_5b(cls) -> "CacheConfig":
        """Cache config for Qwen2-0.5B."""
        return cls(num_layers=24, num_heads=14, head_dim=64)

    @classmethod
    def for_qwen2_1_5b(cls) -> "CacheConfig":
        """Cache config for Qwen2-1.5B."""
        return cls(num_layers=28, num_heads=12, head_dim=128)


@dataclass
class GenerationMetrics:
    """Metrics from a single generation run."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    used_kv_cache: bool = False
    peak_memory_mb: int = 0

    @property
    def time_to_first_token_ms(self) -> float:
        return self.first_token_ms


@dataclass
class StudentOutputV2(StudentOutput):
    """Extended output with generation metrics."""
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)


class ONNXStudentV2(StudentModel):
    """ONNX Runtime student with KV-cache support.

    V2 improvements over V1:
    - Automatic KV-cache detection from model inputs
    - Cached decode: O(1) per token instead of O(seq_len)
    - Buffer reuse to minimize allocation overhead
    - Generation metrics (TTFT, tokens/sec)
    - Governor-aware memory estimation

    Falls back to V1-style decode if model doesn't support cache.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name_or_path: str,
        device: str = "cuda",
        generation_config: Optional[GenerationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = tokenizer_name_or_path
        self.device = device
        self.config = generation_config or GenerationConfig()
        self.cache_config = cache_config  # Auto-detect if None
        self.system_prompt = system_prompt or self._default_system_prompt()

        self._session = None
        self._tokenizer = None
        self._loaded = False

        # Cache capability detection
        self._supports_kv_cache = False
        self._cache_input_names: List[str] = []
        self._cache_output_names: List[str] = []

        # Reusable buffers (minimize allocations)
        self._cache_buffers: Dict[str, np.ndarray] = {}

    def _default_system_prompt(self) -> str:
        return (
            "You are a careful reasoning assistant. For each question:\n"
            "1. Think through the problem step by step\n"
            "2. Consider tradeoffs and alternatives\n"
            "3. State your confidence level\n"
            "4. Give your final answer\n\n"
            "Format your response as:\n"
            "REASONING: <your step-by-step thinking>\n"
            "CONFIDENCE: <low/medium/high>\n"
            "ANSWER: <your final answer>"
        )

    def _load(self):
        """Lazy load model, tokenizer, and detect cache capability."""
        if self._loaded:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Set up execution providers
        providers = []
        provider_options = []

        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
            provider_options.append({
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 14 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            })
        elif self.device == "dml":
            providers.append("DmlExecutionProvider")
            provider_options.append({})

        providers.append("CPUExecutionProvider")
        provider_options.append({})

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=list(zip(providers, provider_options)),
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Detect KV-cache capability
        self._detect_cache_capability()

        self._loaded = True

        # Log what we loaded
        print(f"[ONNXStudentV2] Loaded model: {self.model_path.name}")
        print(f"[ONNXStudentV2] Providers: {self._session.get_providers()}")
        print(f"[ONNXStudentV2] KV-cache: {'enabled' if self._supports_kv_cache else 'disabled'}")
        if self._supports_kv_cache:
            print(f"[ONNXStudentV2] Cache layers: {len(self._cache_input_names)}")

    def _detect_cache_capability(self):
        """Detect if the model supports KV-cache from input/output names."""
        input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]

        # Common KV-cache naming patterns
        cache_patterns = [
            "past_key_values",
            "past_key_value",
            "past",
            "cache",
            "present",
            "key_cache",
            "value_cache",
        ]

        # Find cache inputs
        self._cache_input_names = []
        for name in input_names:
            name_lower = name.lower()
            if any(p in name_lower for p in cache_patterns):
                self._cache_input_names.append(name)

        # Find cache outputs (present states)
        self._cache_output_names = []
        for name in output_names:
            name_lower = name.lower()
            if any(p in name_lower for p in cache_patterns):
                self._cache_output_names.append(name)

        # Cache is supported if we have both inputs and outputs
        self._supports_kv_cache = (
            len(self._cache_input_names) > 0 and
            len(self._cache_output_names) > 0
        )

        # Sort for consistent ordering
        self._cache_input_names.sort()
        self._cache_output_names.sort()

    def _init_cache_buffers(self, batch_size: int = 1):
        """Initialize empty cache buffers."""
        if not self._supports_kv_cache:
            return

        # Get shapes from model inputs
        for inp in self._session.get_inputs():
            if inp.name in self._cache_input_names:
                shape = list(inp.shape)

                # Replace dynamic dims with concrete values
                for i, dim in enumerate(shape):
                    if isinstance(dim, str) or dim is None:
                        if "batch" in str(dim).lower() or i == 0:
                            shape[i] = batch_size
                        elif "seq" in str(dim).lower() or "past" in str(dim).lower():
                            shape[i] = 0  # Start with empty cache
                        else:
                            shape[i] = self.cache_config.head_dim if self.cache_config else 64

                dtype = np.float16 if self.cache_config and self.cache_config.dtype == "float16" else np.float32
                self._cache_buffers[inp.name] = np.zeros(shape, dtype=dtype)

    def _build_prompt(self, item: TrainingItem) -> str:
        """Build the full prompt for the model."""
        return f"{self.system_prompt}\n\nQUESTION: {item.prompt}"

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        """Sample next token from logits."""
        if temperature > 0:
            logits = logits / temperature

        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        if top_k > 0:
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_indices] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)

        if temperature > 0:
            token_id = np.random.choice(len(probs), p=probs)
        else:
            token_id = np.argmax(probs)

        return int(token_id)

    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        penalty: float,
    ) -> np.ndarray:
        """Apply repetition penalty to logits."""
        if penalty == 1.0 or not generated_ids:
            return logits

        logits = logits.copy()
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty

        return logits

    def _check_stop_conditions(
        self,
        text: str,
        token_id: int,
        num_generated: int,
    ) -> bool:
        """Check if generation should stop."""
        if token_id == self._tokenizer.eos_token_id:
            return True

        if num_generated >= self.config.max_new_tokens:
            return True

        for stop in self.config.stop_strings:
            if stop in text:
                return True

        return False

    def _generate_with_cache(self, prompt: str) -> StudentOutputV2:
        """Generate text using KV-cache for efficiency."""
        start_time = time.perf_counter()
        first_token_time = None

        # Tokenize prompt
        inputs = self._tokenizer(
            prompt,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=2048,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids))
        prompt_len = input_ids.shape[1]

        # Initialize empty caches
        self._init_cache_buffers(batch_size=1)

        generated_ids = []
        input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]

        # First forward: process full prompt
        feed_dict = {}
        if "input_ids" in input_names:
            feed_dict["input_ids"] = input_ids.astype(np.int64)
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = attention_mask.astype(np.int64)

        # Add empty caches for first pass
        for cache_name in self._cache_input_names:
            if cache_name in self._cache_buffers:
                feed_dict[cache_name] = self._cache_buffers[cache_name]

        # Position IDs if needed
        if "position_ids" in input_names:
            feed_dict["position_ids"] = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)

        outputs = self._session.run(output_names, feed_dict)

        # Parse outputs
        logits = None
        new_caches = {}

        for i, name in enumerate(output_names):
            if name == "logits" or (logits is None and "logit" in name.lower()):
                logits = outputs[i]
            elif name in self._cache_output_names or any(p in name.lower() for p in ["present", "cache", "past"]):
                # Map output cache to corresponding input name
                # Common pattern: present_key_values.X -> past_key_values.X
                input_name = name.replace("present", "past")
                if input_name not in self._cache_input_names:
                    # Try to match by index
                    idx = self._cache_output_names.index(name) if name in self._cache_output_names else i - 1
                    if idx < len(self._cache_input_names):
                        input_name = self._cache_input_names[idx]
                new_caches[input_name] = outputs[i]

        if logits is None:
            logits = outputs[0]  # Fallback

        # Sample first token
        next_token_logits = logits[0, -1, :]
        next_token_logits = self._apply_repetition_penalty(
            next_token_logits, generated_ids, self.config.repetition_penalty
        )

        if self.config.do_sample:
            next_token = self._sample_token(
                next_token_logits,
                self.config.temperature,
                self.config.top_p,
                self.config.top_k,
            )
        else:
            next_token = int(np.argmax(next_token_logits))

        generated_ids.append(next_token)
        first_token_time = time.perf_counter()

        # Update caches
        self._cache_buffers.update(new_caches)

        current_pos = prompt_len

        # Subsequent tokens: single token + cache
        for step in range(1, self.config.max_new_tokens):
            # Check stop
            current_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            if self._check_stop_conditions(current_text, next_token, len(generated_ids)):
                break

            # Prepare single-token input
            feed_dict = {}
            if "input_ids" in input_names:
                feed_dict["input_ids"] = np.array([[next_token]], dtype=np.int64)

            # Update attention mask for new position
            if "attention_mask" in input_names:
                # Attention over prompt + all generated tokens
                mask_len = current_pos + 1
                feed_dict["attention_mask"] = np.ones((1, mask_len), dtype=np.int64)

            # Position IDs
            if "position_ids" in input_names:
                feed_dict["position_ids"] = np.array([[current_pos]], dtype=np.int64)

            # Add caches
            for cache_name in self._cache_input_names:
                if cache_name in self._cache_buffers:
                    feed_dict[cache_name] = self._cache_buffers[cache_name]

            outputs = self._session.run(output_names, feed_dict)

            # Parse outputs
            logits = None
            new_caches = {}

            for i, name in enumerate(output_names):
                if name == "logits" or (logits is None and "logit" in name.lower()):
                    logits = outputs[i]
                elif name in self._cache_output_names or any(p in name.lower() for p in ["present", "cache", "past"]):
                    input_name = name.replace("present", "past")
                    if input_name not in self._cache_input_names:
                        idx = self._cache_output_names.index(name) if name in self._cache_output_names else i - 1
                        if idx < len(self._cache_input_names):
                            input_name = self._cache_input_names[idx]
                    new_caches[input_name] = outputs[i]

            if logits is None:
                logits = outputs[0]

            # Sample next token
            next_token_logits = logits[0, -1, :]
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, generated_ids, self.config.repetition_penalty
            )

            if self.config.do_sample:
                next_token = self._sample_token(
                    next_token_logits,
                    self.config.temperature,
                    self.config.top_p,
                    self.config.top_k,
                )
            else:
                next_token = int(np.argmax(next_token_logits))

            generated_ids.append(next_token)
            current_pos += 1

            # Update caches
            self._cache_buffers.update(new_caches)

        # Final decode
        end_time = time.perf_counter()
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_time_ms = (end_time - start_time) * 1000
        first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms

        answer, reasoning = self._parse_response(raw_text)

        tokens_per_sec = len(generated_ids) / (total_time_ms / 1000) if total_time_ms > 0 else 0

        metrics = GenerationMetrics(
            prompt_tokens=prompt_len,
            generated_tokens=len(generated_ids),
            first_token_ms=first_token_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_sec,
            used_kv_cache=True,
        )

        return StudentOutputV2(
            answer=answer,
            reasoning=reasoning,
            raw_text=raw_text,
            token_ids=generated_ids,
            generation_time_ms=total_time_ms,
            metrics=metrics,
        )

    def _generate_no_cache(self, prompt: str) -> StudentOutputV2:
        """Generate text without KV-cache (V1-style fallback)."""
        start_time = time.perf_counter()
        first_token_time = None

        inputs = self._tokenizer(
            prompt,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=2048,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids))
        prompt_len = input_ids.shape[1]

        generated_ids = []
        current_ids = input_ids.copy()
        current_mask = attention_mask.copy()

        input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]

        for step in range(self.config.max_new_tokens):
            feed_dict = {}
            if "input_ids" in input_names:
                feed_dict["input_ids"] = current_ids.astype(np.int64)
            if "attention_mask" in input_names:
                feed_dict["attention_mask"] = current_mask.astype(np.int64)

            outputs = self._session.run(output_names, feed_dict)

            logits = outputs[0]
            next_token_logits = logits[0, -1, :]

            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, generated_ids, self.config.repetition_penalty
            )

            if self.config.do_sample:
                next_token = self._sample_token(
                    next_token_logits,
                    self.config.temperature,
                    self.config.top_p,
                    self.config.top_k,
                )
            else:
                next_token = int(np.argmax(next_token_logits))

            generated_ids.append(next_token)

            if first_token_time is None:
                first_token_time = time.perf_counter()

            current_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            if self._check_stop_conditions(current_text, next_token, len(generated_ids)):
                break

            current_ids = np.concatenate([
                current_ids,
                np.array([[next_token]], dtype=np.int64)
            ], axis=1)
            current_mask = np.concatenate([
                current_mask,
                np.array([[1]], dtype=np.int64)
            ], axis=1)

        end_time = time.perf_counter()
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_time_ms = (end_time - start_time) * 1000
        first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms

        answer, reasoning = self._parse_response(raw_text)

        tokens_per_sec = len(generated_ids) / (total_time_ms / 1000) if total_time_ms > 0 else 0

        metrics = GenerationMetrics(
            prompt_tokens=prompt_len,
            generated_tokens=len(generated_ids),
            first_token_ms=first_token_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_sec,
            used_kv_cache=False,
        )

        return StudentOutputV2(
            answer=answer,
            reasoning=reasoning,
            raw_text=raw_text,
            token_ids=generated_ids,
            generation_time_ms=total_time_ms,
            metrics=metrics,
        )

    def _parse_response(self, text: str) -> Tuple[str, str]:
        """Parse structured response into answer and reasoning."""
        answer = ""
        reasoning = ""

        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?:CONFIDENCE:|ANSWER:|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        if not answer:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            answer = lines[-1] if lines else text[:100]

        if not reasoning:
            reasoning = text

        return answer, reasoning

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from response."""
        text_lower = text.lower()

        conf_match = re.search(r"confidence:\s*(low|medium|high|\d+\.?\d*)", text_lower)
        if conf_match:
            val = conf_match.group(1)
            if val == "low":
                return 0.3
            elif val == "medium":
                return 0.6
            elif val == "high":
                return 0.85
            else:
                try:
                    return min(1.0, max(0.0, float(val)))
                except ValueError:
                    pass

        hedge_words = ["maybe", "perhaps", "possibly", "uncertain", "not sure", "might"]
        confidence_words = ["clearly", "obviously", "definitely", "certainly", "must be"]

        hedge_count = sum(1 for w in hedge_words if w in text_lower)
        conf_count = sum(1 for w in confidence_words if w in text_lower)

        base = 0.5
        base -= hedge_count * 0.1
        base += conf_count * 0.15

        return max(0.1, min(0.95, base))

    def generate(
        self,
        item: TrainingItem,
        max_tokens: int = 256,
    ) -> StudentResponse:
        """Generate a response for a training item."""
        self._load()

        original_max = self.config.max_new_tokens
        self.config.max_new_tokens = max_tokens

        try:
            prompt = self._build_prompt(item)

            if self._supports_kv_cache:
                output = self._generate_with_cache(prompt)
            else:
                output = self._generate_no_cache(prompt)

            confidence = self._extract_confidence(output.raw_text)

            return StudentResponse(
                item_id=item.id,
                answer=output.answer,
                reasoning_trace=output.reasoning,
                confidence=confidence,
                latency_ms=output.generation_time_ms,
            )
        finally:
            self.config.max_new_tokens = original_max

    def generate_with_metrics(
        self,
        item: TrainingItem,
        max_tokens: int = 256,
    ) -> Tuple[StudentResponse, GenerationMetrics]:
        """Generate with detailed metrics returned."""
        self._load()

        original_max = self.config.max_new_tokens
        self.config.max_new_tokens = max_tokens

        try:
            prompt = self._build_prompt(item)

            if self._supports_kv_cache:
                output = self._generate_with_cache(prompt)
            else:
                output = self._generate_no_cache(prompt)

            confidence = self._extract_confidence(output.raw_text)

            response = StudentResponse(
                item_id=item.id,
                answer=output.answer,
                reasoning_trace=output.reasoning,
                confidence=confidence,
                latency_ms=output.generation_time_ms,
            )

            return response, output.metrics
        finally:
            self.config.max_new_tokens = original_max

    def update(self, signal: TrainingSignal):
        """Log training signal for offline learning."""
        pass

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        self._load()

        info = {
            "model_path": str(self.model_path),
            "tokenizer": self.tokenizer_path,
            "device": self.device,
            "providers": self._session.get_providers(),
            "vocab_size": self._tokenizer.vocab_size,
            "inputs": [inp.name for inp in self._session.get_inputs()],
            "outputs": [out.name for out in self._session.get_outputs()],
            "supports_kv_cache": self._supports_kv_cache,
            "cache_inputs": self._cache_input_names,
            "cache_outputs": self._cache_output_names,
        }

        if self.cache_config:
            info["cache_size_gb"] = self.cache_config.total_cache_size_gb

        return info

    def estimate_memory_gb(self, prompt_tokens: int, max_new_tokens: int) -> float:
        """Estimate GPU memory needed for generation.

        Used by Governor to size leases appropriately.
        """
        # Base model memory (rough estimate)
        model_size_gb = self.model_path.stat().st_size / (1024 ** 3) if self.model_path.exists() else 2.0

        # KV-cache memory
        if self._supports_kv_cache and self.cache_config:
            # Scale cache by actual sequence length
            total_seq = prompt_tokens + max_new_tokens
            cache_ratio = total_seq / self.cache_config.max_seq_len
            cache_gb = self.cache_config.total_cache_size_gb * cache_ratio
        else:
            cache_gb = 0

        # Activations (rough: ~1.5x model size for forward pass)
        activation_gb = model_size_gb * 0.5

        return model_size_gb + cache_gb + activation_gb

    @property
    def supports_kv_cache(self) -> bool:
        """Check if model supports KV-cache."""
        self._load()
        return self._supports_kv_cache
