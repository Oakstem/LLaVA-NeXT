"""
Generation Metrics and Enhanced Next-Token Generation for LLaVA-based Text Generation

This module provides comprehensive evaluation metrics and enhanced generation logic for
attention-guided text generation, including:
- Confidence metrics (entropy, logit gap)
- Repetitivity and diversity metrics
- Top-k candidate evaluation
- Enhanced next-token generation with coherency tracking
- Analysis functions for comparing configurations and detailed step analysis
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import Counter, defaultdict
import math
from datetime import datetime
import traceback
from pathlib import Path
from datetime import datetime
import traceback
from pathlib import Path


class ConfidenceMetrics:
    """Evaluate confidence and uncertainty in text generation."""
    
    def __init__(self):
        self.history = []
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute entropy of the probability distribution."""
        probs = torch.softmax(logits, dim=-1)
        # Avoid log(0) by adding small epsilon
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.item()
    
    def compute_logit_gap(self, logits: torch.Tensor) -> float:
        """Compute the gap between the highest and second-highest logits."""
        sorted_logits, _ = torch.sort(logits, descending=True)
        if len(sorted_logits) < 2:
            return 0.0
        gap = sorted_logits[0] - sorted_logits[1]
        return gap.item()
    
    def compute_confidence_metrics(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute all confidence metrics for given logits."""
        entropy = self.compute_entropy(logits)
        logit_gap = self.compute_logit_gap(logits)
        
        # Higher logit gap = higher confidence, lower entropy = higher confidence
        confidence_score = logit_gap / (entropy + 1e-10)  # Composite confidence metric
        
        metrics = {
            "entropy": entropy,
            "logit_gap": logit_gap,
            "confidence_score": confidence_score,
            "uncertainty": entropy  # Alias for clarity
        }
        
        return metrics
    
    def update_history(self, step: int, metrics: Dict[str, float]) -> None:
        """Update the history with new metrics."""
        entry = {"step": step, **metrics}
        self.history.append(entry)
    
    def get_average_confidence(self) -> Dict[str, float]:
        """Get average confidence metrics across all steps."""
        if not self.history:
            return {}
        
        metrics_sum = defaultdict(float)
        for entry in self.history[:-1]:     # don't consider the last step as often EOS tokens have extremely high confidence
            for key, value in entry.items():
                if key != "step":
                    if key == 'confidence_score':
                        # Clip confidence score to a reasonable range
                        value = max(0.0, min(value, 5.0))  # Clip to [0, 5]
                    metrics_sum[key] += value
        
        num_steps = len(self.history)
        return {key: value / num_steps for key, value in metrics_sum.items()}


class RepetitivityMetrics:
    """Evaluate repetitivity and diversity in generated text."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.token_history = []
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
    
    def add_token(self, token_id: int, token_text: str) -> None:
        """Add a new token to the history."""
        self.token_history.append({"id": token_id, "text": token_text})
        self._update_ngram_counts()
    
    def _update_ngram_counts(self) -> None:
        """Update n-gram counts with the latest token."""
        tokens = [t["text"] for t in self.token_history]
        
        # Update n-gram counts for n=1 to window_size
        for n in range(1, min(len(tokens) + 1, self.window_size + 1)):
            if len(tokens) >= n:
                ngram = tuple(tokens[-n:])
                self.ngram_counts[n][ngram] += 1
    
    def compute_repetition_penalty(self, n: int = 2) -> float:
        """Compute repetition penalty for n-grams."""
        if len(self.token_history) < n:
            return 0.0
        
        ngrams = self.ngram_counts[n]
        if not ngrams:
            return 0.0
        
        total_ngrams = sum(ngrams.values())
        repeated_ngrams = sum(count - 1 for count in ngrams.values() if count > 1)
        
        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def compute_diversity_metrics(self) -> Dict[str, float]:
        """Compute various diversity metrics."""
        if not self.token_history:
            return {}
        
        tokens = [t["text"] for t in self.token_history]
        unique_tokens = set(tokens)
        
        # Type-Token Ratio (TTR)
        ttr = len(unique_tokens) / len(tokens) if tokens else 0.0
        
        # Entropy-based diversity
        token_counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum((count / total) * math.log(count / total) for count in token_counts.values())
        
        # Repetition penalties for different n-grams
        rep_penalties = {}
        for n in range(2, min(len(tokens) + 1, self.window_size + 1)):
            rep_penalties[f"repetition_penalty_{n}gram"] = self.compute_repetition_penalty(n)
        
        return {
            "type_token_ratio": ttr,
            "entropy_diversity": entropy,
            "unique_tokens": len(unique_tokens),
            "total_tokens": len(tokens),
            **rep_penalties
        }
    
    def get_recent_repetitions(self, lookback: int = 10) -> List[Tuple[str, int]]:
        """Get recent repetitions within the lookback window."""
        if len(self.token_history) < lookback:
            recent_tokens = [t["text"] for t in self.token_history]
        else:
            recent_tokens = [t["text"] for t in self.token_history[-lookback:]]
        
        token_counts = Counter(recent_tokens)
        repetitions = [(token, count) for token, count in token_counts.items() if count > 1]
        return sorted(repetitions, key=lambda x: x[1], reverse=True)


class TopKCandidateEvaluator:
    """Evaluate and analyze top-k candidates at each generation step."""
    
    def __init__(self, k: int = 5, tokenizer=None):
        self.k = k
        self.tokenizer = tokenizer
        self.candidate_history = []
    
    def evaluate_candidates(self, logits: torch.Tensor, step: int) -> Dict[str, Any]:
        """Evaluate top-k candidates for the current step."""
        # Ensure logits are 1D (squeeze any batch dimensions)
        if logits.dim() > 1:
            logits = logits.squeeze()
        
        # Get top-k candidates
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        candidates = []
        for i in range(self.k):
            # Handle tensor indexing properly for different tensor shapes
            if top_k_indices.dim() == 0:  # scalar tensor (k=1 case)
                token_id = top_k_indices.item()
                logit_score = top_k_logits.item()
                prob_score = top_k_probs.item()
            elif top_k_indices.dim() == 1:  # 1D tensor
                token_id = top_k_indices[i].item()
                logit_score = top_k_logits[i].item()
                prob_score = top_k_probs[i].item()
            else:  # Multi-dimensional - shouldn't happen but just in case
                # Flatten and take the i-th element
                token_id = top_k_indices.flatten()[i].item()
                logit_score = top_k_logits.flatten()[i].item()
                prob_score = top_k_probs.flatten()[i].item()
            
            # Get token text if tokenizer available
            token_text = "UNK"
            if self.tokenizer:
                try:
                    token_text = self.tokenizer.decode([token_id]).strip()
                except:
                    token_text = f"ID_{token_id}"
            
            candidates.append({
                "rank": i + 1,
                "token_id": token_id,
                "token_text": token_text,
                "logit_score": logit_score,
                "probability": prob_score,
                "log_probability": math.log(prob_score + 1e-10)
            })
        
        # Compute candidate distribution metrics
        probs = [c["probability"] for c in candidates]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        
        prob_mass_top3 = sum(probs[:3])
        prob_mass_top1 = probs[0] if probs else 0.0
        
        evaluation = {
            "step": step,
            "candidates": candidates,
            "metrics": {
                "candidate_entropy": entropy,
                "top1_probability": prob_mass_top1,
                "top3_probability_mass": prob_mass_top3,
                "probability_spread": max(probs) - min(probs) if probs else 0.0,
                "logit_spread": max(c["logit_score"] for c in candidates) - min(c["logit_score"] for c in candidates) if candidates else 0.0
            }
        }
        
        self.candidate_history.append(evaluation)
        return evaluation
    
    def get_alternative_paths(self, step: int, top_n: int = 3) -> List[str]:
        """Get alternative generation paths by selecting different top candidates."""
        if step >= len(self.candidate_history):
            return []
        
        step_candidates = self.candidate_history[step]["candidates"]
        alternatives = []
        
        for i in range(min(top_n, len(step_candidates))):
            candidate = step_candidates[i]
            alt_text = f"Rank {candidate['rank']}: '{candidate['token_text']}' (p={candidate['probability']:.3f})"
            alternatives.append(alt_text)
        
        return alternatives
    
    def analyze_decision_points(self, confidence_history: dict, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Identify steps where the model had low confidence (potential decision points)."""
        decision_points = []
        
        for conf, evaluation in zip(confidence_history, self.candidate_history):
            metrics = evaluation["metrics"]
            top1_prob = metrics["top1_probability"]
            
            if top1_prob < confidence_threshold:
                decision_points.append({
                    "step": evaluation["step"],
                    "top1_probability": top1_prob,
                    "entropy": metrics["candidate_entropy"],
                    "alternatives": self.get_alternative_paths(evaluation["step"]),
                    "confidence": conf.get("confidence_score", 0.0),
                })
        
        return decision_points


def generate_next_token_with_evaluation(
    model_inputs: Dict,
    model,
    tokenizer,
    gen_config: Dict,
    confidence_tracker: ConfidenceMetrics,
    repetitivity_tracker: RepetitivityMetrics,
    candidate_evaluator: TopKCandidateEvaluator,
    step: int
) -> Tuple[torch.Tensor, str, Any, Dict[str, Any]]:
    """
    Enhanced next-token generation with comprehensive evaluation metrics.
    
    Returns:
        - next_token_id: The selected token ID
        - token_text: The decoded token text
        - model_outputs: Full model outputs
        - evaluation_metrics: Comprehensive metrics for this step
    """
    # Forward pass
    outputs = model(**model_inputs)
    next_token_logits = outputs.logits[:, -1, :]
    
    # Ensure we have the right shape for evaluation - squeeze batch dimension if present
    if next_token_logits.dim() > 1:
        logits_for_eval = next_token_logits.squeeze(0)  # Remove batch dimension for evaluation
    else:
        logits_for_eval = next_token_logits
    
    # Compute confidence metrics before any filtering
    confidence_metrics = confidence_tracker.compute_confidence_metrics(logits_for_eval)
    
    # Evaluate top-k candidates before filtering
    candidate_eval = candidate_evaluator.evaluate_candidates(logits_for_eval, step)
    
    # Apply generation strategy (top-k filtering, temperature, sampling)
    if gen_config.get("do_sample", False):
        # 1. Apply top-k filtering first (if enabled)
        if gen_config.get("top_k") is not None and gen_config.get("top_k") > 0:
            top_k = gen_config["top_k"]
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            filtered_logits = torch.full_like(next_token_logits, float('-inf'))
            filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
            next_token_logits = filtered_logits
        
        # 2. Apply temperature scaling
        if gen_config.get("temperature", 1.0) != 1.0:
            next_token_logits = next_token_logits / gen_config["temperature"]
        
        # 3. Sample from the distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Ensure probs is 2D for multinomial sampling: [batch_size, vocab_size]
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)  # Add batch dimension: [1, vocab_size]
        elif probs.dim() > 2:
            # Flatten to 2D while preserving the last dimension as vocab_size
            probs = probs.view(-1, probs.size(-1))
            # If we have multiple batches, take only the first one
            if probs.size(0) > 1:
                probs = probs[:1]  # Keep only first batch: [1, vocab_size]
        
        # Ensure probs sums to 1 along vocab dimension and handle numerical issues
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Clamp to avoid numerical issues with multinomial
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize after clamping
        
        # Sample with robust error handling
        try:
            next_token_id = torch.multinomial(probs, 1)
        except Exception as e:
            # Fallback to argmax if sampling fails
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
    else:
        # Greedy decoding
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    # Convert to scalar - handle any tensor shape robustly
    # Flatten the tensor and take the first element to ensure we get a scalar
    if isinstance(next_token_id, torch.Tensor):
        # Flatten and take first element
        token_id_scalar = next_token_id.flatten()[0].item()
    else:
        # If it's already a scalar
        token_id_scalar = int(next_token_id)
    
    # Validate the result is actually a Python int
    if not isinstance(token_id_scalar, int):
        token_id_scalar = int(token_id_scalar)
    
    # Decode token
    token_text = tokenizer.decode([token_id_scalar]).strip()
    
    # Update trackers
    confidence_tracker.update_history(step, confidence_metrics)
    repetitivity_tracker.add_token(token_id_scalar, token_text)
    
    # Compute repetitivity metrics
    repetitivity_metrics = repetitivity_tracker.compute_diversity_metrics()
    recent_repetitions = repetitivity_tracker.get_recent_repetitions()
    
    # Compile evaluation metrics
    evaluation_metrics = {
        "step": step,
        "selected_token": {
            "id": token_id_scalar,
            "text": token_text
        },
        "confidence": confidence_metrics,
        "repetitivity": repetitivity_metrics,
        "recent_repetitions": recent_repetitions,
        "top_k_analysis": candidate_eval,
        "generation_config_used": {
            "top_k": gen_config.get("top_k"),
            "temperature": gen_config.get("temperature", 1.0),
            "do_sample": gen_config.get("do_sample", False)
        }
    }
    
    return next_token_id, token_text, outputs, evaluation_metrics


def calculate_average_correlation(all_correlation_metrics: Optional[List[Dict[str, float]]]) -> Dict[str, float]:
    """Calculate the average of correlation metrics over all steps."""
    if not all_correlation_metrics:
        return {}
    
    correlation_sum = defaultdict(float)
    num_metrics = len(all_correlation_metrics)
    
    if num_metrics == 0:
        return {}
        
    for metrics_dict in all_correlation_metrics:
        for key, value in metrics_dict.items():
            correlation_sum[key] += value
            
    avg_correlation = {key: value / num_metrics for key, value in correlation_sum.items()}
    return avg_correlation


def create_generation_summary(
    confidence_tracker: ConfidenceMetrics,
    repetitivity_tracker: RepetitivityMetrics,
    candidate_evaluator: TopKCandidateEvaluator,
    generated_text: str,
    all_step_metrics: List[Dict[str, Any]],
    all_correlation_metrics: Optional[List[Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive summary of the generation process.
    """
    # Average confidence metrics
    avg_confidence = confidence_tracker.get_average_confidence()
    
    # Final diversity metrics
    diversity_metrics = repetitivity_tracker.compute_diversity_metrics()
    
    # Decision points analysis
    decision_points = candidate_evaluator.analyze_decision_points(confidence_tracker.history, confidence_threshold=0.5)
    
    # Average attention correlation metrics
    avg_correlation = calculate_average_correlation(all_correlation_metrics)

    # Confidence trajectory analysis
    confidence_scores = [step["confidence"]["confidence_score"] for step in all_step_metrics]
    entropy_scores = [step["confidence"]["entropy"] for step in all_step_metrics]
    
    confidence_trend = "stable"
    if len(confidence_scores) > 1:
        initial_conf = np.mean(confidence_scores[:len(confidence_scores)//3]) if confidence_scores else 0
        final_conf = np.mean(confidence_scores[-len(confidence_scores)//3:]) if confidence_scores else 0
        
        if final_conf > initial_conf * 1.1:
            confidence_trend = "increasing"
        elif final_conf < initial_conf * 0.9:
            confidence_trend = "decreasing"
    
    # Quality assessment
    quality_indicators = {
        "high_confidence_steps": sum(1 for score in confidence_scores if score > 1.0),
        "low_confidence_steps": sum(1 for score in confidence_scores if score < 0.5),
        "avg_entropy": np.mean(entropy_scores) if entropy_scores else 0,
        "entropy_variance": np.var(entropy_scores) if entropy_scores else 0,
        "num_decision_points": len(decision_points),
        "repetition_issues": any(rep[1] > 2 for rep in repetitivity_tracker.get_recent_repetitions())
    }
    
    summary = {
        "generated_text": generated_text,
        "num_tokens": len(all_step_metrics),
        "average_confidence": avg_confidence,
        "diversity_metrics": diversity_metrics,
        "confidence_trajectory": {
            "trend": confidence_trend,
            "scores": confidence_scores,
            "entropy_scores": entropy_scores
        },
        "decision_points": decision_points,
        "quality_indicators": quality_indicators,
        "step_by_step_metrics": all_step_metrics
    }

    if avg_correlation:
        summary["average_attention_correlation"] = avg_correlation
    
    return summary


def analyze_generation_quality(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the overall quality of the generation based on computed metrics.
    """
    quality_score = 0.0
    quality_factors = {}
    
    # Confidence factor (0-2 points)
    avg_confidence = summary["average_confidence"].get("confidence_score", 0)
    if avg_confidence > 1.0:
        confidence_factor = min(2.0, avg_confidence / 2)
    else:
        confidence_factor = avg_confidence
    quality_score += confidence_factor
    quality_factors["confidence"] = confidence_factor
    
    # Diversity factor (0-2 points)
    num_tokens = summary.get("num_tokens", 0)
    if num_tokens < 2:
        diversity_factor = 0.0
    else:
        ttr = summary["diversity_metrics"].get("type_token_ratio", 0)
        diversity_factor = min(2.0, ttr * 4)  # Scale TTR to 0-2 range
    quality_score += diversity_factor
    quality_factors["diversity"] = diversity_factor
    
    # Repetition penalty (-2 to 0 points)
    rep_penalty_2gram = summary["diversity_metrics"].get("repetition_penalty_2gram", 0)
    repetition_factor = -min(2.0, rep_penalty_2gram * 8)  # Penalty for repetition
    quality_score += repetition_factor
    quality_factors["repetition"] = repetition_factor
    
    # Stability factor (0-1 points)
    entropy_var = summary["quality_indicators"].get("entropy_variance", 0)
    stability_factor = max(0, 1.0 - entropy_var)  # Lower variance = more stable
    quality_score += stability_factor
    quality_factors["stability"] = stability_factor

    # Attention Correlation factor (0-2.5 points)
    attention_factor = 0.0
    if "average_attention_correlation" in summary and summary["average_attention_correlation"]:
        avg_corr = summary["average_attention_correlation"]
        correlation_score = avg_corr.get("normalized_correlation_score", 0.0)
        focus_ratio = avg_corr.get("attention_focus_ratio", 1.0)

        # Scale correlation score to 0-1.5 points
        correlation_component = min(1.5, correlation_score * 5.)
        # Scale focus ratio to 0-1.0 points. A ratio of 1 is neutral (0 points).
        # A ratio of 11 or more gives the full 1.0 point.
        focus_component = min(1.0, max(0, (focus_ratio - 1) / 10.0))

        attention_factor = correlation_component + focus_component
        quality_score += attention_factor
    quality_factors["attention_correlation"] = attention_factor
    
    # Normalize to 0-10 scale
    max_possible_score = 7.5  # 2(conf) + 2(div) + 0(rep) + 1(stab) + 2.5(attn)
    normalized_score = (quality_score / max_possible_score) * 10
    normalized_score = max(0, min(10, normalized_score))
    
    return {
        "overall_quality_score": normalized_score,
        "quality_factors": quality_factors,
        "raw_score": quality_score,
        "interpretation": {
            "excellent": normalized_score >= 8.0,
            "good": 6.0 <= normalized_score < 8.0,
            "fair": 4.0 <= normalized_score < 6.0,
            "poor": normalized_score < 4.0
        }
    }


def calculate_attention_correlation_from_similarity(
    text_to_image_similarity_matrix: np.ndarray,
    attention_mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculates the correlation between a similarity matrix (of image shape) and a given binary mask.

    Args:
        text_to_image_similarity_matrix: 2D numpy array representing similarity scores (e.g., image shape).
        attention_mask: 2D numpy array (same shape) with boolean or 0/1 values indicating target regions.
        threshold: (Unused, kept for compatibility).

    Returns:
        A dictionary containing attention correlation metrics.
    """
    if (
        text_to_image_similarity_matrix is None or text_to_image_similarity_matrix.size == 0 or
        attention_mask is None or attention_mask.size == 0 or
        text_to_image_similarity_matrix.shape != attention_mask.shape
    ):
        return {
            "target_attention_mean": 0.0,
            "off_target_attention_mean": 0.0,
            "attention_focus_ratio": 0.0,
            "normalized_correlation_score": 0.0
        }
    # threshold the similarity matrix to binary values
    text_to_image_similarity_matrix = np.where(text_to_image_similarity_matrix > threshold, 1.0, 0.0)

    flat_similarity = text_to_image_similarity_matrix.flatten()
    flat_mask = attention_mask.astype(bool).flatten()
    total_mask_count = np.sum(flat_mask)
    total_off_mask_count = flat_mask.size - total_mask_count
    

    # On-target: where mask is True; Off-target: where mask is False
    on_target_similarity = flat_similarity[flat_mask]
    off_target_similarity = flat_similarity[~flat_mask]

    mean_on_target = np.sum(on_target_similarity) / total_mask_count if on_target_similarity.size > 0 else 0.0
    mean_off_target = np.sum(off_target_similarity) / total_off_mask_count if off_target_similarity.size > 0 else 0.0

    # Ratio of on-target to off-target similarity
    focus_ratio = np.sum(on_target_similarity) / (np.sum(off_target_similarity) + 1e-9)

    # Normalized score (0 to 1), where 1 is perfect focus
    correlation_score = max(0, (mean_on_target - mean_off_target) / (mean_on_target + mean_off_target + 1e-9))

    return {
        "target_attention_mean": float(mean_on_target),
        "off_target_attention_mean": float(mean_off_target),
        "attention_focus_ratio": float(focus_ratio),
        "normalized_correlation_score": float(correlation_score)
    }


def print_detailed_step_analysis(results: Dict, max_steps: int = 10):
    """Print detailed step-by-step analysis from generation results."""
    if "step_metrics" not in results or not results["step_metrics"]:
        print("No step metrics available for analysis.")
        return
    
    step_metrics = results["step_metrics"]
    print(f"\n{'='*60}")
    print("DETAILED STEP-BY-STEP ANALYSIS")
    print(f"{'='*60}")
    print(f"Showing first {min(max_steps, len(step_metrics))} steps:")
    
    for i, step in enumerate(step_metrics[:max_steps]):
        print(f"\n--- Step {step['step']}: '{step['selected_token']['text']}' ---")
        
        # Confidence metrics
        conf = step["confidence"]
        print(f"Confidence: {conf.get('confidence_score', 0):.3f} | "
              f"Entropy: {conf.get('entropy', 0):.3f} | "
              f"Logit Gap: {conf.get('logit_gap', 0):.3f}")
        
        # Top candidates
        candidates = step["top_k_analysis"]["candidates"]
        print("Top 3 candidates:")
        for j, candidate in enumerate(candidates[:3]):
            selected = "→" if j == 0 else " "
            print(f"  {selected} {candidate['rank']}. '{candidate['token_text']}' "
                  f"(p={candidate['probability']:.3f}, logit={candidate['logit_score']:.2f})")
        
        # Repetitivity (for later steps)
        if step["step"] >= 2:
            rep = step["repetitivity"]
            print(f"Diversity: TTR={rep.get('type_token_ratio', 0):.3f} | "
                  f"Recent repetitions: {len(step.get('recent_repetitions', []))}")
    
    if len(step_metrics) > max_steps:
        print(f"\n... and {len(step_metrics) - max_steps} more steps")


def compare_generation_configs(config_a: Dict, config_b: Dict, image_path: str, 
                              mask_path: str, prompt: str, 
                              run_generation_fn: callable = None) -> Dict:
    """Compare two generation configurations and return analysis results.
    
    Args:
        config_a: First generation configuration
        config_b: Second generation configuration
        image_path: Path to image file
        mask_path: Path to mask file
        prompt: Text prompt
        run_generation_fn: Function to run generation with attention (pass run_generation_with_attention)
    """
    if run_generation_fn is None:
        raise ValueError("run_generation_fn parameter is required. Pass run_generation_with_attention function.")
    
    print(f"\n{'='*60}")
    print("COMPARING GENERATION CONFIGURATIONS")
    print(f"{'='*60}")
    
    results = {}
    
    for i, (config_name, gen_config) in enumerate([("Config A", config_a), ("Config B", config_b)]):
        print(f"\nRunning {config_name}: {gen_config}")
        output_dir = f"attention_output/comparison_{config_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            result = run_generation_fn(
                image_path=image_path,
                mask_path=mask_path,
                prompt=prompt,
                output_dir=output_dir,
                generation_config=gen_config,
                attention_config=None,  # Use defaults
                bias_strength=gen_config.get("bias_strength", 0.0)
            )
            results[config_name] = result
            
            # Quick summary
            print(f"{config_name} Results:")
            print(f"  Generated text: '{result['generated_text']}'")
            if result.get("quality_analysis"):
                score = result["quality_analysis"].get("overall_quality_score", 0)
                print(f"  Quality score: {score:.2f}/10")
        
        except Exception as e:
            print(f"Error running {config_name}: {e}")
            results[config_name] = None
    
    # Comparison analysis
    if all(results.values()):
        print(f"\n{'='*40}")
        print("COMPARISON SUMMARY")
        print(f"{'='*40}")
        
        for config_name, result in results.items():
            quality = result.get("quality_analysis", {})
            summary = result.get("evaluation_summary", {})
            
            print(f"\n{config_name}:")
            print(f"  Text: '{result['generated_text']}'")
            print(f"  Quality: {quality.get('overall_quality_score', 0):.2f}/10")
            print(f"  Tokens: {result['num_tokens']}")
            
            if summary:
                gen_metrics = summary.get("generation_metrics", {})
                print(f"  Avg Confidence: {gen_metrics.get('avg_confidence', 0):.3f}")
                print(f"  Decision Points: {len(summary.get('decision_points', []))}")
    
    return results


def analyze_top_k_impact(image_path: str, mask_path: str, prompt: str, 
                        top_k_values: List[int] = [None, 10, 50, 100],
                        run_generation_fn: callable = None) -> Dict:
    """Analyze the impact of different top-k values on generation quality.
    
    Args:
        image_path: Path to image file
        mask_path: Path to mask file
        prompt: Text prompt
        top_k_values: List of top-k values to test
        run_generation_fn: Function to run generation with attention (pass run_generation_with_attention)
    """
    if run_generation_fn is None:
        raise ValueError("run_generation_fn parameter is required. Pass run_generation_with_attention function.")
    
    print(f"\n{'='*60}")
    print("ANALYZING TOP-K SAMPLING IMPACT")
    print(f"{'='*60}")
    
    base_config = {
        "max_new_tokens": 30,
        "temperature": 0.8,
        "do_sample": True,
        "bias_strength": 1.0
    }
    
    results = {}
    
    for top_k in top_k_values:
        config = base_config.copy()
        config["top_k"] = top_k
        
        config_name = f"top_k_{top_k}" if top_k is not None else "no_top_k"
        print(f"\nTesting top_k = {top_k}...")
        
        output_dir = f"attention_output/top_k_analysis_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            result = run_generation_fn(
                image_path=image_path,
                mask_path=mask_path,
                prompt=prompt,
                output_dir=output_dir,
                generation_config=config,
                attention_config=None,
                bias_strength=config["bias_strength"]
            )
            results[config_name] = result
            
            # Quick summary
            quality_score = 0
            if result.get("quality_analysis"):
                quality_score = result["quality_analysis"].get("overall_quality_score", 0)
            
            print(f"  Result: '{result['generated_text']}'")
            print(f"  Quality: {quality_score:.2f}/10")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config_name] = None
    
    # Summary table
    print(f"\n{'='*60}")
    print("TOP-K IMPACT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Top-K':<10} {'Quality':<8} {'Tokens':<7} {'Generated Text'}")
    print("-" * 60)
    
    for config_name, result in results.items():
        if result:
            top_k_val = config_name.replace("top_k_", "").replace("no_top_k", "None")
            quality = result.get("quality_analysis", {}).get("overall_quality_score", 0)
            tokens = result.get("num_tokens", 0)
            text = result.get("generated_text", "")[:40] + ("..." if len(result.get("generated_text", "")) > 40 else "")
            print(f"{top_k_val:<10} {quality:<8.2f} {tokens:<7} {text}")
    
    return results
