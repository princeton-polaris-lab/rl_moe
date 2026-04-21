"""
EEP (Efficient Expert Pruning) implementation.

Following paper: https://arxiv.org/pdf/2407.00945

Key concepts:
- W_RM (Router Mapping matrix): Maps original routing weights to pruned experts
- W_EM (Expert Merging matrix): Merges expert weights
- Two phases:
  1. Pruning phase: W_RM = W_EM, both one-hot (discrete search)
  2. Merging phase: W_RM and W_EM can have continuous values

The evolutionary search uses accuracy on calibration data as fitness.
"""

import re
import copy
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

# For answer extraction and equivalence checking
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from is_equiv import is_equiv


@dataclass
class EEPIndividual:
    """
    Represents one individual in the evolutionary population.
    
    For each layer, we have:
    - expert_selection: List of r expert indices (which experts are kept)
    - For merging phase: merging_coeffs per retained expert
    """
    # Layer -> list of selected expert indices
    expert_selections: Dict[int, List[int]]
    
    # Layer -> tensor of shape (r, n) for merging coefficients
    # In pruning phase, this is one-hot
    # In merging phase, this can be continuous
    merging_coeffs: Optional[Dict[int, torch.Tensor]] = None
    
    # Cached fitness score
    fitness: float = 0.0


class EEPSelector:
    """
    EEP-based expert selection using evolutionary search.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        num_experts_to_keep: int = 16,
        population_size: int = 30,
        pruning_iterations: int = 40,
        merging_iterations: int = 160,
        top_k_parents: int = 10,
        mutation_rate: float = 0.1,
    ):
        """
        Args:
            model: The MoE model
            tokenizer: Tokenizer for generation
            num_experts_to_keep: r - number of experts to keep per layer
            population_size: Size of the evolutionary population
            pruning_iterations: Iterations for pruning phase
            merging_iterations: Iterations for merging phase
            top_k_parents: Number of top individuals to use as parents
            mutation_rate: Probability of mutation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_experts_to_keep = num_experts_to_keep
        self.population_size = population_size
        self.pruning_iterations = pruning_iterations
        self.merging_iterations = merging_iterations
        self.top_k_parents = top_k_parents
        self.mutation_rate = mutation_rate
        
        # Find MoE layers
        self.moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'router') and hasattr(module, 'experts'):
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    self.moe_layers.append((layer_idx, name, module))
        
        self.moe_layers.sort(key=lambda x: x[0])
        self.num_layers = len(self.moe_layers)
        
        if self.moe_layers:
            first_router = self.moe_layers[0][2].router
            self.num_total_experts = first_router.weight.shape[0]
        else:
            self.num_total_experts = 32
        
        print(f"[EEP] Found {self.num_layers} MoE layers with {self.num_total_experts} experts")
        print(f"[EEP] Will keep {num_experts_to_keep} experts per layer")
        print(f"[EEP] Population size: {population_size}")
        print(f"[EEP] Iterations: {pruning_iterations} (prune) + {merging_iterations} (merge)")
    
    def create_random_individual(self) -> EEPIndividual:
        """Create a random individual (random expert selection per layer)."""
        selections = {}
        for layer_idx, _, _ in self.moe_layers:
            selections[layer_idx] = random.sample(
                range(self.num_total_experts),
                self.num_experts_to_keep
            )
        return EEPIndividual(expert_selections=selections)
    
    def crossover(self, parent1: EEPIndividual, parent2: EEPIndividual) -> EEPIndividual:
        """
        Crossover two parents to create offspring.
        For each layer, randomly pick one parent's selection.
        """
        child_selections = {}
        for layer_idx in parent1.expert_selections:
            if random.random() < 0.5:
                child_selections[layer_idx] = parent1.expert_selections[layer_idx].copy()
            else:
                child_selections[layer_idx] = parent2.expert_selections[layer_idx].copy()
        return EEPIndividual(expert_selections=child_selections)
    
    def mutate_pruning(self, individual: EEPIndividual) -> EEPIndividual:
        """
        Mutate an individual during pruning phase.
        Replace some selected experts with random non-selected ones.
        """
        new_selections = {}
        
        for layer_idx, selected in individual.expert_selections.items():
            new_selected = selected.copy()
            
            for i in range(len(new_selected)):
                if random.random() < self.mutation_rate:
                    # Find a non-selected expert to swap in
                    non_selected = [e for e in range(self.num_total_experts) if e not in new_selected]
                    if non_selected:
                        # Replace this expert with a random non-selected one
                        new_expert = random.choice(non_selected)
                        new_selected[i] = new_expert
            
            new_selections[layer_idx] = new_selected
        
        return EEPIndividual(expert_selections=new_selections)
    
    def apply_expert_selection(self, expert_selections: Dict[int, List[int]]):
        """
        Apply expert selection by modifying router biases.
        Non-selected experts get large negative bias.
        """
        large_negative = -1e9
        device = next(self.model.parameters()).device
        
        for layer_idx, name, mlp_module in self.moe_layers:
            router = mlp_module.router
            selected_set = set(expert_selections.get(layer_idx, range(self.num_total_experts)))
            
            # Modify router bias
            with torch.no_grad():
                for exp_idx in range(self.num_total_experts):
                    if exp_idx not in selected_set:
                        router.bias.data[exp_idx] = large_negative
    
    def restore_router_biases(self, original_biases: Dict[int, torch.Tensor]):
        """Restore original router biases."""
        for layer_idx, name, mlp_module in self.moe_layers:
            router = mlp_module.router
            if layer_idx in original_biases:
                router.bias.data.copy_(original_biases[layer_idx])
    
    def save_router_biases(self) -> Dict[int, torch.Tensor]:
        """Save current router biases."""
        biases = {}
        for layer_idx, name, mlp_module in self.moe_layers:
            router = mlp_module.router
            biases[layer_idx] = router.bias.data.clone()
        return biases
    
    def evaluate_fitness(
        self,
        individual: EEPIndividual,
        problems: List[Dict],
        original_biases: Dict[int, torch.Tensor],
        max_new_tokens: int = 512,
    ) -> float:
        """
        Evaluate fitness of an individual by computing accuracy on problems.
        
        Args:
            individual: The individual to evaluate
            problems: List of problem dicts with 'problem' and 'answer' keys
            original_biases: Original router biases to restore after
            max_new_tokens: Max tokens for generation
            
        Returns:
            Accuracy (0.0 to 1.0)
        """
        # Apply expert selection
        self.apply_expert_selection(individual.expert_selections)
        
        correct = 0
        total = 0
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        for problem in problems:
            prompt = self._format_prompt(problem["problem"])
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Extract answer
            predicted = self._extract_answer(response)
            
            if is_equiv(predicted, problem.get("answer")):
                correct += 1
            total += 1
        
        # Restore original biases
        self.restore_router_biases(original_biases)
        
        return correct / total if total > 0 else 0.0
    
    def _format_prompt(self, problem: str) -> str:
        """Format problem as prompt."""
        return f"""Solve the following math problem. Show your reasoning step by step.

CRITICAL: You MUST wrap your final answer in <answer></answer> tags.

Problem: {problem}

Solution:"""
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer>...</answer> tags."""
        import re
        pattern = r'<answer>((?:(?!<answer>).)*?)</answer>'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        if not matches:
            return None
        
        answer = matches[-1].group(1).strip()
        return answer if answer else None
    
    def run_pruning_phase(
        self,
        problems: List[Dict],
        original_biases: Dict[int, torch.Tensor],
    ) -> List[EEPIndividual]:
        """
        Run the pruning phase of EEP.
        
        Uses evolutionary search with discrete (one-hot) expert selections.
        
        Returns:
            Final population after pruning phase
        """
        print(f"\n[EEP] Starting PRUNING phase ({self.pruning_iterations} iterations)")
        
        # Initialize population with random individuals
        population = [self.create_random_individual() for _ in range(self.population_size)]
        
        for iteration in tqdm(range(self.pruning_iterations), desc="Pruning phase"):
            # Evaluate all individuals
            for individual in population:
                if individual.fitness == 0.0:  # Only evaluate if not cached
                    individual.fitness = self.evaluate_fitness(
                        individual, problems, original_biases
                    )
            
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Report best fitness
            if iteration % 10 == 0 or iteration == self.pruning_iterations - 1:
                best = population[0].fitness
                avg = sum(ind.fitness for ind in population) / len(population)
                print(f"  Iter {iteration}: best={best:.3f}, avg={avg:.3f}")
            
            # Create next generation
            next_gen = []
            
            # Keep top performers (elitism)
            next_gen.extend(population[:self.top_k_parents])
            
            # Fill rest with offspring
            while len(next_gen) < self.population_size:
                # Select two parents from top performers
                parent1 = random.choice(population[:self.top_k_parents])
                parent2 = random.choice(population[:self.top_k_parents])
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutate
                child = self.mutate_pruning(child)
                
                next_gen.append(child)
            
            population = next_gen
        
        # Final evaluation
        for individual in population:
            individual.fitness = self.evaluate_fitness(
                individual, problems, original_biases
            )
        
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        print(f"[EEP] Pruning phase complete. Best fitness: {population[0].fitness:.3f}")
        
        return population
    
    def run(
        self,
        problems: List[Dict],
        skip_merging: bool = True,  # For now, skip merging phase
    ) -> Dict[int, List[int]]:
        """
        Run EEP expert selection.
        
        Args:
            problems: List of problem dicts with 'problem' and 'answer' keys
            skip_merging: If True, skip the merging phase (return after pruning)
            
        Returns:
            Dict mapping layer_idx -> list of selected expert indices
        """
        print(f"\n[EEP] Starting EEP with {len(problems)} calibration problems")
        
        # Save original router biases
        original_biases = self.save_router_biases()
        
        try:
            # Run pruning phase
            population = self.run_pruning_phase(problems, original_biases)
            
            # Get best individual
            best = population[0]
            
            if not skip_merging:
                print("[EEP] Merging phase not implemented, using pruning result")
            
            print(f"\n[EEP] Final best fitness: {best.fitness:.3f}")
            print(f"[EEP] Selected experts (showing first 3 layers):")
            for layer_idx in sorted(best.expert_selections.keys())[:3]:
                print(f"  Layer {layer_idx}: {sorted(best.expert_selections[layer_idx])}")
            
            return best.expert_selections
            
        finally:
            # Always restore original biases
            self.restore_router_biases(original_biases)


def run_eep_pruning(
    model,
    tokenizer,
    problems: List[Dict],
    num_experts_to_keep: int = 16,
    total_iterations: int = 200,
    population_size: int = 30,
) -> Dict[int, List[int]]:
    """
    Convenience function to run EEP expert pruning.
    
    Args:
        model: MoE model
        tokenizer: Tokenizer
        problems: List of problem dicts with 'problem' and 'answer' keys
        num_experts_to_keep: Number of experts to keep per layer
        total_iterations: Total iterations (split 40/160 between pruning/merging)
        population_size: Population size for evolutionary search
        
    Returns:
        Dict mapping layer_idx -> list of selected expert indices
    """
    # Split iterations: 20% for pruning, 80% for merging (following paper: 40+160=200)
    pruning_iters = int(total_iterations * 0.2)
    merging_iters = total_iterations - pruning_iters
    
    selector = EEPSelector(
        model=model,
        tokenizer=tokenizer,
        num_experts_to_keep=num_experts_to_keep,
        population_size=population_size,
        pruning_iterations=pruning_iters,
        merging_iterations=merging_iters,
    )
    
    return selector.run(problems, skip_merging=True)
