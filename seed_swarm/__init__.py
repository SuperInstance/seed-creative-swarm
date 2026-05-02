"""
Seed Creative Swarm — Ensemble of N Seed processes for divergent generation

Architecture:
- 3 Seed-mini processes at temperatures 0.7, 0.85, 1.0 (parallel)
- Each generates a different angle on the same input
- Seed-pro evaluates: "profound or clever?" scoring
- Ensemble vote determines the winning generation

Usage:
    from seed_swarm import CreativeSwarm
    swarm = CreativeSwarm(deepinfra_key="RhZPtvuy4cXzu02LbBSffbXeqs5Yf2IZ")
    
    result = swarm.generate(
        prompt="Design a fleet-wide failure recovery protocol",
        num_mini=3,
        include_pro=True
    )
    
    print(f"Winner: {result['winner']['text'][:100]}")
    print(f"Profound score: {result['pro_score']:.2f}")
    print(f"Votes: {result['votes']}")
"""

import time
import requests
import concurrent.futures
from typing import List, Dict, Any, Optional

DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"


class CreativeSwarm:
    """
    Creative swarm: N Seed-mini processes in parallel, one Seed-pro judge.
    
    The ensemble vote is weighted by Seed-pro's philosophical depth score.
    """
    
    def __init__(self, deepinfra_key: str):
        self.deepinfra_key = deepinfra_key
        self.headers = {
            "Authorization": f"Bearer {deepinfra_key}",
            "Content-Type": "application/json"
        }
    
    def call_seed_mini(self, prompt: str, temperature: float = 0.85) -> str:
        """Call Seed-2.0-mini for one generation."""
        payload = {
            "model": "ByteDance/Seed-2.0-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 600
        }
        response = requests.post(
            f"{DEEPINFRA_BASE}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def call_seed_pro(self, prompt: str) -> Dict[str, Any]:
        """
        Call Seed-2.0-pro for philosophical evaluation.
        
        Returns: {score: 0.0-1.0, verdict: "profound"|"clever"|"mundane", reasoning: str}
        """
        eval_prompt = f"""Evaluate this output philosophically. Is it PROFOUND (deep, original, meaningful) or merely CLEVER (smart but shallow) or MUNDANE (obvious, boring)?

Output to evaluate:
{prompt}

Respond with:
SCORE: 0.0-1.0 (1.0 = most profound)
VERDICT: profound/clever/mundane
REASONING: 1-sentence explanation"""

        payload = {
            "model": "ByteDance/Seed-2.0-pro",
            "messages": [{"role": "user", "content": eval_prompt}],
            "temperature": 0.3,
            "max_tokens": 200
        }
        response = requests.post(
            f"{DEEPINFRA_BASE}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        text = response.json()["choices"][0]["message"]["content"]
        
        # Parse the response
        score = 0.5
        verdict = "clever"
        if "SCORE:" in text:
            try:
                score = float(text.split("SCORE:")[1].split()[0])
            except Exception:
                pass
        if "PROFOUND" in text.upper():
            verdict = "profound"
        elif "MUNDANE" in text.upper():
            verdict = "mundane"
        
        return {"score": score, "verdict": verdict, "reasoning": text}
    
    def generate(
        self,
        prompt: str,
        temperatures: Optional[List[float]] = None,
        num_mini: int = 3,
        include_pro: bool = True
    ) -> Dict[str, Any]:
        """
        Run the creative swarm: parallel Seed-mini generation + optional Seed-pro judge.
        """
        if temperatures is None:
            temperatures = [0.7, 0.85, 1.0][:num_mini]
        
        # Parallel Seed-mini generation
        mini_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_mini) as executor:
            futures = [
                executor.submit(self.call_seed_mini, prompt, temp)
                for temp in temperatures
            ]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    text = future.result()
                    mini_results.append({
                        "temperature": temperatures[i],
                        "text": text,
                        "index": i
                    })
                except Exception as e:
                    mini_results.append({
                        "temperature": temperatures[i],
                        "text": f"[Error: {e}]",
                        "index": i
                    })
        
        # Score each with Seed-pro if requested
        if include_pro:
            for result in mini_results:
                try:
                    pro_eval = self.call_seed_pro(result["text"])
                    result["pro_score"] = pro_eval["score"]
                    result["verdict"] = pro_eval["verdict"]
                    result["pro_reasoning"] = pro_eval["reasoning"]
                except Exception:
                    result["pro_score"] = 0.5
                    result["verdict"] = "unknown"
        
        # Determine winner by ensemble vote
        if include_pro:
            # Weight by philosophical score
            winner = max(mini_results, key=lambda x: x.get("pro_score", 0.5))
        else:
            # Random selection from ensemble
            winner = mini_results[0]
        
        votes = {r["temperature"]: r.get("pro_score", 0.5) for r in mini_results}
        
        return {
            "prompt": prompt,
            "generations": mini_results,
            "winner": winner,
            "pro_score": winner.get("pro_score", 0.5),
            "votes": votes,
            "timestamp": time.time()
        }
    
    def generate_with_variance(
        self,
        prompt: str,
        num_generations: int = 5,
        include_pro: bool = True
    ) -> Dict[str, Any]:
        """
        Run multiple swarm cycles and return the most profound result.
        """
        results = []
        for i in range(num_generations):
            r = self.generate(prompt, num_mini=3, include_pro=include_pro)
            results.append(r)
            time.sleep(0.5)  # Rate limit
        
        # Sort by pro_score
        results.sort(key=lambda x: x["pro_score"], reverse=True)
        
        return {
            "best": results[0],
            "all_results": results,
            "most_profound": results[0]["winner"]["text"][:200]
        }