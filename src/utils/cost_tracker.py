"""Cost tracking and estimation utilities.

This module provides functionality for tracking API costs across
multiple providers and warning when thresholds are exceeded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Record of a single API call cost.
    
    Attributes:
        timestamp: When the API call was made
        provider: Provider name ('openai' or 'anthropic')
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Cost in USD
    """
    
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class CostTracker:
    """Track and report API costs across providers.
    
    Maintains a history of API calls and their associated costs,
    providing methods to query total costs and warn when thresholds
    are exceeded.
    
    Attributes:
        entries: List of cost entries
        warning_threshold: Threshold for cost warnings in USD
        hard_limit: Optional hard limit in USD
        
    Example:
        >>> tracker = CostTracker(warning_threshold=1.0)
        >>> tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.00024)
        >>> print(tracker.total_cost)
        0.00024
    """
    
    def __init__(
        self,
        warning_threshold: float = 1.0,
        hard_limit: float | None = None
    ) -> None:
        """Initialize cost tracker.
        
        Args:
            warning_threshold: Threshold for warnings in USD
            hard_limit: Optional hard limit in USD
        """
        self.entries: list[CostEntry] = []
        self.warning_threshold = warning_threshold
        self.hard_limit = hard_limit
        self._warnings_issued: set[float] = set()
    
    def add_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ) -> None:
        """Add a cost entry.
        
        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
        """
        entry = CostEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
        
        self.entries.append(entry)
        
        # Check for warnings
        total = self.total_cost
        if total >= self.warning_threshold and self.warning_threshold not in self._warnings_issued:
            logger.warning(
                f"⚠️  Cost warning: Total cost ${total:.4f} exceeds "
                f"warning threshold ${self.warning_threshold:.2f}"
            )
            self._warnings_issued.add(self.warning_threshold)
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost across all entries.
        
        Returns:
            Total cost in USD
        """
        return sum(entry.cost for entry in self.entries)
    
    @property
    def total_input_tokens(self) -> int:
        """Calculate total input tokens.
        
        Returns:
            Total input tokens
        """
        return sum(entry.input_tokens for entry in self.entries)
    
    @property
    def total_output_tokens(self) -> int:
        """Calculate total output tokens.
        
        Returns:
            Total output tokens
        """
        return sum(entry.output_tokens for entry in self.entries)
    
    def check_hard_limit(self) -> bool:
        """Check if hard limit has been exceeded.
        
        Returns:
            True if under limit or no limit set, False if limit exceeded
        """
        if self.hard_limit is None:
            return True
        
        return self.total_cost < self.hard_limit
    
    def get_cost_by_provider(self) -> dict[str, float]:
        """Get cost breakdown by provider.
        
        Returns:
            Dictionary mapping provider names to total costs
        """
        costs: dict[str, float] = {}
        for entry in self.entries:
            costs[entry.provider] = costs.get(entry.provider, 0.0) + entry.cost
        return costs
    
    def get_cost_by_model(self) -> dict[str, float]:
        """Get cost breakdown by model.
        
        Returns:
            Dictionary mapping model names to total costs
        """
        costs: dict[str, float] = {}
        for entry in self.entries:
            costs[entry.model] = costs.get(entry.model, 0.0) + entry.cost
        return costs
    
    def format_summary(self) -> str:
        """Format a human-readable cost summary.
        
        Returns:
            Formatted cost summary string
        """
        if not self.entries:
            return "No API calls made yet."
        
        lines = [
            "=" * 50,
            "Cost Summary",
            "=" * 50,
            f"Total Cost: ${self.total_cost:.6f}",
            f"Total Input Tokens: {self.total_input_tokens:,}",
            f"Total Output Tokens: {self.total_output_tokens:,}",
            f"Number of API Calls: {len(self.entries)}",
            ""
        ]
        
        # Cost by provider
        provider_costs = self.get_cost_by_provider()
        if provider_costs:
            lines.append("Cost by Provider:")
            for provider, cost in provider_costs.items():
                lines.append(f"  {provider}: ${cost:.6f}")
            lines.append("")
        
        # Cost by model
        model_costs = self.get_cost_by_model()
        if model_costs:
            lines.append("Cost by Model:")
            for model, cost in model_costs.items():
                lines.append(f"  {model}: ${cost:.6f}")
            lines.append("")
        
        # Warning/limit status
        if self.hard_limit:
            remaining = self.hard_limit - self.total_cost
            lines.append(f"Hard Limit: ${self.hard_limit:.2f}")
            lines.append(f"Remaining: ${remaining:.6f}")
        else:
            lines.append("Hard Limit: Not set")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset cost tracker by clearing all entries."""
        self.entries.clear()
        self._warnings_issued.clear()
        logger.info("Cost tracker reset")