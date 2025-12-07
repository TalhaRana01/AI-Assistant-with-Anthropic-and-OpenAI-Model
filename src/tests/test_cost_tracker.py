"""Tests for cost tracking functionality.

This module tests the CostTracker class including cost calculation,
threshold warnings, and reporting.
"""

from __future__ import annotations

import pytest

from src.utils.cost_tracker import CostTracker


def test_cost_tracker_initialization():
    """Test cost tracker initialization."""
    tracker = CostTracker(warning_threshold=1.0, hard_limit=5.0)
    
    assert tracker.warning_threshold == 1.0
    assert tracker.hard_limit == 5.0
    assert tracker.total_cost == 0.0
    assert len(tracker.entries) == 0


def test_add_cost():
    """Test adding cost entries."""
    tracker = CostTracker()
    
    tracker.add_cost(
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        cost=0.00024
    )
    
    assert len(tracker.entries) == 1
    assert tracker.total_cost == 0.00024
    assert tracker.total_input_tokens == 100
    assert tracker.total_output_tokens == 50


def test_multiple_cost_entries():
    """Test tracking multiple API calls."""
    tracker = CostTracker()
    
    # First call
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.00024)
    
    # Second call
    tracker.add_cost("anthropic", "claude-3-5-haiku-20241022", 200, 100, 0.00048)
    
    assert len(tracker.entries) == 2
    assert tracker.total_cost == pytest.approx(0.00072)
    assert tracker.total_input_tokens == 300
    assert tracker.total_output_tokens == 150


def test_cost_by_provider():
    """Test cost breakdown by provider."""
    tracker = CostTracker()
    
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.0001)
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.0001)
    tracker.add_cost("anthropic", "claude-3-5-haiku-20241022", 100, 50, 0.0002)
    
    costs = tracker.get_cost_by_provider()
    
    assert costs["openai"] == pytest.approx(0.0002)
    assert costs["anthropic"] == pytest.approx(0.0002)


def test_cost_by_model():
    """Test cost breakdown by model."""
    tracker = CostTracker()
    
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.0001)
    tracker.add_cost("openai", "gpt-4o", 100, 50, 0.0005)
    tracker.add_cost("anthropic", "claude-3-5-haiku-20241022", 100, 50, 0.0002)
    
    costs = tracker.get_cost_by_model()
    
    assert costs["gpt-4o-mini"] == pytest.approx(0.0001)
    assert costs["gpt-4o"] == pytest.approx(0.0005)
    assert costs["claude-3-5-haiku-20241022"] == pytest.approx(0.0002)


def test_hard_limit_check():
    """Test hard limit enforcement."""
    tracker = CostTracker(hard_limit=1.0)
    
    # Under limit
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.5)
    assert tracker.check_hard_limit() is True
    
    # At limit (should still be under since < is used)
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.5)
    assert tracker.check_hard_limit() is False
    
    # Over limit
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.1)
    assert tracker.check_hard_limit() is False


def test_no_hard_limit():
    """Test that no hard limit allows unlimited spending."""
    tracker = CostTracker(hard_limit=None)
    
    # Add large cost
    tracker.add_cost("openai", "gpt-4o-mini", 1000000, 1000000, 1000.0)
    
    # Should always return True when no limit set
    assert tracker.check_hard_limit() is True


def test_warning_threshold(caplog):
    """Test that warnings are issued at threshold."""
    tracker = CostTracker(warning_threshold=1.0)
    
    # Should not warn yet
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.5)
    assert "warning" not in caplog.text.lower()
    
    # Should warn when crossing threshold
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.6)
    assert "warning" in caplog.text.lower()


def test_format_summary():
    """Test formatted summary output."""
    tracker = CostTracker()
    
    # Empty tracker
    summary = tracker.format_summary()
    assert "No API calls" in summary
    
    # With data
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.00024)
    summary = tracker.format_summary()
    
    assert "Total Cost" in summary
    assert "0.000240" in summary
    assert "gpt-4o-mini" in summary
    assert "openai" in summary


def test_reset():
    """Test tracker reset functionality."""
    tracker = CostTracker()
    
    # Add some data
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.00024)
    tracker.add_cost("anthropic", "claude-3-5-haiku-20241022", 100, 50, 0.00048)
    
    assert len(tracker.entries) == 2
    assert tracker.total_cost > 0
    
    # Reset
    tracker.reset()
    
    assert len(tracker.entries) == 0
    assert tracker.total_cost == 0.0
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0


def test_entry_attributes():
    """Test that cost entries have correct attributes."""
    tracker = CostTracker()
    
    tracker.add_cost("openai", "gpt-4o-mini", 100, 50, 0.00024)
    
    entry = tracker.entries[0]
    
    assert entry.provider == "openai"
    assert entry.model == "gpt-4o-mini"
    assert entry.input_tokens == 100
    assert entry.output_tokens == 50
    assert entry.cost == 0.00024
    assert entry.timestamp is not None


def test_cost_accumulation():
    """Test that costs accumulate correctly over many calls."""
    tracker = CostTracker()
    
    expected_total = 0.0
    for i in range(10):
        cost = 0.0001 * (i + 1)
        tracker.add_cost("openai", "gpt-4o-mini", 100, 50, cost)
        expected_total += cost
    
    assert len(tracker.entries) == 10
    assert tracker.total_cost == pytest.approx(expected_total)