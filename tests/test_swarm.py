"""Tests for Seed Creative Swarm package."""

import pytest
from unittest.mock import patch, MagicMock
import json

from seed_swarm import CreativeSwarm


@pytest.fixture
def swarm():
    """Create swarm instance with test API key."""
    return CreativeSwarm(deepinfra_key="test_key_123")


@pytest.fixture
def mock_seed_mini_response():
    """Mock successful Seed-mini response."""
    return {
        "choices": [{"message": {"content": "The fleet sails at midnight."}}]
    }


@pytest.fixture
def mock_seed_pro_profound():
    """Mock Seed-pro response: profound."""
    return {
        "choices": [{"message": {"content": "SCORE: 0.85\nVERDICT: profound\nREASONING: Deeply original perspective."}}]
    }


@pytest.fixture
def mock_seed_pro_clever():
    """Mock Seed-pro response: clever."""
    return {
        "choices": [{"message": {"content": "SCORE: 0.55\nVERDICT: clever\nREASONING: Smart but lacks depth."}}]
    }


class TestCreativeSwarm:
    """Test CreativeSwarm class."""

    def test_init(self, swarm):
        """Test swarm initialization."""
        assert swarm.deepinfra_key == "test_key_123"
        assert "Authorization" in swarm.headers
        assert "Bearer test_key_123" in swarm.headers["Authorization"]

    @patch("seed_swarm.requests.post")
    def test_call_seed_mini(self, mock_post, swarm, mock_seed_mini_response):
        """Test Seed-mini API call."""
        mock_post.return_value = MagicMock(
            json=lambda: mock_seed_mini_response,
            raise_for_status=lambda: None
        )

        result = swarm.call_seed_mini("Tell me about boats", temperature=0.8)

        assert result == "The fleet sails at midnight."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "ByteDance/Seed-2.0-mini"
        assert call_kwargs["json"]["temperature"] == 0.8

    @patch("seed_swarm.requests.post")
    def test_call_seed_pro(self, mock_post, swarm, mock_seed_pro_profound):
        """Test Seed-pro evaluation."""
        mock_post.return_value = MagicMock(
            json=lambda: mock_seed_pro_profound,
            raise_for_status=lambda: None
        )

        result = swarm.call_seed_pro("Some creative output about ships")

        assert result["score"] == 0.85
        assert result["verdict"] == "profound"
        mock_post.assert_called_once()
        assert mock_post.call_args[1]["json"]["model"] == "ByteDance/Seed-2.0-pro"

    @patch("seed_swarm.requests.post")
    def test_call_seed_pro_mundane(self, mock_post, swarm):
        """Test Seed-pro mundane verdict parsing."""
        mock_post.return_value = MagicMock(
            json=lambda: {
                "choices": [{"message": {"content": "SCORE: 0.2\nVERDICT: mundane\nREASONING: Boring."}}]
            },
            raise_for_status=lambda: None
        )

        result = swarm.call_seed_pro("Hello world")
        assert result["verdict"] == "mundane"
        assert result["score"] == 0.2

    @patch("seed_swarm.requests.post")
    def test_generate_parallel_mini(self, mock_post, swarm, mock_seed_mini_response, mock_seed_pro_profound):
        """Test parallel mini generation."""
        mock_post.side_effect = [
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_pro_profound, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_pro_profound, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_pro_profound, raise_for_status=lambda: None),
        ]

        result = swarm.generate(
            prompt="Write about oceans",
            temperatures=[0.7, 0.85, 1.0],
            num_mini=3,
            include_pro=True
        )

        assert "generations" in result
        assert len(result["generations"]) == 3
        assert "winner" in result
        assert "pro_score" in result
        assert "timestamp" in result

    @patch("seed_swarm.requests.post")
    def test_generate_without_pro(self, mock_post, swarm, mock_seed_mini_response):
        """Test generation without Seed-pro judge."""
        mock_post.side_effect = [
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
        ]

        result = swarm.generate(
            prompt="Write about oceans",
            temperatures=[0.7, 0.85, 1.0],
            num_mini=3,
            include_pro=False
        )

        assert result["pro_score"] == 0.5  # Default when no pro
        assert "generations" in result

    @patch("seed_swarm.requests.post")
    def test_generate_with_custom_temperatures(self, mock_post, swarm, mock_seed_mini_response, mock_seed_pro_profound):
        """Test generation with custom temperatures."""
        mock_post.side_effect = [
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_pro_profound, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_pro_profound, raise_for_status=lambda: None),
        ]

        result = swarm.generate(
            prompt="Write about oceans",
            temperatures=[0.6, 0.9],
            num_mini=2,
            include_pro=True
        )

        assert len(result["generations"]) == 2
        for gen in result["generations"]:
            assert gen["temperature"] in [0.6, 0.9]

    @patch("seed_swarm.requests.post")
    def test_generate_handles_api_error(self, mock_post, swarm):
        """Test graceful handling of API errors."""
        mock_post.side_effect = [
            Exception("API Error"),
            MagicMock(json=lambda: {"choices": [{"message": {"content": "Fallback"}}]}, raise_for_status=lambda: None),
            MagicMock(json=lambda: {"choices": [{"message": {"content": "Fallback"}}]}, raise_for_status=lambda: None),
        ]

        result = swarm.generate(
            prompt="Write about oceans",
            temperatures=[0.7, 0.85],
            num_mini=2,
            include_pro=True
        )

        # Should still return results, one with error
        assert len(result["generations"]) == 2

    @patch("seed_swarm.requests.post")
    def test_generate_with_variance(self, mock_post, swarm, mock_seed_mini_response, mock_seed_pro_profound):
        """Test generate_with_variance runs multiple cycles."""
        # First cycle: low score
        low_score = {
            "choices": [{"message": {"content": "Low quality."}}]
        }
        # Second cycle: high score  
        high_score = {
            "choices": [{"message": {"content": "High quality output."}}]
        }
        
        mock_post.side_effect = [
            # Cycle 1: 3 minis + 3 pros
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: low_score, raise_for_status=lambda: None),
            MagicMock(json=lambda: low_score, raise_for_status=lambda: None),
            MagicMock(json=lambda: low_score, raise_for_status=lambda: None),
            # Cycle 2: 3 minis + 3 pros
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mock_seed_mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: high_score, raise_for_status=lambda: None),
            MagicMock(json=lambda: high_score, raise_for_status=lambda: None),
            MagicMock(json=lambda: high_score, raise_for_status=lambda: None),
        ]

        result = swarm.generate_with_variance(
            prompt="Write about oceans",
            num_generations=2,
            include_pro=True
        )

        assert "best" in result
        assert "all_results" in result
        assert len(result["all_results"]) == 2
        assert "most_profound" in result

    def test_votes_dict_format(self, swarm):
        """Test votes dictionary has correct structure."""
        # Verify votes format without API call
        votes = {0.7: 0.5, 0.85: 0.5, 1.0: 0.5}
        for temp, score in votes.items():
            assert isinstance(temp, float)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


class TestIntegration:
    """Integration-style tests with mocked responses."""

    @patch("seed_swarm.requests.post")
    def test_full_swarm_cycle(self, mock_post, swarm):
        """Test complete swarm execution."""
        mini_response = {"choices": [{"message": {"content": "Generated text"}}]}
        pro_response = {"choices": [{"message": {"content": "SCORE: 0.75\nVERDICT: profound\nREASONING: Good."}}]}
        
        # Build response chain: 3 minis then 3 pros
        mock_post.side_effect = [
            MagicMock(json=lambda: mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: mini_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: pro_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: pro_response, raise_for_status=lambda: None),
            MagicMock(json=lambda: pro_response, raise_for_status=lambda: None),
        ]

        result = swarm.generate(
            prompt="Design a fleet protocol",
            num_mini=3,
            include_pro=True
        )

        assert result["prompt"] == "Design a fleet protocol"
        assert len(result["generations"]) == 3
        # Winner should be one of the generations with the mocked text
        assert result["generations"][0]["text"] == "Generated text"
        assert result["pro_score"] == 0.75
        assert len(result["votes"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])