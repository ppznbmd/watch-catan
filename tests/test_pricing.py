"""The dollar figure every script prints comes from `arena.play.usd`."""

import pytest

from arena.play import usd


def test_an_uncached_gpt_5_6_prompt_is_billed_as_a_cache_write():
    """What OpenAI charged on 2026-09-18: every uncached luna token at $0.25/M,
    not the $0.20/M list rate. Pricing it at list understates a match by 12%."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0,
             "output_tokens": 0, "cache_write_input_tokens": 1_000_000}
    assert usd(usage, "gpt-5.6-luna") == pytest.approx(0.25)


def test_a_log_from_before_writes_were_recorded_is_priced_as_fully_written():
    """The v3 baseline carries no write count. It was played with writes on, so
    re-pricing it at list would reproduce the understatement."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}
    assert usd(usage, "gpt-5.6-luna") == pytest.approx(0.25)


def test_a_reported_zero_writes_is_billed_at_the_plain_input_rate():
    """With prompt_cache_options explicit and no breakpoint, nothing is written;
    a report that still charged 1.25x would hide the saving."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0,
             "output_tokens": 0, "cache_write_input_tokens": 0}
    assert usd(usage, "gpt-5.6-luna") == pytest.approx(0.20)


def test_cached_tokens_are_never_charged_as_writes():
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 600_000, "output_tokens": 0}
    assert usd(usage, "gpt-5.6-luna") == pytest.approx(0.4 * 0.25 + 0.6 * 0.02)


def test_a_provider_without_a_write_fee_is_unchanged():
    """DeepSeek caches without charging for the write; applying the GPT-5.6 rule
    to it would overprice every DeepSeek run."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 1_000_000}
    assert usd(usage, "deepseek-flash") == pytest.approx(0.30 + 1.20)


def test_a_model_with_no_price_on_file_gets_no_dollar_figure():
    assert usd({"input_tokens": 10, "output_tokens": 10}, "some-new-model") is None


def test_an_answer_served_at_flex_is_billed_at_half():
    """The GPT-5.6 flex rates are half the standard ones on every line."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 1_000_000,
             "cache_write_input_tokens": 0, "service_tier": "flex"}
    assert usd(usage, "gpt-5.6-sol") == pytest.approx((4.00 + 20.00) / 2)


def test_an_answer_without_a_reported_tier_is_billed_at_the_standard_rate():
    """Asking for flex is not getting it. Pricing an unreported tier at half would
    understate the bill whenever the provider served the request normally."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0,
             "cache_write_input_tokens": 0}
    assert usd(usage, "gpt-5.6-sol") == pytest.approx(4.00)
    assert usd({**usage, "service_tier": "default"}, "gpt-5.6-sol") == pytest.approx(4.00)


def epoch(text):
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def test_a_deepseek_request_made_off_peak_is_billed_at_half():
    """DeepSeek halves every rate outside its weekday peak hours. The loss probe
    ran entirely off-peak and was reported at $1.76 for a $0.88 bill."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 1_000_000,
             "requested_at": epoch("2026-09-24T00:30:00+00:00")}  # Thursday, before 01:00
    assert usd(usage, "deepseek-flash") == pytest.approx((0.30 + 1.20) / 2)


@pytest.mark.parametrize("at", ["2026-09-24T01:00:00+00:00", "2026-09-24T09:59:00+00:00"])
def test_a_deepseek_request_made_at_peak_is_billed_in_full(at):
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0,
             "requested_at": epoch(at)}
    assert usd(usage, "deepseek-flash") == pytest.approx(0.30)


def test_weekend_hours_are_off_peak_all_day():
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0,
             "requested_at": epoch("2026-09-26T02:00:00+00:00")}  # Saturday
    assert usd(usage, "deepseek-flash") == pytest.approx(0.15)


def test_a_record_without_its_request_time_is_priced_at_peak():
    """Unknown hour, so the most it can have cost: a report must not claim a
    discount nobody can show was given."""
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}
    assert usd(usage, "deepseek-flash") == pytest.approx(0.30)


def test_off_peak_hours_never_discount_an_openai_model():
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0,
             "cache_write_input_tokens": 0, "requested_at": epoch("2026-09-26T02:00:00+00:00")}
    assert usd(usage, "gpt-5.6-luna") == pytest.approx(0.20)
