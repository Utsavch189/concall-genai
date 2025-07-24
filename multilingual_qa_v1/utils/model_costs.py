def cost_gemini_25_flash(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for Gemini 2.5 Flash model.
    :param input_tokens: Number of input tokens.
    :param output_tokens: Number of output tokens.
    :return: Total cost in USD.
    """
    input_cost_per_million = 0.30   # USD per 1M input tokens
    output_cost_per_million = 2.50  # USD per 1M output tokens
    return (input_tokens / 1_000_000) * input_cost_per_million + \
           (output_tokens / 1_000_000) * output_cost_per_million


def cost_gpt4o(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for GPT-4o model.
    :param input_tokens: Number of input tokens.
    :param output_tokens: Number of output tokens.
    :return: Total cost in USD.
    """
    input_cost_per_million = 2.50   # USD per 1M input tokens
    output_cost_per_million = 10.00 # USD per 1M output tokens
    return (input_tokens / 1_000_000) * input_cost_per_million + \
           (output_tokens / 1_000_000) * output_cost_per_million