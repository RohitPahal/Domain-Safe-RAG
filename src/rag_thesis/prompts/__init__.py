"""Versioned, provider-neutral prompts for the four LLM roles.

Each module builds the messages for one role as plain ``ChatMessage`` objects and names its
``VERSION``. The text is exactly what the thesis runs used. Rule: changing a prompt's text means
bumping its version (the version is part of the cache keys, so old cached outputs are never
served for a new prompt); ``tests/unit/test_prompts.py`` pins each text to its version.
"""
