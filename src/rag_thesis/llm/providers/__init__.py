"""Provider adapters: the only modules that know an LLM SDK.

Each adapter implements ``ChatModel.complete`` for one API family: it maps a ``ChatRequest`` to the
SDK call, the SDK response back to a ``ChatResponse``, and SDK exceptions to
``rag_thesis.llm.errors``. SDKs are imported lazily, so an adapter's SDK is only needed when that
provider is actually used. Adding a provider = one new module here + one registry entry + passing
``tests/unit/test_llm_providers.py``.
"""
