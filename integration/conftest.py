"""
Integration test configuration.

Enables LangChain debug mode so every LLM prompt and response is printed
to stdout during the test run. Use `pytest -s` (already set in the Makefile
target) to prevent pytest from capturing this output.
"""
from langchain.globals import set_debug

set_debug(True)
