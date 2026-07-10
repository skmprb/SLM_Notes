"""
Phase 9: Prompt Management — LangChain Comparison

| Our Code                  | LangChain Equivalent                          |
|---------------------------|-----------------------------------------------|
| PromptTemplate            | ChatPromptTemplate / PromptTemplate           |
| PromptRegistry            | LangChain Hub (langchainhub)                  |
| template.render(**vars)   | prompt.format(**vars) / prompt.invoke()       |
| Load from .md files       | load_prompt() from YAML/JSON                  |
| Variable extraction {var} | input_variables auto-detection                |
| Versioning (future)       | LangSmith prompt versioning                   |
| Hot-reload from files     | No built-in (custom watcher)                  |
"""
