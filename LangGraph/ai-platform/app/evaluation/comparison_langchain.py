"""
Phase 15: Evaluation Framework — LangChain Comparison

| Our Code                  | LangChain Equivalent                          |
|---------------------------|-----------------------------------------------|
| TestCase                  | LangSmith Dataset examples                    |
| EvaluationRunner          | LangSmith evaluate() / run_on_dataset()       |
| ContainsEvaluator         | CriteriaEvalChain / custom evaluator          |
| LatencyEvaluator          | Custom RunEvaluator                           |
| EvalSuiteResult           | LangSmith experiment results                  |
| LLM-as-judge (future)     | LangChain LLM evaluators (correctness, etc.)  |
| SAMPLE_TEST_SUITE         | LangSmith golden datasets                     |
| pass_rate                 | LangSmith experiment metrics                  |
| RAGAS metrics (future)    | ragas library (faithfulness, relevance)        |
"""
