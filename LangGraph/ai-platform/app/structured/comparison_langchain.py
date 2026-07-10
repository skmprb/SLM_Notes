"""
Phase 8: Structured Output — LangChain Comparison

Shows how our structured output system maps to LangChain's with_structured_output().
"""

# ============================================================
# 1. BASIC STRUCTURED OUTPUT (Pydantic)
# ============================================================

# --- OUR CODE ---
from pydantic import BaseModel, Field
from app.structured import OutputSchema, StructuredOutputService

class Person(BaseModel):
    """A person's information."""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    city: str = Field(description="City of residence")

# schema = OutputSchema.from_pydantic(Person)
# service = StructuredOutputService()
# result = service.generate(
#     messages=[{"role": "user", "content": "John is 30, lives in NYC"}],
#     schema=schema,
# )
# result.data  # Person(name="John", age=30, city="NYC")

# --- LANGCHAIN EQUIVALENT ---
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field
#
# class Person(BaseModel):
#     """A person's information."""
#     name: str = Field(description="Full name")
#     age: int = Field(description="Age in years")
#     city: str = Field(description="City of residence")
#
# model = ChatOpenAI(model="gpt-4o")
# structured_model = model.with_structured_output(Person)
# result = structured_model.invoke("John is 30, lives in NYC")
# result  # Person(name="John", age=30, city="NYC")


# ============================================================
# 2. STRATEGY SELECTION
# ============================================================

# --- OUR CODE ---
# # Explicit strategy
# result = service.generate(messages, schema, strategy=StructuredOutputStrategy.JSON_SCHEMA)
# result = service.generate(messages, schema, strategy=StructuredOutputStrategy.FUNCTION_CALLING)
# result = service.generate(messages, schema, strategy=StructuredOutputStrategy.JSON_MODE)
#
# # Auto-select (OpenAI → json_schema, others → function_calling)
# result = service.generate(messages, schema)

# --- LANGCHAIN EQUIVALENT ---
# model.with_structured_output(Person, method="json_schema")
# model.with_structured_output(Person, method="function_calling")
# model.with_structured_output(Person, method="json_mode")
#
# # Auto-select (default varies by provider)
# model.with_structured_output(Person)


# ============================================================
# 3. JSON SCHEMA (without Pydantic)
# ============================================================

# --- OUR CODE ---
# schema = OutputSchema.from_json_schema(
#     name="person",
#     schema={
#         "type": "object",
#         "properties": {
#             "name": {"type": "string"},
#             "age": {"type": "integer"},
#         },
#         "required": ["name", "age"],
#     },
# )
# result = service.generate(messages, schema)
# result.data  # {"name": "John", "age": 30}

# --- LANGCHAIN EQUIVALENT ---
# json_schema = {
#     "title": "Person",
#     "type": "object",
#     "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
#     "required": ["name", "age"],
# }
# model.with_structured_output(json_schema, method="json_schema")


# ============================================================
# 4. ERROR HANDLING / VALIDATION
# ============================================================

# --- OUR CODE ---
# result = service.generate(messages, schema)
# if result.success:
#     use(result.data)  # Validated Pydantic instance or dict
# else:
#     log(result.validation_error)  # "JSON parse error: ..." or "Validation error: ..."
#     fallback(result.raw_text)     # Raw LLM output for debugging

# --- LANGCHAIN EQUIVALENT ---
# # With include_raw=True:
# result = model.with_structured_output(Person, include_raw=True).invoke(msg)
# result["parsed"]         # Person instance (or None if failed)
# result["parsing_error"]  # Error details (or None if success)
# result["raw"]            # Raw AIMessage


# ============================================================
# 5. FUNCTION CALLING TRICK (How it works under the hood)
# ============================================================

# --- OUR CODE ---
# # We create a fake tool whose parameters ARE the output schema:
# tool = {
#     "type": "function",
#     "function": {
#         "name": "output_Person",
#         "description": "Return the structured output",
#         "parameters": person_json_schema,
#     },
# }
# # Then force the LLM to call it → arguments = structured data

# --- LANGCHAIN EQUIVALENT ---
# # LangChain does the same thing internally when method="function_calling":
# # It creates a tool from the schema and forces the model to call it
# # The tool_call.args become the structured output


# ============================================================
# 6. KEY DIFFERENCES
# ============================================================

# | Aspect              | Our Code                        | LangChain                      |
# |---------------------|---------------------------------|--------------------------------|
# | API                 | service.generate(msgs, schema)  | model.with_structured_output() |
# | Schema input        | OutputSchema (Pydantic or dict) | Pydantic, TypedDict, JSON      |
# | Strategy control    | Explicit enum parameter         | method= parameter              |
# | Auto-selection      | Based on provider_name          | Based on provider class        |
# | Validation          | ParsedOutput.success + error    | include_raw=True               |
# | Provider-agnostic   | Yes (same service, any provider)| Yes (same API, any model)      |
# | Retry on failure    | Manual (can add)                | Not built-in                   |

# Our approach gives us:
# ✅ Explicit strategy control (know exactly what's happening)
# ✅ Unified error handling (success/error in one response)
# ✅ Provider-agnostic (same code for OpenAI, Anthropic, Gemini)
# ✅ Easy to add retry-with-repair (re-prompt on validation failure)
# ✅ Understanding of what with_structured_output() does internally
