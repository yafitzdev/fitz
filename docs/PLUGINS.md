# Extension Points

fitz-sage intentionally has a small extension surface. It does not expose
generic chunker, source, or data-cleanup plugins. Those concerns are part of
the user's document-preparation boundary or the engine's typed ingestion
pipeline.

## Supported Configuration

The public configuration surface selects built-in implementations:

| Concern | Configuration | Implementations |
| --- | --- | --- |
| Document parser | `parser` | `cpu`, `docling`, `docling_vision`, `glm_ocr` |
| Reranker | `rerank` | `onnx` or a compatible `onnx/<model-id>` repository |
| Governance | `governance` | `pyrrho`, `pyrrho/<local-path>`, or `pyrrho/<owner/repo@commit>` |
| Optional chat roles | role-specific provider spec | `endpoint`, `openai`, `azure_openai`, `enterprise` |
| Optional vision | `vision` | the same OpenAI-compatible endpoint protocol |

Parser modes are selected explicitly. Files are routed to built-in parsers by
extension; adding a Python file to a directory does not auto-register a new
parser.

An alternate reranker repository must contain a compatible tokenizer and
`onnx/model_int8.onnx`. Other artifact layouts are available only through
direct low-level `OnnxReranker` construction, not engine YAML.

## Chat And Vision Providers

Optional chat and vision roles use the OpenAI-compatible protocol. The
`glm_ocr` parser is a separate native Ollama integration. Most custom chat
deployments should use `endpoint/<model>` with their own URL:

```yaml
synthesizer: endpoint/qwen2.5-14b-instruct
chat_base_url: http://localhost:8080/v1
```

Authentication, enterprise OAuth, and preset providers are documented in
[Configuration](CONFIG.md). A new provider name is a package contribution, not
a runtime plugin: implement the relevant protocol under
`fitz_sage/llm/providers/`, register it in `fitz_sage/llm/config.py`, and add
dispatch tests.

## Custom Engines

Domain-specific retrieval belongs in a `KnowledgeEngine` implementation. A
custom engine can define its own ingestion, retrieval, and configuration while
returning the shared `Answer` and `EvidencePack` contracts. Register engine
factories through the in-process runtime registry; see
[Custom Engines](CUSTOM_ENGINES.md).

The installed package does not discover third-party Python entry points.
Automatic discovery only scans engine modules bundled under
`fitz_sage/engines/`. An application-owned engine must import its registration
module before calling `run()` or `create_engine()`.

## Deliberate Non-Extensions

fitz-sage does not silently normalize identifiers, expand organization-specific
abbreviations, rewrite raw logs, or clean source data. Applications may perform
that work before calling fitz-sage. There is no public vocabulary-mapping API.

The KRAG engine also does not expose its typed-unit extraction as a chunker
plugin. Code symbols, document sections, and table specifications are internal
retrieval units whose contracts evolve with the engine.

## See Also

- [Configuration](CONFIG.md)
- [Architecture](ARCHITECTURE.md)
- [Ingestion](INGESTION.md)
- [OpenAI-compatible endpoint](features/platform/openai-compatible-endpoint.md)
- [Enterprise gateway](features/platform/enterprise-gateway.md)
