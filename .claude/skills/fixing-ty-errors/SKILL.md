---
name: fixing-ty-errors
description: Fix ty type-check failures. Use when make lint-typing or uv run ty check reports errors, or when annotating code to satisfy ty.
---

# Add Or Fix Type Checking - ty type checker

## Input

- `<target>`: module or directory to type-check (if known).
- Optional `make lint-typing`, `uv run ty check ...`, or CI output showing
  typing failures.

## Workflow

1. **Identify scope from the failing run**:
   - If you already have `make lint-typing`, `uv run ty check ...`, or CI
     output, extract the failing file/module paths.
   - If not, run:
     ```bash
     make lint-typing
     ```
   - Choose the narrowest target that covers the failures.

2. **Run `ty check` for the target** to get a focused baseline:
   ```bash
   uv run ty check <target>
   ```
   Use `uv run ty check am_diag/` when no narrower target is known.

3. **Triage errors by category** before fixing anything:
   - Wrong/missing type annotations on signatures
   - Attribute access on union types (for example `X | None`)
   - Functions returning broad unions (for example `str | list | BatchEncoding`)
   - Mixin/protocol self-type issues
   - Dynamic attributes on objects or modules
   - Third-party stub gaps (missing kwargs, missing `__version__`, etc.)

4. **Apply fixes using this priority order** (simplest first):

   a. **Narrow unions with `isinstance()` / `if x is None` / `hasattr()`**.
      This is the primary tool for resolving union-type errors. `ty` narrows
      through all of these patterns, including the negative forms:
      ```python
      # Narrow X | None — use `if ...: raise`, never `assert`
      if x is None:
          raise ValueError("x must not be None")
      x.method()  # ty knows x is X here

      # Narrow str | UploadFile
      if isinstance(field, str):
          raise TypeError("Expected file upload, got string")
      await field.read()  # ty knows field is UploadFile here

      # Narrow broad union parameters early in a function body
      # (common for methods accepting e.g. list | dict | BatchEncoding)
      if isinstance(encoded_inputs, (list, tuple)):
          raise TypeError("Expected a mapping, got sequence")
      encoded_inputs.keys()  # ty sees only the dict/mapping types now
      ```

   b. **Use local variables to help ty track narrowing across closures**.
      When `self.x` is `X | None` and you need to pass it to nested functions
      or closures, `ty` cannot track that `self.x` stays non-None. Copy to a
      local variable and narrow the local:
      ```python
      manager = self.batching_manager
      if manager is None:
          raise RuntimeError("Manager not initialized")
      # Use `manager` (not `self.batching_manager`) in nested functions
      ```

   c. **Split chained calls when the intermediate type is a broad union**.
      If `func().method()` fails because `func()` returns a union, split it:
      ```python
      # BAD: ty can't narrow through chained calls
      result = func(return_dict=True).to(device)["input_ids"]

      # GOOD: split, narrow to the actual type, then chain
      result = func(return_dict=True)
      if not isinstance(result, BatchEncoding):
          raise TypeError("Expected BatchEncoding")
      inputs = result.to(device)["input_ids"]
      ```

   d. **Fix incorrect type hints at the source**. If a parameter is typed `X | None`
      but can never be `None` when actually called, remove `None` from the hint.

   e. **Annotate untyped attributes**. Add type annotations to instance variables
      set in `__init__` or elsewhere (for example `self.foo: list[int] = []`).
      Declare class-level attributes that are set dynamically later
      (for example `_cache: Cache`, `_token_tensor: torch.Tensor | None`).

   f. **Use `@overload` for methods with input-dependent return types**.
      When a method returns different types based on the input type (e.g.
      `__getitem__` with str vs int keys), use `@overload` to declare each
      signature separately:
      ```python
      from typing import overload

      @overload
      def __getitem__(self, item: str) -> ValueType: ...
      @overload
      def __getitem__(self, item: int) -> EncodingType: ...
      @overload
      def __getitem__(self, item: slice) -> dict[str, ValueType]: ...

      def __getitem__(self, item: int | str | slice) -> ValueType | EncodingType | dict[str, ValueType]:
          ...  # actual implementation
      ```
      This eliminates `cast()` calls at usage sites by giving the checker
      precise return types for each call pattern.

   g. **Make container classes generic to propagate value types**.
      When a class like `UserDict` holds values whose type changes after
      transformation (e.g. lists → tensors after `.to()`), specialize the base
      class with a type variable so methods can return narrowed types:
      ```python
      from collections import UserDict
      from typing import Any
      from typing_extensions import TypeVar

      _V = TypeVar("_V", default=Any)  # default=Any keeps existing code working

      class MyDict(UserDict[str, _V]):
          def to(self, device) -> "MyDict[torch.Tensor]":
              # after .to(), values are tensors; to_tensor is the project's
              # own typed conversion function
              return MyDict({key: to_tensor(value, device) for key, value in self.items()})
      ```
      The `default=Any` (from `typing_extensions`) means unparameterized usage
      like `MyDict()` stays `MyDict[Any]` — no existing code needs to change.
      Only methods that narrow the value type (like `.to()`) declare a specific
      return type, and they build and return a converted instance instead of
      returning `self`. This eliminates `cast()` at all call sites.

   h. **Use `self: "ProtocolType"` for mixins**. When a mixin accesses attributes
      from its host class, define a local Protocol in the nearest appropriate
      project module and annotate `self` on methods that need it. Apply this
      consistently to all methods in the mixin. Import under `TYPE_CHECKING` to
      avoid circular imports.

   i. **Use a typed accessor for dynamic module attributes** (for example
      `torch.npu`, `torch.xpu`, `torch.compiler`). Instead of repeating
      `getattr(torch, "npu")` or `hasattr(torch, "npu") and torch.npu.is_available()`
      at call sites, define one accessor in the nearest appropriate project
      typing/helper module. It returns the backend as a small `Protocol` type, or
      `None` when the backend is missing:
      ```python
      class NpuBackend(Protocol):
          def is_available(self) -> bool: ...
          def device_count(self) -> int: ...

      def get_torch_npu(mod: ModuleType) -> NpuBackend | None:
          backend = getattr(mod, "npu", None)
          if backend is None or not backend.is_available():
              return None
          return cast(NpuBackend, backend)
      ```
      Callers narrow the returned value: `npu = get_torch_npu(torch)`, then
      `if npu is not None: npu.device_count()`. `ty` checks every attribute used on
      `npu` against the Protocol.

      **Key rules for accessors**:
      - Declare only the attributes the project uses on the Protocol.
      - Do not use `TypeGuard[Any]` for this. Inside the guarded branch it makes the
        whole module `Any`, so `ty` stops checking even the module's known
        attributes.
      - Do the `getattr()` and `cast()` once, inside the accessor, after the check.
        Call sites stay free of both.

   j. **Use `getattr()` / `setattr()` for dynamic model/config attributes**.
      For runtime-injected fields (for example config/model flags), use
      `getattr(obj, "field", default)` for reads and `setattr(obj, "field", value)`
      for writes. Also use `getattr()` for third-party packages missing type stubs
      (for example `getattr(safetensors, "__version__", "unknown")`).
      Avoid `getattr(torch, "npu")` at call sites — use a typed accessor instead
      (see above).

   k. **Use `cast()` as a last resort before `# ty: ignore`**.
      Use when you've structurally validated the type but the checker can't see it:
      pattern-matched AST nodes, known-typed dict values, or validated API responses.
      ```python
      # After structural validation confirms the type:
      stmt = cast(cst.Assign, node.body[0])
      annotations = cast(list[Annotation], [])
      ```
      Do not use `cast()` for module attribute narrowing at call sites — use a
      typed accessor.
      Do not use `cast()` when `@overload` or generics can solve it at the source.

   l. **Use `# ty: ignore[<rule>]` only for third-party stub defects**. This means
      cases where the third-party package's type stubs are wrong or incomplete
      and there is no way to narrow or cast around it. Examples:
      - A kwarg that exists at runtime but is missing from the stubs
      - A method that exists but isn't declared in the stubs
      Always name the `ty` rule, for example `# ty: ignore[unknown-argument]`, not a
      bare `# ty: ignore`. Do not write mypy-style codes such as
      `# type: ignore[call-arg]`. `ty` does not read them, so the diagnostic is
      not suppressed.

5. **Things to never do**:
   - **Never use `assert` for type narrowing.** Asserts are stripped by `python -O`
     and must not be relied on for correctness. Use `if ...: raise` instead.
   - **Never use `# ty: ignore` as a first resort.** Exhaust all approaches above first.
   - Do not use `getattr(torch, "backend")` to access dynamic device backends
     (`npu`, `xpu`, `hpu`, `musa`, `mlu`, `neuron`, `compiler`) — use a typed
     accessor
   - Do not use `cast()` for module attribute narrowing at call sites — use a
     typed accessor
   - Do not use `cast()` when `@overload` or generics can eliminate it at the source
   - Do not add helper methods or abstractions just to satisfy the type checker
     (especially for only 1-2 occurrences)
   - Do not pollute base classes with domain-specific fields; use Protocols
   - Do not add `if x is not None` guards for values guaranteed non-None
     by the call chain; fix the annotation instead
   - Do not use conditional inheritance patterns; annotate `self` instead

6. **Organization**:
   - Keep shared Protocols and type aliases near the code that owns the concept,
     or in a small project typing helper when reuse justifies it.
   - Import type-only symbols under `if TYPE_CHECKING:` to avoid circular deps
   - Use `from __future__ import annotations` for PEP 604 syntax (`X | Y`)

7. **Verify and close the PR loop**:
   - Re-run `uv run ty check <target>` on the same target.
   - Re-run `make lint-typing` to confirm the repository typing target passes.
   - If working toward merge readiness, run `make lint-all` and relevant tests.
   - Ensure runtime behavior did not change and run relevant tests
