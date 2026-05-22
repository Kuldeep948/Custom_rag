"""
Source code processing service.
Uses Python's built-in AST for Python files to create semantically meaningful chunks
(functions, classes, methods) rather than arbitrary text splits.
"""
import ast
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.logging import get_logger
from app.services.ingestion.chunker import TextChunk, chunk_text, estimate_tokens

logger = get_logger(__name__)


class CodeProcessor:
    """
    Processes source code files into semantically meaningful chunks.

    For Python files: uses AST to extract functions, classes, and methods.
    For other languages: falls back to line-based chunking.
    """

    PYTHON_EXTENSIONS = {".py", ".pyw"}
    SUPPORTED_EXTENSIONS = {
        ".py", ".pyw", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".cpp", ".c", ".h", ".go", ".rs",
        ".rb", ".php", ".cs", ".swift", ".kt",
    }

    def process(self, file_path: str) -> Tuple[List[TextChunk], dict]:
        """
        Process a source code file and return chunks + metadata.

        Args:
            file_path: Path to the source code file

        Returns:
            Tuple of (chunks, metadata_dict)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Code file not found: {file_path}")

        extension = path.suffix.lower()
        source_code = path.read_text(encoding="utf-8", errors="replace")

        logger.info(
            "code_processing_start",
            file=path.name,
            extension=extension,
            size=len(source_code),
        )

        metadata = {
            "language": self._detect_language(extension),
            "extension": extension,
            "line_count": source_code.count("\n") + 1,
            "file_size": len(source_code),
        }

        if extension in self.PYTHON_EXTENSIONS:
            chunks = self._process_python(source_code, path.name)
            metadata["extraction_method"] = "ast_chunking"
        else:
            chunks = self._process_generic(source_code)
            metadata["extraction_method"] = "text_chunking"

        metadata["chunk_count"] = len(chunks)

        logger.info(
            "code_processing_complete",
            file=path.name,
            chunks=len(chunks),
            language=metadata["language"],
        )

        return chunks, metadata

    def _process_python(self, source_code: str, filename: str) -> List[TextChunk]:
        """
        AST-aware chunking for Python files.
        Extracts module docstring, imports, functions, and classes as separate chunks.
        """
        chunks: List[TextChunk] = []
        lines = source_code.splitlines()

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.warning("python_ast_parse_failed", error=str(e), fallback="line_based")
            return self._process_generic(source_code)

        chunk_index = 0

        # 1. Extract module-level docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.append(
                TextChunk(
                    content=f"# Module Documentation\n\n{module_docstring}",
                    chunk_index=chunk_index,
                    token_count=estimate_tokens(module_docstring),
                    start_line=1,
                    end_line=1,
                    metadata={"chunk_type": "module_docstring"},
                )
            )
            chunk_index += 1

        # 2. Extract imports as a single chunk
        import_lines = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = node.end_lineno
                import_lines.extend(lines[start:end])

        if import_lines:
            import_text = "\n".join(import_lines)
            chunks.append(
                TextChunk(
                    content=f"# Imports\n\n{import_text}",
                    chunk_index=chunk_index,
                    token_count=estimate_tokens(import_text),
                    metadata={"chunk_type": "imports"},
                )
            )
            chunk_index += 1

        # 3. Extract top-level functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function_chunk(node, lines, chunk_index)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

            elif isinstance(node, ast.ClassDef):
                class_chunks = self._extract_class_chunks(node, lines, chunk_index)
                for c in class_chunks:
                    c.chunk_index = chunk_index
                    chunks.append(c)
                    chunk_index += 1

        # 4. If no structured chunks found, fall back to text chunking
        if len(chunks) <= 2:
            logger.debug("python_ast_no_structure_found", fallback="text_chunking")
            return self._process_generic(source_code)

        return chunks

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        chunk_index: int,
    ) -> Optional[TextChunk]:
        """Extract a function as a single chunk."""
        start_line = node.lineno - 1
        end_line = node.end_lineno
        func_lines = lines[start_line:end_line]
        content = "\n".join(func_lines)

        if not content.strip():
            return None

        # If function is very long, include a summary header
        docstring = ast.get_docstring(node)
        if len(content) > 4000 and docstring:
            # For very long functions, create a summary chunk
            summary = (
                f"def {node.name}(...):\n"
                f'    """{docstring}"""\n'
                f"    # ... ({end_line - start_line} lines)"
            )
            content = summary

        return TextChunk(
            content=content,
            chunk_index=chunk_index,
            token_count=estimate_tokens(content),
            start_line=node.lineno,
            end_line=node.end_lineno,
            function_name=node.name,
            metadata={
                "chunk_type": "function",
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "has_docstring": docstring is not None,
            },
        )

    def _extract_class_chunks(
        self,
        node: ast.ClassDef,
        lines: List[str],
        start_index: int,
    ) -> List[TextChunk]:
        """Extract a class and its methods as separate chunks."""
        chunks = []

        # Class header + docstring
        class_start = node.lineno - 1
        class_end = node.end_lineno
        class_lines = lines[class_start:class_end]
        class_content = "\n".join(class_lines)

        # Class overview chunk (header + docstring + class variables)
        docstring = ast.get_docstring(node)
        overview_lines = []
        for line in class_lines[:20]:  # First 20 lines for overview
            overview_lines.append(line)
        overview = "\n".join(overview_lines)

        chunks.append(
            TextChunk(
                content=overview,
                chunk_index=start_index,
                token_count=estimate_tokens(overview),
                start_line=node.lineno,
                end_line=min(node.lineno + 20, node.end_lineno),
                class_name=node.name,
                metadata={
                    "chunk_type": "class_overview",
                    "class_name": node.name,
                    "has_docstring": docstring is not None,
                },
            )
        )

        # Individual methods
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_start = child.lineno - 1
                method_end = child.end_lineno
                method_lines = lines[method_start:method_end]
                method_content = "\n".join(method_lines)

                if method_content.strip():
                    chunks.append(
                        TextChunk(
                            content=method_content,
                            chunk_index=start_index + len(chunks),
                            token_count=estimate_tokens(method_content),
                            start_line=child.lineno,
                            end_line=child.end_lineno,
                            function_name=child.name,
                            class_name=node.name,
                            metadata={
                                "chunk_type": "method",
                                "class_name": node.name,
                                "method_name": child.name,
                                "is_async": isinstance(child, ast.AsyncFunctionDef),
                            },
                        )
                    )

        return chunks

    def _process_generic(self, source_code: str) -> List[TextChunk]:
        """
        Line-based chunking for non-Python files or fallback.
        Groups lines into chunks respecting the configured chunk size.
        """
        return chunk_text(source_code)

    def _detect_language(self, extension: str) -> str:
        """Map file extension to language name."""
        mapping = {
            ".py": "python", ".pyw": "python",
            ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp", ".c": "c", ".h": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        return mapping.get(extension.lower(), "unknown")
