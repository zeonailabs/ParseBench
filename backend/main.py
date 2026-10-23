import os
import time
from dotenv import load_dotenv
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.benchmark_metrics import get_benchmark_summary
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
app = FastAPI(
    title="Document Parser Benchmark API",
    description="Upload documents and benchmark parser output and latency.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)



MAX_FILE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# These are the initial formats accepted by the application.
# Actual support can vary by parser version and document content.
PARSERS = [
    {
        "id": "liteparse",
        "name": "LiteParse",
        "type": "local",
        "supported_formats": [
            ".pdf", ".docx", ".xlsx", ".pptx",
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp",
        ],
    },
    {
        "id": "docling",
        "name": "Docling",
        "type": "local",
        "supported_formats": [
            ".pdf", ".docx", ".xlsx", ".pptx",
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".html",
        ],
    },
    {
        "id": "llamaparse",
        "name": "LlamaParse",
        "type": "cloud",
        "supported_formats": [
            ".pdf", ".docx", ".xlsx", ".pptx",
            ".png", ".jpg", ".jpeg", ".tif", ".tiff",
        ],
    },
    {
        "id": "azure",
        "name": "Azure Document Intelligence",
        "type": "cloud",
        "supported_formats": [
            ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
        ],
    },
]

PARSER_BY_ID = {parser["id"]: parser for parser in PARSERS}


@app.get("/")
def root():
    return {
        "message": "Document Parser Benchmark API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "document-parser-api"}


@app.get("/api/benchmark/summary")
def benchmark_summary():
    return get_benchmark_summary()


@app.get("/api/parsers")
def list_parsers():
    return {"parsers": PARSERS}


def parse_with_liteparse(file_path: Path) -> str:
    """Use the same LiteParse CLI approach as the existing full benchmark."""
    with tempfile.TemporaryDirectory(prefix="parser_output_") as temp_dir:
        output_path = Path(temp_dir) / "parsed.md"

        completed = subprocess.run(
            [
                "lit", "parse", str(file_path),
                "--format", "markdown",
                "-o", str(output_path),
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"LiteParse exited with code {completed.returncode}")

        if not output_path.is_file():
            raise RuntimeError("LiteParse completed without producing an output file.")

        return output_path.read_text(encoding="utf-8")


def parse_with_docling(file_path: Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


def parse_with_llamaparse(file_path: Path) -> str:
    from llama_cloud import LlamaCloud

    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        raise RuntimeError("LlamaParse is not configured on the server.")

    client = LlamaCloud()
    uploaded = client.files.create(file=str(file_path), purpose="parse")

    result = client.parsing.parse(
        file_id=uploaded.id,
        tier="fast",
        version="latest",
        expand=["markdown"],
    )

    markdown_result = result.markdown
    pages = getattr(markdown_result, "pages", None) or []

    page_texts = [
        page.markdown
        for page in pages
        if getattr(page, "success", True)
        and getattr(page, "markdown", None)
    ]

    if not page_texts:
        raise RuntimeError("LlamaParse returned no successful Markdown pages.")

    return "\n\n".join(page_texts)


def parse_with_azure(file_path: Path) -> str:
    from parsers.azure_parser import parse_document

    return parse_document(file_path)


PARSER_FUNCTIONS = {
    "liteparse": parse_with_liteparse,
    "docling": parse_with_docling,
    "llamaparse": parse_with_llamaparse,
    "azure": parse_with_azure,
}


@app.post("/api/parse/{parser_id}")
async def parse_document(
    parser_id: str,
    file: UploadFile = File(...),
):
    parser = PARSER_BY_ID.get(parser_id)

    if parser is None:
        raise HTTPException(status_code=404, detail="Unknown parser.")

    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()

    if not extension or extension not in parser["supported_formats"]:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{parser['name']} is not configured to accept "
                f"the {extension or 'unknown'} file format."
            ),
        )

    temp_path = None

    try:
        # Stream to disk with a hard size limit instead of loading an
        # arbitrarily large upload into memory.
        total_bytes = 0

        with tempfile.NamedTemporaryFile(
            prefix="document_upload_",
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 20 MB upload limit.",
                    )

                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        started = time.perf_counter()

        try:
            markdown = PARSER_FUNCTIONS[parser_id](temp_path)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Parser timed out while processing the document.",
            )
        except HTTPException:
            raise
        except Exception as exc:
            # Return a useful error without exposing server paths or secrets.
            message = str(exc).replace(str(temp_path), "[temporary upload]")
            raise HTTPException(
                status_code=502,
                detail=f"{type(exc).__name__}: {message[:800]}",
            ) from None

        elapsed_seconds = time.perf_counter() - started
        output_bytes = len(markdown.encode("utf-8"))

        return {
            "status": "success",
            "parser_id": parser_id,
            "parser_name": parser["name"],
            "filename": filename,
            "format": extension,
            "input_bytes": total_bytes,
            "latency_seconds": round(elapsed_seconds, 4),
            "output_bytes": output_bytes,
            "output_characters": len(markdown),
            "output_markdown": markdown,
            "metrics_note": (
                "Latency measures parser execution, including cloud API "
                "upload and response time where applicable. It excludes "
                "the initial browser upload to this API."
            ),
        }

    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
