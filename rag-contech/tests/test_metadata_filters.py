import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest import extract_front_matter_metadata, infer_document_metadata
from query import build_where_clause


def test_extract_front_matter_metadata_reads_fields():
    text = """---
tool: Buildertrend
category: estimation
cost_band: medium
---
Contenu du document.
"""

    metadata = extract_front_matter_metadata(text)

    assert metadata["tool"] == "Buildertrend"
    assert metadata["category"] == "estimation"
    assert metadata["cost_band"] == "medium"


def test_infer_document_metadata_from_filename():
    metadata = infer_document_metadata("procore-estimating.md")

    assert metadata["tool"] == "Procore"
    assert metadata["category"] == "estimation"
    assert metadata["source"] == "procore-estimating.md"


def test_build_where_clause_uses_only_selected_filters():
    where = build_where_clause(
        source="buildertrend.pdf",
        tool="Buildertrend",
        category="estimation",
        cost="medium",
    )

    assert where == {
        "$and": [
            {"source": {"$eq": "buildertrend.pdf"}},
            {"tool": {"$eq": "Buildertrend"}},
            {"category": {"$eq": "estimation"}},
            {"cost_band": {"$eq": "medium"}},
        ]
    }
