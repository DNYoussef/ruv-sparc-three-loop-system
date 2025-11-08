"""
Ingest Deep Research SOP Documentation into Memory MCP

This script ingests all Deep Research SOP documentation including:
- 4 P0 commands (/init-datasheet, /prisma-init, /assess-risks, /init-model-card)
- 4 P0 agents (data-steward, ethics-agent, archivist, evaluator)
- Gap analysis document
- Graphviz diagrams (8 total)
- Updated CLAUDE.md with SOP triggers

This enables AI models to retrieve information about the Deep Research SOP,
Quality Gate system, and research lifecycle management.
"""
import sys
import io
from pathlib import Path
from typing import List, Dict, Any

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add memory-mcp-triple-system to path
memory_mcp_path = Path(__file__).parent.parent / "Desktop" / "memory-mcp-triple-system"
sys.path.insert(0, str(memory_mcp_path))

from src.chunking.semantic_chunker import SemanticChunker
from src.indexing.embedding_pipeline import EmbeddingPipeline
from src.indexing.vector_indexer import VectorIndexer


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def collect_sop_files() -> Dict[str, List[Path]]:
    """
    Collect all SOP documentation files to ingest.

    Returns:
        Dictionary mapping categories to lists of Path objects
    """
    base_dir = Path(__file__).parent.parent

    files = {
        'commands': [],
        'agents': [],
        'gap_analysis': [],
        'graphviz_diagrams': [],
        'claude_md': []
    }

    # Collect command files
    commands_dir = base_dir / ".claude" / "commands" / "research"
    if commands_dir.exists():
        files['commands'] = list(commands_dir.glob("*.md"))

    # Collect agent files
    agents_dir = base_dir / ".claude" / "agents" / "research"
    if agents_dir.exists():
        files['agents'] = list(agents_dir.glob("*.md"))

    # Gap analysis document
    gap_analysis_file = base_dir / "docs" / "deep-research-sop-gap-analysis.md"
    if gap_analysis_file.exists():
        files['gap_analysis'] = [gap_analysis_file]

    # Graphviz diagrams
    graphviz_base = base_dir / "docs" / "12fa" / "graphviz"
    if graphviz_base.exists():
        files['graphviz_diagrams'].extend((graphviz_base / "commands").glob("*.dot"))
        files['graphviz_diagrams'].extend((graphviz_base / "agents").glob("*.dot"))

    # CLAUDE.md
    claude_md = base_dir / "CLAUDE.md"
    if claude_md.exists():
        files['claude_md'] = [claude_md]

    print("📋 Found SOP Documentation Files:")
    for category, file_list in files.items():
        if file_list:
            print(f"\n  {category.replace('_', ' ').title()}: {len(file_list)} files")
            for f in file_list:
                print(f"    • {f.name}")

    return files


def ingest_file(
    file_path: Path,
    category: str,
    chunker: SemanticChunker,
    embedder: EmbeddingPipeline,
    indexer: VectorIndexer
) -> int:
    """
    Ingest a single SOP documentation file.

    Args:
        file_path: Path to documentation file
        category: Category of the file (commands, agents, etc.)
        chunker: SemanticChunker instance
        embedder: EmbeddingPipeline instance
        indexer: VectorIndexer instance

    Returns:
        Number of chunks created
    """
    print(f"  Processing: {file_path.name}")

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract title from first # header or filename
    title = file_path.stem.replace('-', ' ').title()
    if content.startswith('# '):
        first_line = content.split('\n')[0]
        title = first_line.replace('# ', '').strip()
    elif content.startswith('---'):
        # Extract from YAML frontmatter
        lines = content.split('\n')
        for line in lines[1:]:
            if line.startswith('name:'):
                title = line.replace('name:', '').strip().strip('"')
                break
            if line.startswith('---'):
                break

    # Chunk the document
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    chunks = []
    for idx, para in enumerate(paragraphs):
        # Skip very short paragraphs
        if len(para) < 50:
            continue

        # Skip lines that are just headers (single line starting with #)
        if para.startswith('#') and '\n' not in para:
            continue

        # Skip YAML frontmatter
        if para.startswith('---') and idx == 0:
            continue

        # Determine metadata based on category
        metadata = {
            'title': title,
            'filename': file_path.name,
            'category': f'sop-{category}',
            'source': 'deep_research_sop',
            'ingestion_type': 'sop_documentation',
            'layer': 'long_term'  # SOP docs should be long-term
        }

        # Add specific tags based on file type (convert to comma-separated string for ChromaDB)
        if category == 'commands':
            metadata['type'] = 'command'
            metadata['tags'] = 'SOP,Pipeline,Quality-Gate'
        elif category == 'agents':
            metadata['type'] = 'agent'
            metadata['tags'] = 'SOP,Agent,Quality-Gate'
        elif category == 'gap_analysis':
            metadata['type'] = 'analysis'
            metadata['tags'] = 'SOP,Gap-Analysis,Planning'
        elif category == 'graphviz_diagrams':
            metadata['type'] = 'diagram'
            metadata['tags'] = 'SOP,Graphviz,Visualization'
        elif category == 'claude_md':
            metadata['type'] = 'configuration'
            metadata['tags'] = 'SOP,Configuration,Triggers'

        chunks.append({
            'text': para,
            'file_path': str(file_path),
            'chunk_index': idx,
            'metadata': metadata
        })

    if not chunks:
        print(f"    ⚠️  WARNING: No chunks created from {file_path.name}")
        return 0

    print(f"    Chunks: {len(chunks)}")

    # Generate embeddings
    chunk_texts = [c['text'] for c in chunks]
    embeddings = embedder.encode(chunk_texts)
    print(f"    Embeddings: {len(embeddings)} generated")

    # Index in ChromaDB
    indexer.index_chunks(chunks, embeddings.tolist())
    print(f"    ✅ Indexed {len(chunks)} chunks")

    return len(chunks)


def ingest_all_sop_documentation():
    """
    Main function to ingest all SOP documentation.
    """
    print_header("DEEP RESEARCH SOP - KNOWLEDGE BASE INGESTION")

    # Initialize memory components
    print("🔧 Initializing memory components...")
    chunker = SemanticChunker()
    embedder = EmbeddingPipeline()
    indexer = VectorIndexer(persist_directory="./chroma_data")
    indexer.create_collection()
    print("✅ Memory components initialized\n")

    # Collect files
    sop_files = collect_sop_files()

    # Track statistics
    stats = {
        'total_chunks': 0,
        'files_processed': 0,
        'categories': {}
    }

    # Process each category
    for category, file_list in sop_files.items():
        if not file_list:
            continue

        print_header(f"Ingesting {category.replace('_', ' ').title()}")

        category_chunks = 0
        for file_path in file_list:
            try:
                chunks = ingest_file(file_path, category, chunker, embedder, indexer)
                category_chunks += chunks
                stats['files_processed'] += 1
            except Exception as e:
                print(f"    ❌ Error processing {file_path.name}: {e}")

        stats['categories'][category] = category_chunks
        stats['total_chunks'] += category_chunks
        print(f"\n✅ {category.replace('_', ' ').title()}: {category_chunks} chunks indexed")

    # Print summary
    print_header("INGESTION COMPLETE")
    print(f"📊 Statistics:")
    print(f"   Total Chunks: {stats['total_chunks']}")
    print(f"   Files Processed: {stats['files_processed']}\n")
    print(f"📂 Categories:")
    for category, count in stats['categories'].items():
        print(f"   {category.replace('_', '-')}: {count} chunks")

    print("\n✅ Deep Research SOP knowledge base successfully populated!")
    print(f"   Memory system now contains comprehensive knowledge about:")
    print(f"   - 4 P0 Commands (/init-datasheet, /prisma-init, /assess-risks, /init-model-card)")
    print(f"   - 4 P0 Agents (data-steward, ethics-agent, archivist, evaluator)")
    print(f"   - Gap analysis and implementation roadmap")
    print(f"   - 8 Graphviz workflow diagrams")
    print(f"   - SOP trigger patterns from CLAUDE.md")

    print("\n🔍 Test retrieval with:")
    print('   python scripts/memory-search.py --query "How does data-steward agent work?"')
    print('   python scripts/memory-search.py --query "What is Quality Gate 1?"')
    print('   python scripts/memory-search.py --query "How to assess risks?"')

    return stats


if __name__ == "__main__":
    ingest_all_sop_documentation()
