"""
Ingest Deep Research SOP Skills into Memory MCP

This script ingests all 9 Deep Research SOP skills including:
- baseline-replication
- method-development
- holistic-evaluation
- deep-research-orchestrator
- gate-validation
- literature-synthesis
- reproducibility-audit
- deployment-readiness
- research-publication

Plus their GraphViz process diagrams for complete knowledge representation.

This enables AI models to retrieve skill instructions, workflow patterns,
and integration guidance across sessions via Memory MCP persistence.
"""
import sys
import io
from pathlib import Path
from typing import List, Dict, Any
import json

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


def collect_skill_files() -> Dict[str, List[Path]]:
    """
    Collect all Deep Research SOP skill files.

    Returns:
        Dictionary mapping categories to lists of Path objects
    """
    base_dir = Path(__file__).parent.parent

    skills = {
        'skill_docs': [],
        'graphviz_diagrams': []
    }

    # Collect skill SKILL.md files
    skills_dir = base_dir / ".claude" / "skills"
    if skills_dir.exists():
        skill_names = [
            "baseline-replication",
            "method-development",
            "holistic-evaluation",
            "deep-research-orchestrator",
            "gate-validation",
            "literature-synthesis",
            "reproducibility-audit",
            "deployment-readiness",
            "research-publication"
        ]

        for skill_name in skill_names:
            skill_file = skills_dir / skill_name / "SKILL.md"
            if skill_file.exists():
                skills['skill_docs'].append(skill_file)

            # GraphViz diagrams
            graphviz_file = skills_dir / skill_name / f"{skill_name}-process.dot"
            if graphviz_file.exists():
                skills['graphviz_diagrams'].append(graphviz_file)

    print("📋 Found Deep Research SOP Skill Files:")
    for category, file_list in skills.items():
        if file_list:
            print(f"\n  {category.replace('_', ' ').title()}: {len(file_list)} files")
            for f in file_list:
                print(f"    • {f.name}")

    return skills


def ingest_file(
    file_path: Path,
    category: str,
    chunker: SemanticChunker,
    embedder: EmbeddingPipeline,
    indexer: VectorIndexer
) -> int:
    """
    Ingest a single skill file.

    Args:
        file_path: Path to skill file
        category: Category of the file (skill_docs, graphviz_diagrams)
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

    # Extract title
    title = file_path.stem.replace('-', ' ').title()

    # For SKILL.md files, extract name from YAML frontmatter
    if file_path.name == "SKILL.md":
        if content.startswith('---'):
            lines = content.split('\n')
            for line in lines[1:]:
                if line.startswith('name:'):
                    title = line.replace('name:', '').strip().strip('"')
                    break
                if line.startswith('---'):
                    break

    # Chunk the document
    # For skill files, split by headers and major sections
    if category == 'skill_docs':
        # Split by ## headers (major sections)
        sections = []
        current_section = []
        current_header = ""

        for line in content.split('\n'):
            if line.startswith('## ') and len(current_section) > 0:
                # Save previous section
                sections.append({
                    'header': current_header,
                    'content': '\n'.join(current_section)
                })
                current_section = []
                current_header = line.strip('# ').strip()
            current_section.append(line)

        # Add last section
        if current_section:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_section)
            })

        # Create chunks from sections
        chunks = []
        for idx, section in enumerate(sections):
            # Skip very short sections
            if len(section['content']) < 100:
                continue

            # Skip YAML frontmatter
            if section['content'].startswith('---') and idx == 0:
                continue

            metadata = {
                'title': title,
                'filename': file_path.name,
                'section': section['header'],
                'category': f'sop-skill-{category}',
                'source': 'deep_research_sop_skills',
                'ingestion_type': 'skill_documentation',
                'layer': 'long_term',  # Skills should be long-term
                'type': 'skill',
                'tags': 'SOP,Skill,Pipeline,Quality-Gate'
            }

            chunks.append({
                'text': section['content'],
                'file_path': str(file_path),
                'chunk_index': idx,
                'metadata': metadata
            })

    elif category == 'graphviz_diagrams':
        # For GraphViz, keep entire file as one chunk
        metadata = {
            'title': title,
            'filename': file_path.name,
            'category': f'sop-skill-{category}',
            'source': 'deep_research_sop_skills',
            'ingestion_type': 'graphviz_diagram',
            'layer': 'long_term',
            'type': 'diagram',
            'tags': 'SOP,Skill,GraphViz,Workflow'
        }

        chunks = [{
            'text': content,
            'file_path': str(file_path),
            'chunk_index': 0,
            'metadata': metadata
        }]

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


def ingest_all_skills():
    """
    Main function to ingest all Deep Research SOP skills.
    """
    print_header("DEEP RESEARCH SOP SKILLS - KNOWLEDGE BASE INGESTION")

    # Initialize memory components
    print("🔧 Initializing memory components...")
    chunker = SemanticChunker()
    embedder = EmbeddingPipeline()
    indexer = VectorIndexer(persist_directory="./chroma_data")
    indexer.create_collection()
    print("✅ Memory components initialized\n")

    # Collect files
    skill_files = collect_skill_files()

    # Track statistics
    stats = {
        'total_chunks': 0,
        'files_processed': 0,
        'categories': {}
    }

    # Process each category
    for category, file_list in skill_files.items():
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

    print("\n✅ Deep Research SOP Skills knowledge base successfully populated!")
    print(f"   Memory system now contains comprehensive knowledge about:")
    print(f"   - 9 Complete Skills (baseline-replication → research-publication)")
    print(f"   - 9 GraphViz workflow diagrams")
    print(f"   - Complete 3-phase, 9-pipeline, 3-gate research lifecycle")
    print(f"   - Integration with 4 P0 agents (data-steward, ethics-agent, archivist, evaluator)")

    print("\n🔍 Test retrieval with:")
    print('   python scripts/memory-search.py --query "How to replicate baselines?"')
    print('   python scripts/memory-search.py --query "Quality Gate 2 requirements"')
    print('   python scripts/memory-search.py --query "Deployment readiness checklist"')
    print('   python scripts/memory-search.py --query "Publication reproducibility checklist"')

    # Save statistics
    stats_file = Path("docs") / "skill_ingestion_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n📈 Statistics saved to: {stats_file}")

    return stats


if __name__ == "__main__":
    ingest_all_skills()
