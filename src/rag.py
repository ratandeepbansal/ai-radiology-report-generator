"""
RAG (Retrieval-Augmented Generation) Module for MedAssist Copilot
Implements vector database for semantic similarity search of patient reports
Supports ChromaDB and FAISS
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system using vector database
    for semantic similarity search of patient reports
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        vector_db_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        reports_path: Optional[str] = None
    ):
        """
        Initialize the RAG system

        Args:
            embedding_model: Name of the embedding model
            vector_db_type: Type of vector database ('chromadb' or 'faiss')
            collection_name: Name of the collection/index
            reports_path: Path to patient reports JSON file
        """
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL_NAME
        self.vector_db_type = vector_db_type or config.VECTOR_DB_TYPE
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.reports_path = reports_path or config.PATIENT_REPORTS_PATH

        logger.info(f"Initializing RAG System...")
        logger.info(f"  Embedding model: {self.embedding_model_name}")
        logger.info(f"  Vector DB: {self.vector_db_type}")

        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info(f"✅ Embedding model loaded")

        # Initialize vector database
        self.vector_db = None
        self.collection = None
        self._initialize_vector_db()

        # Load reports
        self.reports = []
        self.report_embeddings = []
        self.load_reports()

        # Statistics
        self.stats = {
            'total_reports': 0,
            'total_queries': 0,
            'average_query_time': 0.0,
            'cache_hits': 0
        }

        # Query cache
        self.query_cache = {}

    def _initialize_vector_db(self):
        """Initialize the vector database"""
        try:
            if self.vector_db_type == "chromadb":
                import chromadb
                from chromadb.config import Settings

                # Create ChromaDB client
                self.vector_db = chromadb.Client(
                    Settings(
                        persist_directory="./chromadb",
                        anonymized_telemetry=False
                    )
                )

                # Get or create collection
                try:
                    self.collection = self.vector_db.get_collection(self.collection_name)
                    logger.info(f"✅ Loaded existing collection: {self.collection_name}")
                except:
                    self.collection = self.vector_db.create_collection(
                        name=self.collection_name,
                        metadata={"description": "Patient radiology reports"}
                    )
                    logger.info(f"✅ Created new collection: {self.collection_name}")

            elif self.vector_db_type == "faiss":
                import faiss

                self.vector_db = faiss
                self.collection = None  # Will be created when reports are loaded
                logger.info("✅ FAISS initialized")

            else:
                raise ValueError(f"Unsupported vector DB type: {self.vector_db_type}")

        except ImportError as e:
            logger.error(f"Failed to import {self.vector_db_type}: {str(e)}")
            logger.info("Install with: pip install chromadb  OR  pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise

    def load_reports(self) -> bool:
        """
        Load patient reports and create embeddings

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load reports from JSON
            if not os.path.exists(self.reports_path):
                logger.warning(f"Reports file not found: {self.reports_path}")
                return False

            with open(self.reports_path, 'r', encoding='utf-8') as f:
                self.reports = json.load(f)

            logger.info(f"Loaded {len(self.reports)} reports")

            # Check if already embedded in ChromaDB
            if self.vector_db_type == "chromadb":
                existing_count = self.collection.count()
                if existing_count > 0:
                    logger.info(f"Found {existing_count} existing embeddings in ChromaDB")
                    self.stats['total_reports'] = existing_count
                    return True

            # Generate embeddings
            logger.info("Generating embeddings for reports...")
            start_time = time.time()

            documents = []
            metadatas = []
            ids = []

            for i, report in enumerate(self.reports):
                # Create document text from report
                doc_text = self._create_document_text(report)
                documents.append(doc_text)

                # Create metadata
                metadata = {
                    'patient_id': report['patient_id'],
                    'date': report['date'],
                    'age': str(report.get('metadata', {}).get('age', '')),
                    'gender': report.get('metadata', {}).get('gender', ''),
                }
                metadatas.append(metadata)

                # Create ID
                ids.append(f"report_{i}")

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            embedding_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")

            # Store in vector database
            if self.vector_db_type == "chromadb":
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info("✅ Stored embeddings in ChromaDB")

            elif self.vector_db_type == "faiss":
                # Create FAISS index
                dimension = embeddings.shape[1]
                self.collection = self.vector_db.IndexFlatL2(dimension)
                self.collection.add(embeddings)
                self.report_embeddings = embeddings
                logger.info("✅ Created FAISS index")

            self.stats['total_reports'] = len(self.reports)
            return True

        except Exception as e:
            logger.error(f"Error loading reports: {str(e)}")
            return False

    def _create_document_text(self, report: Dict[str, Any]) -> str:
        """
        Create searchable text from a report

        Args:
            report: Report dictionary

        Returns:
            Combined text for embedding
        """
        parts = []

        # Add findings
        if 'report' in report and 'findings' in report['report']:
            parts.append(f"Findings: {report['report']['findings']}")

        # Add impression
        if 'report' in report and 'impression' in report['report']:
            parts.append(f"Impression: {report['report']['impression']}")

        # Add metadata
        if 'metadata' in report:
            meta = report['metadata']
            if 'age' in meta:
                parts.append(f"Age: {meta['age']}")
            if 'gender' in meta:
                parts.append(f"Gender: {meta['gender']}")
            if 'indication' in meta:
                parts.append(f"Indication: {meta['indication']}")

        return " | ".join(parts)

    def search_by_similarity(
        self,
        query: str,
        top_k: int = 3,
        patient_id: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar reports using semantic similarity

        Args:
            query: Query text (e.g., findings from current X-ray)
            top_k: Number of similar reports to return
            patient_id: Optional patient ID to filter results
            use_cache: Whether to use query cache

        Returns:
            List of similar reports with scores
        """
        # Check cache
        cache_key = f"{query}_{top_k}_{patient_id}"
        if use_cache and cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            return self.query_cache[cache_key]

        try:
            start_time = time.time()
            self.stats['total_queries'] += 1

            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            )

            results = []

            if self.vector_db_type == "chromadb":
                # Query ChromaDB
                search_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k * 2 if patient_id else top_k,  # Get more if filtering
                    include=['metadatas', 'documents', 'distances']
                )

                # Process results
                for i in range(len(search_results['ids'][0])):
                    doc_id = search_results['ids'][0][i]
                    metadata = search_results['metadatas'][0][i]
                    document = search_results['documents'][0][i]
                    distance = search_results['distances'][0][i]

                    # Filter by patient_id if provided
                    if patient_id and metadata.get('patient_id') != patient_id:
                        continue

                    # Convert distance to similarity score (0-1)
                    similarity = 1 / (1 + distance)

                    # Find original report
                    report_idx = int(doc_id.split('_')[1])
                    if report_idx < len(self.reports):
                        result = {
                            'report': self.reports[report_idx],
                            'similarity': similarity,
                            'distance': distance,
                            'document': document,
                            'metadata': metadata
                        }
                        results.append(result)

                        if len(results) >= top_k:
                            break

            elif self.vector_db_type == "faiss":
                # Query FAISS
                query_embedding = query_embedding.reshape(1, -1)
                distances, indices = self.collection.search(query_embedding, top_k * 2)

                # Process results
                for i, idx in enumerate(indices[0]):
                    if idx == -1:  # No more results
                        break

                    distance = distances[0][i]
                    similarity = 1 / (1 + distance)

                    # Get report
                    if idx < len(self.reports):
                        report = self.reports[idx]

                        # Filter by patient_id if provided
                        if patient_id and report.get('patient_id') != patient_id:
                            continue

                        result = {
                            'report': report,
                            'similarity': similarity,
                            'distance': float(distance),
                            'document': self._create_document_text(report)
                        }
                        results.append(result)

                        if len(results) >= top_k:
                            break

            query_time = time.time() - start_time
            self.stats['average_query_time'] = (
                (self.stats['average_query_time'] * (self.stats['total_queries'] - 1) + query_time)
                / self.stats['total_queries']
            )

            logger.info(f"Found {len(results)} similar reports in {query_time:.3f}s")

            # Cache results
            if use_cache:
                self.query_cache[cache_key] = results

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def search_by_patient(
        self,
        patient_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get all reports for a specific patient

        Args:
            patient_id: Patient identifier
            top_k: Maximum number of reports to return

        Returns:
            List of patient's reports sorted by date (most recent first)
        """
        patient_reports = [
            r for r in self.reports
            if r.get('patient_id') == patient_id
        ]

        # Sort by date (most recent first)
        patient_reports.sort(
            key=lambda x: x.get('date', ''),
            reverse=True
        )

        return patient_reports[:top_k]

    def get_relevant_context(
        self,
        query: str,
        patient_id: Optional[str] = None,
        max_context_length: int = 500
    ) -> str:
        """
        Get relevant context for report generation

        Args:
            query: Current findings or query
            patient_id: Optional patient ID
            max_context_length: Maximum length of context string

        Returns:
            Formatted context string
        """
        # Search for similar reports
        similar_reports = self.search_by_similarity(
            query=query,
            top_k=config.RAG_TOP_K,
            patient_id=patient_id
        )

        if not similar_reports:
            return "No relevant prior reports found."

        # Format context
        context_parts = []

        for i, result in enumerate(similar_reports, 1):
            report = result['report']
            similarity = result['similarity']

            # Only include highly relevant reports
            if similarity < config.SIMILARITY_THRESHOLD:
                continue

            context = f"Prior Report {i} (similarity: {similarity:.2f}):\n"
            context += f"Date: {report['date']}\n"
            context += f"Findings: {report['report']['findings'][:200]}...\n"
            context += f"Impression: {report['report']['impression'][:100]}...\n"

            context_parts.append(context)

            # Check length
            current_length = sum(len(p) for p in context_parts)
            if current_length >= max_context_length:
                break

        if not context_parts:
            return "No highly relevant prior reports found."

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return self.stats.copy()

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")


# Utility functions

def quick_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Quick function to search for similar reports

    Args:
        query: Query text
        top_k: Number of results

    Returns:
        List of similar reports
    """
    rag = RAGSystem()
    return rag.search_by_similarity(query, top_k=top_k)


# Main execution for testing
if __name__ == "__main__":
    print("=" * 70)
    print("MedAssist Copilot - RAG System Test")
    print("=" * 70)

    # Initialize RAG system
    print("\n1️⃣  Initializing RAG System...")
    try:
        rag = RAGSystem()
        print("   ✅ RAG system initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {str(e)}")
        exit(1)

    # Test similarity search
    print("\n2️⃣  Testing similarity search...")

    test_queries = [
        "chest X-ray showing pneumonia with consolidation",
        "normal chest X-ray with clear lung fields",
        "pleural effusion on the left side",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        results = rag.search_by_similarity(query, top_k=2)

        if results:
            print(f"   Found {len(results)} similar reports:")
            for j, result in enumerate(results, 1):
                report = result['report']
                print(f"\n   Result {j}:")
                print(f"     Patient: {report['patient_id']}")
                print(f"     Date: {report['date']}")
                print(f"     Similarity: {result['similarity']:.3f}")
                print(f"     Impression: {report['report']['impression'][:80]}...")
        else:
            print("   No results found")

    # Test patient-specific search
    print("\n3️⃣  Testing patient-specific search...")
    test_patient = "P001"
    print(f"   Searching for patient: {test_patient}")

    patient_reports = rag.search_by_patient(test_patient)
    if patient_reports:
        print(f"   Found {len(patient_reports)} report(s)")
        for report in patient_reports:
            print(f"     - Date: {report['date']}")
    else:
        print("   No reports found for patient")

    # Test context generation
    print("\n4️⃣  Testing context generation...")
    context = rag.get_relevant_context(
        query="chest X-ray with possible infection",
        patient_id="P002"
    )
    print(f"   Generated context ({len(context)} chars):")
    print(f"   {context[:200]}...")

    # Show statistics
    print("\n5️⃣  RAG Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("RAG system test complete!")
    print("=" * 70)
