"""
EVA Symbolic — символьное ядро EVA (полное).

Многообразие символьных сборок + дуальный движок + генерация + мета-уровень.
21 модуль.
"""

from .char_vocab import CharacterVocab
from .potential_field import PotentialField
from .potential_dynamics import PotentialDynamics
from .assembly_graph import AssemblyState, AssemblyEdge, AssemblyNode
from .assembly_grammar import AssemblyGrammar, AssemblyPattern
from .semantic_closure import SemanticClosureChecker
from .continuation_propagator import ContinuationPropagator
from .logic_bridge import LogicBridge, TransformRule, TransformOp
from .assembly_validator import AssemblyValidator
from .assembly_explorer import AssemblyExplorer, ExplorationResult
from .sleep_mode_symbolic import SleepModeSymbolic
from .topological_field import TopologicalField, ManifoldPoint, LocalChart
from .natural_clusterer import NaturalClusterer, Domain
from .geodesic_navigator import GeodesicNavigator, TangentSpace, TangentVector, GeodesicPath, TangentDirection
from .curvature_analyzer import CurvatureAnalyzer
from .contradiction_filter import SymbolicContradictionFilter, ForbiddenConnection, ContradictionType
from .concept_miner import SymbolicConceptMiner, Concept
from .symbolic_generator import SymbolicGenerator
from .conditional_binding import TemporalConditionalBinding
from .assembly_template_bank import AssemblyTemplateBank, AssemblyTemplate
from .intrinsic_reward import IntrinsicReward, MetaPatterns, HierarchicalCompressor
from .attention_feedback import AttentionFeedback
from .advanced_modules import UncertaintyQuantifier, ActiveInference, CausalDiscovery
from .advanced_methods import NGramContext, DynamicVocab, PatternToConcept, MultiLevelPredictor
from .hierarchical_layer import WordToken, WordDiscovery, MultiLayerManifold, HierarchicalPredictor, LogicCompiler
from .word_level import WordBoundaryDetector, GrammaticalRoleDiscovery, SemanticClustering, WordLevelGenerator, SelfConsistencyCheck
from .knowledge_base import KnowledgeDomain, KnowledgeBase, CrossDomainNavigator, IntelligentContextRouter
from .library import DomainAutoNamer, DomainIndex, CatalogEntry, LibrarianMap, LibraryStats, LibraryManager
from .contemplation import LogicGuard, ContemplationLoop
from .manifold_attention import ManifoldAttention, MultiScaleAttentionStack, CoordinateProjector
from .parallel_trainer import ParallelSymbolicTrainer
from .potential_trainer import PotentialTrainer

__all__ = [
    "CharacterVocab", "PotentialField", "PotentialDynamics",
    "AssemblyState", "AssemblyEdge", "AssemblyNode",
    "AssemblyGrammar", "AssemblyPattern",
    "SemanticClosureChecker", "ContinuationPropagator",
    "LogicBridge", "TransformRule", "TransformOp",
    "AssemblyValidator", "AssemblyExplorer", "ExplorationResult",
    "SleepModeSymbolic",
    "TopologicalField", "ManifoldPoint", "LocalChart",
    "NaturalClusterer", "Domain",
    "GeodesicNavigator", "TangentSpace", "TangentVector", "GeodesicPath", "TangentDirection",
    "CurvatureAnalyzer",
    "SymbolicContradictionFilter", "ForbiddenConnection", "ContradictionType",
    "SymbolicConceptMiner", "Concept",
    "SymbolicGenerator",
    "TemporalConditionalBinding",
    "AssemblyTemplateBank", "AssemblyTemplate",
    "IntrinsicReward", "MetaPatterns", "HierarchicalCompressor",
    "AttentionFeedback",
    "UncertaintyQuantifier", "ActiveInference", "CausalDiscovery",
    "NGramContext", "DynamicVocab", "PatternToConcept", "MultiLevelPredictor",
    "WordToken", "WordDiscovery", "MultiLayerManifold", "HierarchicalPredictor", "LogicCompiler",
    "WordBoundaryDetector", "GrammaticalRoleDiscovery", "SemanticClustering",
    "WordLevelGenerator", "SelfConsistencyCheck",
    "KnowledgeDomain", "KnowledgeBase", "CrossDomainNavigator", "IntelligentContextRouter",
    "DomainAutoNamer", "DomainIndex", "CatalogEntry", "LibrarianMap", "LibraryStats", "LibraryManager",
    "LogicGuard", "ContemplationLoop",
    "ManifoldAttention", "MultiScaleAttentionStack", "CoordinateProjector",
    "ParallelSymbolicTrainer",
    "PotentialTrainer",
]
