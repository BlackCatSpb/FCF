"""
EVA Symbolic — символьное ядро EVA.

Активные модули (используются train_* скриптами).
Архивированные модули — в eva/archive/.
"""

from .char_vocab import CharacterVocab
from .potential_field import PotentialField
from .potential_dynamics import PotentialDynamics
from .potential_function import PotentialFunction
from .topological_field import TopologicalField, ManifoldPoint, LocalChart
from .natural_clusterer import NaturalClusterer, Domain
from .geodesic_navigator import GeodesicNavigator, TangentSpace, TangentVector, GeodesicPath, TangentDirection
from .curvature_analyzer import CurvatureAnalyzer
from .contradiction_filter import SymbolicContradictionFilter, ForbiddenConnection, ContradictionType
from .concept_miner import SymbolicConceptMiner, Concept
from .symbolic_generator import SymbolicGenerator
from .conditional_binding import TemporalConditionalBinding
from .sleep_mode_symbolic import SleepModeSymbolic
from .advanced_methods import NGramContext, DynamicVocab, PatternToConcept, MultiLevelPredictor
from .hierarchical_layer import WordToken, WordDiscovery, MultiLayerManifold, HierarchicalPredictor, LogicCompiler
from .word_level import WordBoundaryDetector, GrammaticalRoleDiscovery, SemanticClustering, WordLevelGenerator, SelfConsistencyCheck
from .knowledge_base import KnowledgeDomain, KnowledgeBase, CrossDomainNavigator, IntelligentContextRouter
from .library import DomainAutoNamer, DomainIndex, CatalogEntry, LibrarianMap, LibraryStats, LibraryManager
from .contemplation import LogicGuard, ContemplationLoop
from .potential_trainer import PotentialTrainer
