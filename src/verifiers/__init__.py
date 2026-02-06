from .deepseek import DeepSeekWeakVerifier

try:
    from .math_shepherd import MathShepherdScorer
except ImportError:
    MathShepherdScorer = None