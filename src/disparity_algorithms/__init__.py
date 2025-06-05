from .base_disparity import BaseDisparityMatcher
from .block_matching import StereoBlockMatcher
from .sgbm import SemiGlobalBlockMatcher
from .census_cost_matching import CensusCostMatcher
from .cascade_matching import CascadeMatcher
from .efficient_large_scale_stereo import EfficientLargeScaleStereoMatcher
from .feature_based_matching import FeatureBasedMatcher
from .roi_disparity import ROIDisparity

# You can define a list of available algorithms here for easy access
AVAILABLE_ALGORITHMS = {
    "BlockMatching": StereoBlockMatcher,
    "SGBM": SemiGlobalBlockMatcher,
    "CensusCost": CensusCostMatcher,
    "Cascade": CascadeMatcher,
    "ELAS": EfficientLargeScaleStereoMatcher,
    "FeatureBased": FeatureBasedMatcher,
    "ROIDisparity": ROIDisparity,
}

__all__ = ['BaseDisparityMatcher', 'AVAILABLE_ALGORITHMS', 'ROIDisparity']