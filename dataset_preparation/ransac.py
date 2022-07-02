import numpy
from typing import NamedTuple, Optional, Tuple

RansacOptions = NamedTuple(
    "RansacOptions",
    [
        ("min_iterations", int),
        ("max_iterations", int),
        ("stopping_threshold", float),
        ("min_inliers", int),
        ("inlier_threshold", float),
        ("samples_per_iteration", int)])


class RansacBase(object):

    def run(self, options:RansacOptions) -> Optional[numpy.ndarray]:
        i = 0
        bestfit = None
        besterr = numpy.inf
        while i < options.min_iterations or (i < options.max_iterations and besterr > options.stopping_threshold):
            maybe_idxs, test_idxs = self.generate_sample_indices(options.samples_per_iteration)
            maybe_model = self.fit_model(maybe_idxs)
            test_err = self.test_points(test_idxs, maybe_model)
            also_idxs = None
            betterdata = maybe_idxs
            if test_idxs.size > 0:
                also_idxs = test_idxs[test_err < options.inlier_threshold]
                betterdata = numpy.concatenate((maybe_idxs, also_idxs))
            if also_idxs is None or len(also_idxs) >= options.min_inliers:
                bettermodel = self.fit_model(betterdata)
                better_errs = self.test_points(betterdata, bettermodel)
                thiserr = numpy.mean(better_errs)
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
            i +=1
        if bestfit is None:
            raise ValueError("did not meet fit acceptance criteria")
        return bestfit

    def generate_sample_indices(self, sample_count:int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()

    def test_points(self, test_points:numpy.ndarray, maybe_model:numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    def fit_model(self, candidates:numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()