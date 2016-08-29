# Calculate the number of time intervals in which there are a minimum number of
# visits
# ebellm@caltech.edu

import lsst.sims.maf.metrics as metrics
import numpy as np


class VisitsInIntervalMetric(metrics.BaseMetric):
    """
    Determine if there is at least one interval of intervalLength days with at
    least minVisits visits.
    Optionally, require at least nPairs pairs are separated by minPairGap

    Parameters
    ----------
    intervalLength : float
        Interval length in which to search for minVisits (days)
    minVisits : integer >= 2;
        Number of total visits required in specified interval
    nPairs : integer >= 0;
        Number of pairs required to be separated by deltaT.  Omit check if 0.
    minPairGap : float > 0.
        Mininum spacing in time between nPairs  (days)
    reduceFunc : function, optional
       Function that can operate on array-like structures. Typically numpy function.
       Default np.min.
    """

    def __init__(self, intervalLength, minVisits, nPairs=0, minPairGap=0.,
                 timeCol='expMJD',
                 metricName='VisitsInInterval', **kwargs):

        units = 'boolean'
        assert (minVisits >= 2)
        assert (nPairs >= 0)
        assert (minPairGap >= 0.)
        assert (nPairs <= (minVisits + 1))
        self.intervalLength = intervalLength
        self.minVisits = minVisits
        self.nPairs = nPairs
        self.minPairGap = minPairGap
        self.timeCol = timeCol
        super(VisitsInIntervalMetric, self).__init__(col=[self.timeCol],
                                                     units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """Calculate whether or not there are >=minVisits in intervalLength
        (optionally requiring that nPairs be separated by at least minPairGap)
        Parameters
        ----------
        dataSlice : numpy.array
            Numpy structured array containing the data related to the visits provided by the slicer.
        slicePoint : dict, optional
            Dictionary containing information about the slicepoint currently active in the slicer.
        Returns
        -------
        float
           1 or 0 if the visits supply the requested pattern.
        """

        # comment below out to test
        dataSlice.sort(order=self.timeCol)

        # nested loops will be slow but effective.  Break out of the loop once
        # we have one true case.
        t = dataSlice[self.timeCol]
        for ti0 in t:
            timax = ti0 + self.intervalLength
            w = (t >= ti0) & (t <= timax)
            if np.sum(w) >= self.minVisits:
                if (self.nPairs == 0) or (self.minPairGap == 0):
                    # if not checking pair gaps, break once we find one
                    # interval with minVisits
                    return True
                else:
                    # run check for minPairGap.
                    tw = t[w]
                    # construct all pairwise time differences.  Could
                    # be a memory hog if we have lots of points
                    x = np.resize(tw, (len(tw), len(tw)))
                    dt = x - x.T
                    # walk row by row to find pairs separated by >= minPairGap
                    pairStart = 0
                    nPairsFound = 0
                    for i in range(len(tw)):
                        if i < pairStart:
                            continue
                        # should always be starting on the diagonal
                        assert (dt[i, pairStart] == 0)
                        row = dt[i, pairStart:]
                        wdt = np.where(row >= self.minPairGap)[0]
                        if len(wdt) > 0:
                            nPairsFound += 1
                            pairStart += wdt[0]
                            # break out once we've found enough pairs
                            if nPairsFound >= self.nPairs:
                                return True
                        else:
                            # increment to stay on the diagonal
                            pairStart += 1
            return False

        return result


def test():
    m1 = VisitsInIntervalMetric(45, 3, nPairs=0, minPairGap=0.)
    m2 = VisitsInIntervalMetric(45, 3, nPairs=2, minPairGap=2.)

    test_runs = [
        {'expMJD': np.array([0, 1, 2, 5, 40]), 'm1':True, 'm2':True},
        {'expMJD': np.array([0, 2, 45]), 'm1':True, 'm2':True},
        {'expMJD': np.array([0, 1, 43, 45]), 'm1':True, 'm2':True},
        {'expMJD': np.array([0, 1, 2, 3, 4, 5, 45]), 'm1':True, 'm2':True},
        {'expMJD': np.array([0, 1, 45]), 'm1':True, 'm2':False},
        {'expMJD': np.array([0, 44, 45]), 'm1':True, 'm2':False}]

    for run in test_runs:
        assert(m1.run(run) == run['m1'])
        assert(m2.run(run) == run['m2'])
