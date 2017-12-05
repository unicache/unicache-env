from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (
    bytes, dict, int, list, object, range, str,
    ascii, chr, hex, input, next, oct, open,
    pow, round, super,
    filter, map, zip
) # Python2 support

import os
import sys
import gym
import math
import numpy
import pickle
import random
import bisect
import logging

from .request import Request
from .input_dataset import inputDataset

logger = logging.getLogger(__name__)
logger.setLevel('WARNING')

VERSION = 0.02 # Used to identify different version of .tmp file

class State:
    ''' State returned to users '''

    def __init__(self, storeSize, sampleSize):
        self.storeSize = storeSize
        self.cached = numpy.array([False] * sampleSize, dtype = bool) # Cached contents
        self.cachedNum = 0
        self.arriving = None # Newly arriving content ID
        self.history = [] # Requests handled

    def newReq(self, req):
        ''' [INTERNAL METHOD] Add the next request
            @return : bool. Whether need to evict a content from cache '''

        assert self.arriving is None
        self.history.append(req)
        if self.cached[req.content]:
            return False
        if self.cachedNum < self.storeSize:
            self.cached[req.content] = True
            self.cachedNum += 1
            return False
        self.arriving = req.content
        return True

    def evict(self, content):
        ''' [INTERNAL METHOD] Remove a content from the cache and accept the newly arriving request
            If you are end users, don't call this directly. Call IqiyiEnv.step instead '''

        assert self.cached[content]
        self.cached[content] = False
        self.cached[self.arriving] = True
        self.arriving = None

def dist(la1, lo1, la2, lo2):
    ''' Calculate distance from latitudes and longitudes IN BEIJING
        @return : Kilometer '''

    if la1 is None or lo1 is None or la2 is None or lo2 is None:
        logger.error("Current dataset doesn't have geographic data")
        exit(1)
    x = (la1 - la2) * 222
    y = (lo1 - lo2) * 85
    return math.sqrt(x * x + y * y)

def filterVersion(version):
    ''' Return a filter corresponding to version code '''

    LOCAL_MASK = 0x3
    LOCAL_FILTER = {
        0x0: lambda req: True, # Do nothing
        0x1: lambda req: dist(req.latitude, req.longitude, 39.976586, 116.317694) < 1.0, # 中关村
        0x2: lambda req: dist(req.latitude, req.longitude, 39.982440, 116.347954) < 1.0, # 北航
    }

    def f(req):
        return LOCAL_FILTER[version & LOCAL_MASK](req)
    return f

class Env(gym.Env):
    metadata = {
        'render.modes': []
    }

    def __init__(self, dataset, capDivCont, sampleSize, version):
        ''' Constructor. You have to determine these parameters via envoriment IDs
            @param dataset : 'iqiyi' or 'movielens'
            @param capDivCont : Storage size / total content number
            @param sampleSize : int. Randomly select this size of CONTENTS. `None` means using the whole set (iqiyi = 233045, movielens = 26744 contents)
            @param version: int. Version code. See filterVersion '''

        fakeSeed = random.randrange(sys.maxsize)
        try:
            with open(os.path.dirname(__file__) + '/.%s_%s_%s_%s_%s.tmp'%(dataset, capDivCont, sampleSize, fakeSeed, version), 'rb') as f:
                self.requests, self.sampleSize, _version = pickle.load(f)
                if _version != VERSION:
                    logger.info("Old cache found, will not use")
                    raise FileNotFoundError
                logger.info('Loading from cache')
        except:
            logger.info('Input cache not found, loading from raw input')
            self.requests = inputDataset(dataset)
            self.requests = filter(filterVersion(version), self.requests)
            self.requests = list(self.requests) # self.requests will be used twice, so can't be an iterator

            def unique(sequence):
                last = None
                for item in sequence:
                    if item != last:
                        last = item
                        yield item

            contents = list(unique(sorted(map(lambda r: r.content, self.requests))))
            logger.info('%d contents in total'%(len(contents)))
            if sampleSize is not None:
                if sampleSize > len(contents):
                    logger.warning('sampling size larger than total size')
                else:
                    contents = [contents[i] for i in sorted(random.sample(range(len(contents)), sampleSize))]
            for req in self.requests:
                pos = bisect.bisect_left(contents, req.content)
                req.content =  pos if pos < len(contents) and contents[pos] == req.content else None
            self.requests = list(filter(lambda r: r.content is not None, self.requests))
            self.sampleSize = len(contents) # Don't use parameter `sampleSize`, which can be None

            with open(os.path.dirname(__file__) + '/.%s_%s_%s_%s_%s.tmp'%(dataset, capDivCont, sampleSize, fakeSeed, version), 'wb') as f:
                pickle.dump((self.requests, self.sampleSize, VERSION), f)
                logger.info('Cached input')

        self.requestsIter = None
        self.state = None
        self.storeSize = int(capDivCont * self.sampleSize)
        self.done = True
        if self.storeSize == 0:
            logging.warning("Storage size = 0. Please increase capacity.")

    def _nextState(self):
        ''' Get next state which needs eviction
            @return Wheter episode ended, how many request hit '''

        oldCnt = len(self.state.history)
        try:
            while self.state.arriving is None:
                self.state.newReq(next(self.requestsIter))
        except StopIteration:
            return True, len(self.state.history) - oldCnt - 1
        return False, len(self.state.history) - oldCnt - 1

    def _reset(self):
        ''' Reset the environment
            @return : Initial state '''
        
        self.requestsIter = iter(self.requests)
        self.state = State(self.storeSize, self.sampleSize)
        self.done, hit = self._nextState()
        if self.done:
            logger.warning('All contents hit. Maybe storage capacity is too high. Returning None')
            return None
        return self.state

    def _step(self, action):
        ''' Perform an action
            @param action : Content ID. Which content to be evicted
            @return : next state, reward, wheter ended, extra info (not used yet) '''

        if self.state is None:
            logger.error('You should call `reset` before `step`')
            return
        if self.done:
            logger.warning('You are stepping after episode has been done')
        if not self.state.cached[action]:
            logger.error('Invalid action: evicting uncached content')
            return
        self.state.evict(action)
        self.done, hit = self._nextState()
        assert not self.done or len(self.state.history) == len(self.requests)
        return self.state, hit, self.done, {}

    def _render(self, mode = 'human', close = False):
        if close:
            return
        logger.warning('`render` is not implemented yet')

