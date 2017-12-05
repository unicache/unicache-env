import os
import datetime

from .request import Request

def inputIqiyi():
    try:
        with open(os.path.dirname(__file__) + '/raw/iqiyi.csv', encoding = 'gb18030') as f:
            return list(map(
                lambda row: Request(float(row[0]), float(row[1]), datetime.datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S"), row[4].strip(), row[3].strip()),
                map(lambda row: row.strip().split('|'), f)
            ))
    except FileNotFoundError:
        logger.error('Data file not found. This file may not in the git repo')

def inputMovieLens():
    try:
        with open(os.path.dirname(__file__) + '/raw/ratings.csv') as f:
            next(f)
            return list(sorted(map(
                lambda row: Request(None, None, datetime.datetime.fromtimestamp(int(row[3])), row[0], row[1]),
                map(lambda row: row.strip().split(','), f)
            ), key = lambda req: req.time))
    except FileNotFoundError:
        logger.error('Data file not found. This file may not in the git repo')

def inputDataset(dataset):
    ''' Input a speific dataset
        @param dataset : 'iqiyi' or 'movielens'
        @return : List of requests '''

    return {
        'iqiyi': inputIqiyi,
        'movielens': inputMovieLens
    }[dataset]()
    
